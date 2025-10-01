package com.app.backend.trade.lstm;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.ta4j.core.BarSeries;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static com.app.backend.trade.strategy.StrategieBackTest.FEE_PCT;
import static com.app.backend.trade.strategy.StrategieBackTest.SLIP_PAGE_PCT;

/**
 * LstmTuningService
 *
 * Objectif global:
 *  - Orchestrer le tuning (recherche d'hyperparamètres) de modèles LSTM pour différents symboles (actifs boursiers / crypto).
 *  - Gérer le parallélisme de manière adaptative (CPU vs GPU) et surveiller la mémoire pour éviter les OutOfMemoryError.
 *  - Évaluer chaque configuration via un pipeline d'entraînement + walk-forward + agrégation de m��triques trading.
 *  - Persister:
 *      * les métriques de chaque configuration testée
 *      * la meilleure configuration sélectionnée
 *      * le modèle (poids) associé pour usage ultérieur (prédiction).
 *
 * Points clés à comprendre pour un débutant:
 *  1. Hyperparamètres = réglages (ex: windowSize, lstmNeurons, learningRate).
 *  2. On teste une grille (grid) de configurations (ou génération aléatoire).
 *  3. Pour chaque configuration:
 *       - entraînement LSTM
 *       - évaluation walk-forward (découpage chronologique en segments successifs)
 *       - calcul de métriques business (profitFactor, winRate, drawdown, expectancy, etc.)
 *  4. Sélection du "best model" basée ici sur un businessScore (et non uniquement MSE).
 *  5. Sauvegarde en base: hyperparams + métriques + modèle sérialisé.
 *  6. Protection mémoire: avant de lancer (ou entre étapes) on appelle waitForMemory().
 *  7. Suivi de progression: structure TuningProgress + logs heartbeat périodiques.
 *
 * Ne surtout pas modifier la logique interne sans tests, car:
 *  - Dépendances implicites entre trainLstmScalarV2, walkForwardEvaluate et saveModelToDb.
 *  - Certains resets mémoire ND4J (invokeGc / destroyAllWorkspacesForCurrentThread) sont critiques pour la stabilité.
 *
 * Extension future possible (guides):
 *  - Ajouter une stratégie de scoring différente (ex: pondérer profitFactor et drawdown), voir computeBusinessScoreV2.
 *  - Exposer un endpoint REST pour annuler un tuning en cours (nécessiterait un contrôle d'interruption).
 *  - Ajouter un mode "early stop global" si X configs consécutives échouent.
 */
@Service
public class LstmTuningService {
    private static final Logger logger = LoggerFactory.getLogger(LstmTuningService.class);

    // Dépendances injectées (services métiers)
    public final LstmTradePredictor lstmTradePredictor;      // Service d'entraînement / prédiction LSTM
    public final LstmHyperparamsRepository hyperparamsRepository; // Accès persistence hyperparamètres + métriques

    // --- Paramétrage dynamique du parallélisme ---
    @Value("${lstm.tuning.maxThreads:0}")
    private int configuredMaxThreads; // Si 0 => auto-calcul
    private int effectiveMaxThreads = 12; // Valeur de secours si auto-calcul échoue
    private boolean cudaBackend = false;  // Flag détecté au démarrage (backend ND4J CUDA ou non)

    /**
     * Permet (ex: via un endpoint admin) de forcer dynamiquement le nombre max de threads de tuning.
     * Thread-safe: synchronized pour éviter des recalculs simultanés incohérents.
     */
    public synchronized void setMaxParallelTuningThreads(int maxThreads){
        this.configuredMaxThreads = maxThreads;
        computeEffectiveMaxThreads();
    }

    /** Lecture du parallélisme effectif retenu (utilisé pour dimensionner les pools). */
    public int getEffectiveMaxThreads(){ return effectiveMaxThreads; }

    /**
     * Calcule le niveau de parallélisme effectif en fonction:
     *  - de la configuration explicite (configuredMaxThreads)
     *  - des coeurs CPU disponibles
     *  - de la présence d'un backend CUDA (réduction pour limiter surcharge CPU)
     *  - d'une limite "sécurité" (cap à 8 par défaut)
     *
     * Important: ne pas changer la logique => impacts potentiels sur la charge système.
     */
    private synchronized void computeEffectiveMaxThreads(){
        int procs = Runtime.getRuntime().availableProcessors();
        int base;
        if(configuredMaxThreads > 0){
            base = configuredMaxThreads; // L'utilisateur a forcé une valeur
        } else {
            // Heuristique: garder 1 coeur libre pour le système
            base = Math.max(1, procs - 1);
        }
        if(cudaBackend){
            // Quand GPU présent: on réduit la contention CPU => moitié des coeurs, max 4
            int gpuCap = Math.max(1, Math.min(4, procs / 2 == 0 ? 1 : procs / 2));
            base = Math.min(base, gpuCap);
        }
        // Limite mémoire/robustesse: ne pas dépasser 8 threads (éviter OOM)
        base = Math.max(1, Math.min(base, 8));
        this.effectiveMaxThreads = base;
        logger.info("[TUNING] Parallélisme effectif (auto) = {} (configuré={}, cpuCores={}, cuda={})",
                effectiveMaxThreads, configuredMaxThreads, procs, cudaBackend);
    }

    /**
     * Structure de suivi d'avancement pour un symbole en cours de tuning.
     * Utilisée pour:
     *  - Reporting temps réel (heartbeat logs)
     *  - APIs éventuelles de monitoring
     */
    public static class TuningProgress {
        public String symbol;                 // Symbole concerné (ex: BTCUSDT)
        public int totalConfigs;              // Nombre total de configurations prévues
        public AtomicInteger testedConfigs = new AtomicInteger(0); // Compteur thread-safe des configs terminées
        public String status = "en_cours";    // États possibles: en_cours | termine | failed | erreur
        public long startTime;                // Timestamp début
        public long endTime;                  // Timestamp fin (si terminé/échoué)
        public long lastUpdate;               // Dernière mise à jour (permet de détecter une éventuelle stagnation)
    }

    private final ConcurrentHashMap<String, TuningProgress> tuningProgressMap = new ConcurrentHashMap<>();
    private ScheduledExecutorService progressLoggerExecutor; // Planifie le heartbeat p��riodique

    /**
     * Méthode de cycle de vie Spring appelée après l'injection des dépendances.
     * Rôles:
     *  - Détection backend ND4J (CUDA ou CPU)
     *  - Initialisation du parallélisme effectif
     *  - Démarrage d'une tâche planifiée pour logguer périodiquement l'avancement des tunings
     *
     * Ne pas retirer les blocs try/finally: robustesse critique.
     */
    @PostConstruct
    public void initNd4jCuda() {
        try {
            System.setProperty("org.deeplearning4j.cudnn.enabled", "true"); // Active cuDNN si dispo (optimisations)
            org.nd4j.linalg.factory.Nd4jBackend backend = org.nd4j.linalg.factory.Nd4j.getBackend();
            String backendName = backend.getClass().getSimpleName();
            logger.info("ND4J backend utilisé : {}", backendName);
            cudaBackend = backendName.toLowerCase().contains("cuda") || backendName.toLowerCase().contains("jcublas");
            if (!cudaBackend) {
                logger.warn("Le backend ND4J n'est pas CUDA. Utilisation CPU uniquement.");
                logger.warn("Pour forcer CUDA : -Dorg.nd4j.linalg.defaultbackend=org.nd4j.linalg.jcublas.JCublasBackend");
            } else {
                logger.info("Backend CUDA détecté (optimization parallélisme ajustée).");
            }
        } catch (Exception e) {
            logger.error("Erreur lors de la détection du backend ND4J : {}", e.getMessage());
        } finally {
            // Toujours recalculer (même si échec)
            computeEffectiveMaxThreads();
        }

        // Heartbeat: toutes les 30 secondes, tant qu'il y a des tunings actifs
        progressLoggerExecutor = Executors.newSingleThreadScheduledExecutor();
        progressLoggerExecutor.scheduleAtFixedRate(() -> {
            try {
                if (tuningProgressMap.isEmpty()) return;
                long now = System.currentTimeMillis();
                tuningProgressMap.forEach((sym, p) -> {
                    double pct = p.totalConfigs > 0 ? (100.0 * p.testedConfigs.get() / p.totalConfigs) : 0.0;
                    long idleMs = now - p.lastUpdate;
                    String pctStr = String.format("%.2f", pct);
                    logger.info("[TUNING][HEARTBEAT] {} status={} {}/{} ({}%) idle={}s",
                            sym, p.status, p.testedConfigs.get(), p.totalConfigs, pctStr, idleMs/1000);
                });
            } catch (Exception ex) {
                // Log en debug pour éviter bruit excessif
                logger.debug("Heartbeat tuning erreur: {}", ex.getMessage());
            }
        }, 10, 30, TimeUnit.SECONDS);
    }

    /**
     * Structure centralisée des exceptions rencontrées pendant le tuning.
     * Permet une post-analyse (ex: patterns d'échecs récurrents).
     */
    public static class TuningExceptionReportEntry {
        public String symbol;
        public LstmConfig config;
        public String message;
        public String stacktrace;
        public long timestamp;
        public TuningExceptionReportEntry(String symbol, LstmConfig config, String message, String stacktrace, long timestamp) {
            this.symbol = symbol;
            this.config = config;
            this.message = message;
            this.stacktrace = stacktrace;
            this.timestamp = timestamp;
        }
    }
    private final java.util.concurrent.ConcurrentLinkedQueue<TuningExceptionReportEntry> tuningExceptionReport = new java.util.concurrent.ConcurrentLinkedQueue<>();

    public LstmTuningService(LstmTradePredictor lstmTradePredictor, LstmHyperparamsRepository hyperparamsRepository) {
        this.lstmTradePredictor = lstmTradePredictor;
        this.hyperparamsRepository = hyperparamsRepository;
    }

    /** Accès externe (monitoring) à l'état des tunings en cours. */
    public ConcurrentHashMap<String, TuningProgress> getTuningProgressMap() {
        return tuningProgressMap;
    }

    /** Retourne une copie snapshot des exceptions collectées (thread-safe via nouvelle ArrayList). */
    public java.util.List<TuningExceptionReportEntry> getTuningExceptionReport() {
        return new java.util.ArrayList<>(tuningExceptionReport);
    }

    /**
     * Méthode principale: tuning multi-threadé d'un symbole unique.
     *
     * Flux détaillé:
     *  1. Vérifie si le symbole a déjà un modèle en base (évite recalcul inutile).
     *  2. Protection mémoire: waitForMemory() avant de démarrer.
     *  3. Initialise la structure de progression (TuningProgress).
     *  4. Crée un pool de threads dimensionné par effectiveMaxThreads.
     *  5. Pour chaque configuration de la grille:
     *       - Soumet une tâche:
     *           a. Fixe les seeds (reproductibilité)
     *           b. Entraîne un modèle complet (trainLstmScalarV2)
     *           c. Évalue walk-forward (walkForwardEvaluate)
     *           d. Agrège les métriques (profitFactor, winRate, etc.)
     *           e. Sauvegarde les métriques (saveTuningMetrics)
     *           f. Retourne un TuningResult encapsulant les scores-clés
     *           g. En cas d'exception: log + ajout au tuningExceptionReport
     *           h. Nettoyage mémoire ND4J (invokeGc + destroy workspaces)
     *  6. Parcourt les Future pour:
     *       - Suivi progression
     *       - Sélection du meilleur businessScore
     *  7. Sauvegarde finale:
     *       - Hyperparamètres gagnants
     *       - Modèle (saveModelToDb)
     *  8. Mise à jour du statut (termine | failed)
     *
     * Choix de scoring:
     *  - Sélection basée sur businessScore (métrique composite) et non sur meanMSE direct.
     *
     * Important:
     *  - Ne pas modifier l'ordre des nettoyages mémoire (risque de fuite ou fragmentation).
     *  - Ne pas supprimer les catch/return null => ils préviennent les arrêts brutaux.
     *
     * @param symbol symbole à tuner
     * @param grid liste de configurations LstmConfig
     * @param series série temporelle (historique prix) nécessaire à l'entraînement/évaluation
     * @param jdbcTemplate accès base de données pour persistance du modèle
     * @return la meilleure configuration trouvée ou null si aucune valide
     */
    public LstmConfig tuneSymbolMultiThread(String symbol, List<LstmConfig> grid, BarSeries series, JdbcTemplate jdbcTemplate) {
        // 1. Vérifier si déjà tuné (évite duplication)
        if(isSymbolAlreydyTuned(symbol, jdbcTemplate)){
            return null;
        }

        // 2. Attendre si mémoire saturée
        waitForMemory();

        long startSymbol = System.currentTimeMillis();

        // 3. Initialisation progression
        TuningProgress progress = new TuningProgress();
        progress.symbol = symbol;
        progress.totalConfigs = grid.size();
        progress.testedConfigs.set(0);
        progress.status = "en_cours";
        progress.startTime = startSymbol;
        progress.lastUpdate = startSymbol;
        tuningProgressMap.put(symbol, progress);

        logger.info("[TUNING] Début du tuning V2 pour le symbole {} ({} configs) | maxThreadsEffectif={} (configuré={})",
                symbol, grid.size(), effectiveMaxThreads, configuredMaxThreads);

        double bestScore = Double.POSITIVE_INFINITY; // (Ici retenu pour trace; sélection réelle basée sur businessScore)
        LstmConfig existing = hyperparamsRepository.loadHyperparams(symbol);
        if(existing != null){
            logger.info("Hyperparamètres existants trouvés pour {}. Ignorer le tuning.", symbol);
            return existing;
        }

        LstmConfig bestConfig = null;
        MultiLayerNetwork bestModel = null;
        LstmTradePredictor.ScalerSet bestScalers = null;

        // 4. Dimensionnement du pool threads (sécurité: min entre taille grille, coeurs et limite effective)
        int numThreads = Math.min(Math.min(grid.size(), Runtime.getRuntime().availableProcessors()), effectiveMaxThreads);
        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(numThreads);
        java.util.List<java.util.concurrent.Future<TuningResult>> futures = new java.util.ArrayList<>();

        // 5. Soumission des tâches par configuration
        for (int i = 0; i < grid.size(); i++) {
            waitForMemory(); // Protection mémoire avant chaque tâche
            final int configIndex = i + 1;
            LstmConfig config = grid.get(i);

            // Activation explicite de certains modes (garantie pour ce pipeline)
            config.setUseScalarV2(true);
            config.setUseWalkForwardV2(true);

            futures.add(executor.submit(() -> {
                MultiLayerNetwork model = null;
                try {
                    long startConfig = System.currentTimeMillis();

                    // a. Seed global
                    lstmTradePredictor.setGlobalSeeds(config.getSeed());
                    logger.info("[TUNING][V2] [{}] Début config {}/{}", symbol, configIndex, grid.size());

                    // b. Entraînement sur la série complète (train principal)
                    LstmTradePredictor.TrainResult trFull = lstmTradePredictor.trainLstmScalarV2(series, config, null);
                    model = trFull.model;
                    LstmTradePredictor.ScalerSet scalers = trFull.scalers;

                    // c. Évaluation walk-forward
                    LstmTradePredictor.WalkForwardResultV2 wf = lstmTradePredictor.walkForwardEvaluate(series, config);
                    double meanMse = wf.meanMse;

                    // d. Agrégation métriques trading
                    double sumPF=0, sumWin=0, sumExp=0, maxDDPct=0, sumBusiness=0, sumProfit=0;
                    int splits=0;
                    int totalTrades=0;

                    for(LstmTradePredictor.TradingMetricsV2 m : wf.splits){
                        if(Double.isFinite(m.profitFactor)) sumPF += m.profitFactor; else sumPF += 0;
                        sumWin += m.winRate;
                        sumExp += m.expectancy;
                        if(m.maxDrawdownPct > maxDDPct) maxDDPct = m.maxDrawdownPct;
                        sumBusiness += (Double.isFinite(m.businessScore)? m.businessScore:0);
                        sumProfit += m.totalProfit;
                        totalTrades += m.numTrades;
                        splits++;
                    }

                    if(splits==0){
                        // Sécurité: aucun split valide => config invalide
                        throw new IllegalStateException("Aucun split valide walk-forward");
                    }

                    double meanPF = sumPF / splits;
                    double meanWinRate = sumWin / splits;
                    double meanExpectancy = sumExp / splits;
                    double meanBusinessScore = sumBusiness / splits;

                    // e. Prédiction "instantanée" sur dernier bar (utilisée pour direction indicative)
                    double predicted = lstmTradePredictor.predictNextCloseScalarV2(series, config, model, scalers);
                    double lastClose = series.getLastBar().getClosePrice().doubleValue();
                    double th = lstmTradePredictor.computeSwingTradeThreshold(series, config);
                    String direction = (predicted - lastClose) > th ? "up" : ((predicted - lastClose) < -th ? "down" : "stable");

                    // f. Calcul RMSE
                    double rmse = Double.isFinite(meanMse) && meanMse>=0? Math.sqrt(meanMse): Double.NaN;

                    // g. Persistance métriques complètes pour analyse ultérieure
                    hyperparamsRepository.saveTuningMetrics(
                            symbol, config, meanMse, rmse, direction,
                            sumProfit, meanPF, meanWinRate, maxDDPct, totalTrades, meanBusinessScore,
                            wf.splits.stream().mapToDouble(m->m.sortino).average().orElse(0.0),
                            wf.splits.stream().mapToDouble(m->m.calmar).average().orElse(0.0),
                            wf.splits.stream().mapToDouble(m->m.turnover).average().orElse(0.0),
                            wf.splits.stream().mapToDouble(m->m.avgBarsInPosition).average().orElse(0.0)
                    );

                    long endConfig = System.currentTimeMillis();
                    logger.info("[TUNING][V2] [{}] Fin config {}/{} | meanMSE={}, PF={}, winRate={}, DD%={}, expectancy={}, businessScore={}, trades={} durée={} ms",
                            symbol, configIndex, grid.size(), meanMse, meanPF, meanWinRate, maxDDPct,
                            meanExpectancy, meanBusinessScore, totalTrades, (endConfig-startConfig));

                    // h. Mise à jour progression (thread-safe)
                    TuningProgress p = tuningProgressMap.get(symbol);
                    if (p != null) {
                        p.testedConfigs.incrementAndGet();
                        p.lastUpdate = System.currentTimeMillis();
                    }

                    // i. Retour résultat encapsulé
                    return new TuningResult(config, model, scalers, meanMse, meanPF, meanWinRate, maxDDPct, meanBusinessScore);

                } catch (Exception e){
                    // Gestion centralisée des erreurs par config
                    logger.error("[TUNING][V2] Erreur config {} : {}", configIndex, e.getMessage());
                    String stack = org.apache.commons.lang3.exception.ExceptionUtils.getStackTrace(e);
                    tuningExceptionReport.add(new TuningExceptionReportEntry(symbol, config, e.getMessage(), stack, System.currentTimeMillis()));
                    return null;
                } finally {
                    // j. Libération références + GC ND4J
                    model = null;
                    org.nd4j.linalg.factory.Nd4j.getMemoryManager().invokeGc();
                    org.nd4j.linalg.factory.Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
                    System.gc();
                }
            }));
        }

        // 6. Fermeture soumissions
        executor.shutdown();

        int failedConfigs = 0;
        double bestBusinessScore = Double.NEGATIVE_INFINITY;

        // Récupération séquentielle des résultats (ordre des Future conservé)
        for (int i = 0; i < futures.size(); i++) {
            try {
                TuningResult result = futures.get(i).get(); // Bloquant
                logger.info("[TUNING][V2] [{}] Progression : {}/{} configs terminées", symbol, i+1, grid.size());

                // Ignorer configs échouées / invalides
                if (result == null || Double.isNaN(result.businessScore) || Double.isInfinite(result.businessScore)) {
                    failedConfigs++;
                    continue;
                }

                // Sélection basée sur businessScore
                if (result.businessScore > bestBusinessScore) {
                    bestBusinessScore = result.businessScore;
                    bestConfig = result.config;
                    bestModel = result.model;
                    bestScalers = result.scalers;
                    bestScore = result.score; // Conservation du meanMSE pour info
                }
            } catch (Exception e) {
                failedConfigs++;
                progress.status = "erreur";
                progress.lastUpdate = System.currentTimeMillis();
                logger.error("Erreur lors de la récup��ration du résultat de tuning : {}", e.getMessage());
                String stack = e.getCause() != null
                        ? org.apache.commons.lang3.exception.ExceptionUtils.getStackTrace(e.getCause())
                        : org.apache.commons.lang3.exception.ExceptionUtils.getStackTrace(e);
                tuningExceptionReport.add(new TuningExceptionReportEntry(symbol, null, e.getMessage(), stack, System.currentTimeMillis()));
            }
        }

        long endSymbol = System.currentTimeMillis();

        // 7. Cas d'échec total
        if (failedConfigs == grid.size()) {
            progress.status = "failed";
            progress.endTime = endSymbol;
            progress.lastUpdate = endSymbol;
            logger.error("[TUNING][V2][EARLY STOP] Toutes les configs ont échoué pour le symbole {}.", symbol);
            return null;
        }

        // 8. Succès global
        progress.status = "termine";
        progress.endTime = endSymbol;
        progress.lastUpdate = endSymbol;

        if (bestConfig != null && bestModel != null && bestScalers != null) {
            // Sauvegarde hyperparamètres gagnants
            hyperparamsRepository.saveHyperparams(symbol, bestConfig);
            try {
                // Sauvegarde modèle (sera rechargé pour prédictions)
                lstmTradePredictor.saveModelToDb(symbol, bestModel, jdbcTemplate, bestConfig, bestScalers);
            } catch (Exception e) {
                logger.error("Erreur lors de la sauvegarde du meilleur modèle : {}", e.getMessage());
            }
            logger.info("[TUNING][V2] Fin tuning {} | Best businessScore={} | windowSize={}, neurons={}, dropout={}, lr={}, l1={}, l2={}, meanMSE={}, durée={} ms",
                    symbol, bestBusinessScore, bestConfig.getWindowSize(), bestConfig.getLstmNeurons(),
                    bestConfig.getDropoutRate(), bestConfig.getLearningRate(), bestConfig.getL1(), bestConfig.getL2(),
                    bestScore, (endSymbol - startSymbol));
        } else {
            logger.warn("[TUNING][V2] Aucun modèle/scaler valide trouvé pour {}", symbol);
        }

        // 9. Nettoyage global mémoire (sécurité)
        try {
            org.nd4j.linalg.factory.Nd4j.getMemoryManager().invokeGc();
            org.nd4j.linalg.factory.Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
            System.gc();
        } catch (Exception e) {
            logger.warn("Nettoyage ND4J échec : {}", e.getMessage());
        }

        return bestConfig;
    }

    /**
     * Vérifie en base si un modèle existe déjà pour le symbole.
     * Utilisé pour éviter un tuning redondant (économie temps/ressources).
     * @param symbol symbole recherché
     * @param jdbcTemplate accès DB
     * @return true si déjà tuné (au moins 1 enregistrement), sinon false
     */
    public boolean isSymbolAlreydyTuned(String symbol, JdbcTemplate jdbcTemplate) {
        String sql = "SELECT COUNT(*) FROM lstm_models WHERE symbol = ?";
        try {
            Integer count = jdbcTemplate.queryForObject(sql, Integer.class, symbol);
            return count != null && count > 0;
        } catch (Exception e) {
            logger.error("Erreur lors de la vérification du tuning du symbole {} : {}", symbol, e.getMessage());
            return false;
        }
    }

    /**
     * Génère automatiquement une grille de configurations optimisée pour le swing trade professionnel.
     * Adaptée pour 1200 bougies par symbole avec paramètres réalistes.
     * @return liste de LstmConfig à tester
     */
    public List<LstmConfig> generateSwingTradeGrid() {
        List<LstmConfig> grid = new java.util.ArrayList<>();

        // Paramètres optimisés pour swing trade professionnel (3-10 jours)
        int[] windowSizes = {30, 40, 50}; // Fenêtres plus larges pour swing trade
        int[] lstmNeurons = {128, 192, 256}; // Plus de neurones pour capturer patterns complexes
        double[] dropoutRates = {0.25, 0.3, 0.35}; // Dropout adapté pour éviter overfitting
        double[] learningRates = {0.0002, 0.0003, 0.0005}; // LR plus faibles pour stabilité
        double[] l1s = {0.0}; // Pas de L1 pour swing trade
        double[] l2s = {0.003, 0.005, 0.008}; // L2 plus élevé pour généralisation

        int numEpochs = 200; // Plus d'époques pour swing trade
        int patience = 25; // Patience plus élevée
        double minDelta = 0.00012; // MinDelta plus fin
        int kFolds = 5;
        String optimizer = "adam";
        String[] scopes = {"window"};
        String[] swingTypes = {"range", "breakout", "mean_reversion"}; // Tous les types swing

        for (String swingType : swingTypes) {
            for (String scope : scopes) {
                for (int windowSize : windowSizes) {
                    for (int neurons : lstmNeurons) {
                        for (double dropout : dropoutRates) {
                            for (double lr : learningRates) {
                                for (double l1 : l1s) {
                                    for (double l2 : l2s) {
                                        LstmConfig config = new LstmConfig();
                                        config.setWindowSize(windowSize);
                                        config.setLstmNeurons(neurons);
                                        config.setDropoutRate(dropout);
                                        config.setLearningRate(lr);
                                        config.setNumEpochs(numEpochs);
                                        config.setPatience(patience);
                                        config.setMinDelta(minDelta);
                                        config.setKFolds(kFolds);
                                        config.setOptimizer(optimizer);
                                        config.setL1(l1);
                                        config.setL2(l2);
                                        config.setNormalizationScope(scope);
                                        config.setNormalizationMethod("auto");
                                        config.setSwingTradeType(swingType);
                                        config.setUseScalarV2(true);
                                        config.setUseWalkForwardV2(true);

                                        // Paramètres swing trade spécifiques
                                        config.setNumLstmLayers(3); // 3 couches pour plus de profondeur
                                        config.setBidirectional(false); // Pas bidirectionnel pour swing
                                        config.setAttention(true); // Attention activée
                                        config.setHorizonBars(7); // Horizon 7 jours pour swing
                                        config.setThresholdK(1.5); // Seuil plus élevé
                                        config.setBatchSize(64); // Batch size optimisé
                                        config.setWalkForwardSplits(5); // Plus de splits
                                        config.setEmbargoBars(3); // Embargo pour éviter leakage

                                        // Paramètres trading Alpaca
                                        config.setCapital(10000.0);
                                        config.setRiskPct(0.02); // 2% de risque par trade
                                        config.setSizingK(1.2);
                                        config.setFeePct(0.0); // Commission-free Alpaca
                                        config.setSlippagePct(0.0005); // Slippage réaliste

                                        // Business score optimisé
                                        config.setBusinessProfitFactorCap(4.0);
                                        config.setBusinessDrawdownGamma(1.5);

                                        grid.add(config);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return grid;
    }

    /**
     * Génère automatiquement une grille de configurations adaptée au swing trade enrichie.
     * @param features liste des features à utiliser
     * @param horizonBars tableau des horizons de prédiction à tester
     * @param numLstmLayers tableau du nombre de couches LSTM
     * @param batchSizes tableau des batch sizes
     * @param bidirectionals tableau des valeurs bidirectional
     * @param attentions tableau des valeurs attention
     * @return liste de LstmConfig à tester
     */
    public List<LstmConfig> generateSwingTradeGrid(int[] horizonBars, int[] numLstmLayers, int[] batchSizes, boolean[] bidirectionals, boolean[] attentions) {
        List<LstmConfig> grid = new java.util.ArrayList<>();
        int[] windowSizes = {20, 30, 40};
        int[] lstmNeurons = {64, 128, 256};
        double[] dropoutRates = {0.2, 0.3};
        double[] learningRates = {0.0005, 0.001};
        double[] l1s = {0.0};
        double[] l2s = {0.0001, 0.001};
        int numEpochs = 150;
        int patience = 10;
        double minDelta = 0.0005;
        int kFolds = 3;
        String optimizer = "adam";
        String[] scopes = {"window"};
        String[] swingTypes = {"range", "mean_reversion"};
        for (String swingType : swingTypes) {
            for (String scope : scopes) {
                for (int windowSize : windowSizes) {
                    for (int neurons : lstmNeurons) {
                        for (double dropout : dropoutRates) {
                            for (double lr : learningRates) {
                                for (double l1 : l1s) {
                                    for (double l2 : l2s) {
                                        for (int numLayers : numLstmLayers) {
                                            for (int batchSize : batchSizes) {
                                                for (boolean bidir : bidirectionals) {
                                                    for (boolean att : attentions) {
                                                        for (int horizon : horizonBars) {
                                                            LstmConfig config = new LstmConfig();
                                                            config.setWindowSize(windowSize);
                                                            config.setLstmNeurons(neurons);
                                                            config.setDropoutRate(dropout);
                                                            config.setLearningRate(lr);
                                                            config.setNumEpochs(numEpochs);
                                                            config.setPatience(patience);
                                                            config.setMinDelta(minDelta);
                                                            config.setKFolds(kFolds);
                                                            config.setOptimizer(optimizer);
                                                            config.setL1(l1);
                                                            config.setL2(l2);
                                                            config.setNormalizationScope(scope);
                                                            config.setNormalizationMethod("auto");
                                                            config.setSwingTradeType(swingType);
                                                            config.setUseScalarV2(true);
                                                            config.setUseWalkForwardV2(true);
                                                            config.setNumLstmLayers(numLayers);
                                                            config.setBatchSize(batchSize);
                                                            config.setBidirectional(bidir);
                                                            config.setAttention(att);
                                                            config.setHorizonBars(horizon);
                                                            grid.add(config);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return grid;
    }

    /**
     * Génère une grille aléatoire de configurations adaptée au swing trade.
     * @param n nombre de configurations à générer
     * @return liste de LstmConfig aléatoires
     */
    public List<LstmConfig> generateRandomSwingTradeGrid(int n) {
        java.util.Random rand = new java.util.Random();
        int[] windowSizes = {10, 20, 30};
        int[] lstmNeurons = {64, 128};
        double[] dropoutRates = {0.2, 0.3};
        double[] learningRates = {0.0005, 0.001};
        double[] l1s = {0.0};
        double[] l2s = {0.0001, 0.001};
        int numEpochs = 50;
        int patience = 5;
        double minDelta = 0.0005;
        int kFolds = 3;
        String optimizer = "adam";
        String[] scopes = {"window"};
        String[] swingTypes = {"range", "breakout", "mean_reversion"};
        List<LstmConfig> grid = new java.util.ArrayList<>();
        for (int i = 0; i < n; i++) {
            LstmConfig config = new LstmConfig();
            config.setWindowSize(windowSizes[rand.nextInt(windowSizes.length)]);
            config.setLstmNeurons(lstmNeurons[rand.nextInt(lstmNeurons.length)]);
            config.setDropoutRate(dropoutRates[rand.nextInt(dropoutRates.length)]);
            config.setLearningRate(learningRates[rand.nextInt(learningRates.length)]);
            config.setNumEpochs(numEpochs);
            config.setPatience(patience);
            config.setMinDelta(minDelta);
            config.setKFolds(kFolds);
            config.setOptimizer(optimizer);
            config.setL1(l1s[rand.nextInt(l1s.length)]);
            config.setL2(l2s[rand.nextInt(l2s.length)]);
            config.setNormalizationScope(scopes[rand.nextInt(scopes.length)]);
            config.setNormalizationMethod("auto");
            config.setSwingTradeType(swingTypes[rand.nextInt(swingTypes.length)]);
            config.setUseScalarV2(true);
            config.setUseWalkForwardV2(true);
            grid.add(config);
        }
        return grid;
    }

    /**
     * Génère une grille aléatoire de configurations adaptée au swing trade avec seed pour reproductibilité.
     * @param n nombre de configurations à générer
     * @param cvMode mode de validation croisée
     * @param seed seed pour la reproductibilité
     * @return liste de LstmConfig aléatoires
     */
    public List<LstmConfig> generateRandomSwingTradeGrid(int n, String cvMode, long seed) {
        java.util.Random rand = new java.util.Random(seed);
        int[] windowSizes = {10, 20, 30};
        int[] lstmNeurons = {64, 128};
        double[] dropoutRates = {0.2, 0.3};
        double[] learningRates = {0.0005, 0.001};
        double[] l1s = {0.0};
        double[] l2s = {0.0001, 0.001};
        int numEpochs = 50;
        int patience = 5;
        double minDelta = 0.0005;
        int kFolds = 3;
        String optimizer = "adam";
        String[] scopes = {"window"};
        String[] swingTypes = {"range", "breakout", "mean_reversion"};
        List<LstmConfig> grid = new java.util.ArrayList<>();
        for (int i = 0; i < n; i++) {
            LstmConfig config = new LstmConfig();
            config.setWindowSize(windowSizes[rand.nextInt(windowSizes.length)]);
            config.setLstmNeurons(lstmNeurons[rand.nextInt(lstmNeurons.length)]);
            config.setDropoutRate(dropoutRates[rand.nextInt(dropoutRates.length)]);
            config.setLearningRate(learningRates[rand.nextInt(learningRates.length)]);
            config.setNumEpochs(numEpochs);
            config.setPatience(patience);
            config.setMinDelta(minDelta);
            config.setKFolds(kFolds);
            config.setOptimizer(optimizer);
            config.setL1(l1s[rand.nextInt(l1s.length)]);
            config.setL2(l2s[rand.nextInt(l2s.length)]);
            config.setNormalizationScope(scopes[rand.nextInt(scopes.length)]);
            config.setNormalizationMethod("auto");
            config.setSwingTradeType(swingTypes[rand.nextInt(swingTypes.length)]);
            config.setUseScalarV2(true);
            config.setUseWalkForwardV2(true);
            // Ajout du mode de validation croisée
            config.setCvMode(cvMode);
            grid.add(config);
        }
        return grid;
    }

    /**
     * Génère une grille aléatoire de configurations adaptée au swing trade enrichie.
     * @param n nombre de configurations à générer
     * @param features liste des features à utiliser
     * @param horizonBars tableau des horizons de prédiction à tester
     * @param numLstmLayers tableau du nombre de couches LSTM
     * @param batchSizes tableau des batch sizes
     * @param bidirectionals tableau des valeurs bidirectional
     * @param attentions tableau des valeurs attention
     * @return liste de LstmConfig aléatoires
     */
    public List<LstmConfig> generateRandomSwingTradeGrid(int n, int[] horizonBars, int[] numLstmLayers, int[] batchSizes, boolean[] bidirectionals, boolean[] attentions) {
        java.util.Random rand = new java.util.Random();
        int[] windowSizes = {10, 20, 30};
        int[] lstmNeurons = {64, 128};
        double[] dropoutRates = {0.2, 0.3};
        double[] learningRates = {0.0005, 0.001};
        double[] l1s = {0.0};
        double[] l2s = {0.0001, 0.001};
        int numEpochs = 50;
        int patience = 5;
        double minDelta = 0.0005;
        int kFolds = 3;
        String optimizer = "adam";
        String[] scopes = {"window"};
        String[] swingTypes = {"range", "breakout", "mean_reversion"};
        List<LstmConfig> grid = new java.util.ArrayList<>();
        for (int i = 0; i < n; i++) {
            LstmConfig config = new LstmConfig();
            config.setWindowSize(windowSizes[rand.nextInt(windowSizes.length)]);
            config.setLstmNeurons(lstmNeurons[rand.nextInt(lstmNeurons.length)]);
            config.setDropoutRate(dropoutRates[rand.nextInt(dropoutRates.length)]);
            config.setLearningRate(learningRates[rand.nextInt(learningRates.length)]);
            config.setNumEpochs(numEpochs);
            config.setPatience(patience);
            config.setMinDelta(minDelta);
            config.setKFolds(kFolds);
            config.setOptimizer(optimizer);
            config.setL1(l1s[rand.nextInt(l1s.length)]);
            config.setL2(l2s[rand.nextInt(l2s.length)]);
            config.setNormalizationScope(scopes[rand.nextInt(scopes.length)]);
            config.setNormalizationMethod("auto");
            config.setSwingTradeType(swingTypes[rand.nextInt(swingTypes.length)]);
            config.setUseScalarV2(true);
            config.setUseWalkForwardV2(true);
            config.setNumLstmLayers(numLstmLayers[rand.nextInt(numLstmLayers.length)]);
            config.setBatchSize(batchSizes[rand.nextInt(batchSizes.length)]);
            config.setBidirectional(bidirectionals[rand.nextInt(bidirectionals.length)]);
            config.setAttention(attentions[rand.nextInt(attentions.length)]);
            config.setHorizonBars(horizonBars[rand.nextInt(horizonBars.length)]);
            grid.add(config);
        }
        return grid;
    }

    /**
     * Génère automatiquement une grille de configurations adaptée au swing trade avec plusieurs modes de validation croisée.
     * @param cvModes liste des modes de validation croisée à tester (ex : split, timeseries, kfold)
     * @return liste de LstmConfig à tester
     */
    public List<LstmConfig> generateSwingTradeGrid(List<String> cvModes) {
        List<LstmConfig> grid = new java.util.ArrayList<>();
        int[] windowSizes = {20, 30, 40};
        int[] lstmNeurons = {64, 128, 256};
        double[] dropoutRates = {0.2, 0.3};
        double[] learningRates = {0.0005, 0.001};
        double[] l1s = {0.0};
        double[] l2s = {0.0001, 0.001};
        int numEpochs = 150;
        int patience = 10;
        double minDelta = 0.0005;
        int kFolds = 3;
        String optimizer = "adam";
        String[] scopes = {"window"};
        String[] swingTypes = {"range", "mean_reversion"};
        for (String swingType : swingTypes) {
            for (String scope : scopes) {
                for (int windowSize : windowSizes) {
                    for (int neurons : lstmNeurons) {
                        for (double dropout : dropoutRates) {
                            for (double lr : learningRates) {
                                for (double l1 : l1s) {
                                    for (double l2 : l2s) {
                                        for (String cvMode : cvModes) {
                                            LstmConfig config = new LstmConfig();
                                            config.setWindowSize(windowSize);
                                            config.setLstmNeurons(neurons);
                                            config.setDropoutRate(dropout);
                                            config.setLearningRate(lr);
                                            config.setNumEpochs(numEpochs);
                                            config.setPatience(patience);
                                            config.setMinDelta(minDelta);
                                            config.setKFolds(kFolds);
                                            config.setOptimizer(optimizer);
                                            config.setL1(l1);
                                            config.setL2(l2);
                                            config.setNormalizationScope(scope);
                                            config.setNormalizationMethod("auto");
                                            config.setSwingTradeType(swingType);
                                            config.setUseScalarV2(true);
                                            config.setUseWalkForwardV2(true);
                                            config.setCvMode(cvMode);
                                            grid.add(config);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return grid;
    }

    /**
     * Génère automatiquement une grille de configurations adaptée au swing trade enrichie avec plusieurs modes de validation croisée.
     * @param features liste des features à utiliser
     * @param horizonBars tableau des horizons de prédiction à tester
     * @param numLstmLayers tableau du nombre de couches LSTM
     * @param batchSizes tableau des batch sizes
     * @param bidirectionals tableau des valeurs bidirectional
     * @param attentions tableau des valeurs attention
     * @param cvModes liste des modes de validation croisée à tester
     * @return liste de LstmConfig à tester
     */
    public List<LstmConfig> generateSwingTradeGrid(List<String> features, int[] horizonBars, int[] numLstmLayers, int[] batchSizes, boolean[] bidirectionals, boolean[] attentions, List<String> cvModes) {
        List<LstmConfig> grid = new java.util.ArrayList<>();
        int[] windowSizes = {20, 30, 40};
        int[] lstmNeurons = {64, 128, 256};
        double[] dropoutRates = {0.2, 0.3};
        double[] learningRates = {0.0005, 0.001};
        double[] l1s = {0.0};
        double[] l2s = {0.0001, 0.001};
        int numEpochs = 150;
        int patience = 10;
        double minDelta = 0.0005;
        int kFolds = 3;
        String optimizer = "adam";
        String[] scopes = {"window"};
        String[] swingTypes = {"range", "mean_reversion"};
        for (String swingType : swingTypes) {
            for (String scope : scopes) {
                for (int windowSize : windowSizes) {
                    for (int neurons : lstmNeurons) {
                        for (double dropout : dropoutRates) {
                            for (double lr : learningRates) {
                                for (double l1 : l1s) {
                                    for (double l2 : l2s) {
                                        for (int numLayers : numLstmLayers) {
                                            for (int batchSize : batchSizes) {
                                                for (boolean bidir : bidirectionals) {
                                                    for (boolean att : attentions) {
                                                        for (int horizon : horizonBars) {
                                                            for (String cvMode : cvModes) {
                                                                LstmConfig config = new LstmConfig();
                                                                config.setWindowSize(windowSize);
                                                                config.setLstmNeurons(neurons);
                                                                config.setDropoutRate(dropout);
                                                                config.setLearningRate(lr);
                                                                config.setNumEpochs(numEpochs);
                                                                config.setPatience(patience);
                                                                config.setMinDelta(minDelta);
                                                                config.setKFolds(kFolds);
                                                                config.setOptimizer(optimizer);
                                                                config.setL1(l1);
                                                                config.setL2(l2);
                                                                config.setNormalizationScope(scope);
                                                                config.setNormalizationMethod("auto");
                                                                config.setSwingTradeType(swingType);
                                                                config.setUseScalarV2(true);
                                                                config.setUseWalkForwardV2(true);
                                                                config.setNumLstmLayers(numLayers);
                                                                config.setBatchSize(batchSize);
                                                                config.setBidirectional(bidir);
                                                                config.setAttention(att);
                                                                config.setHorizonBars(horizon);
                                                                config.setFeatures(features);
                                                                config.setCvMode(cvMode);
                                                                grid.add(config);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return grid;
    }

    /**
     * Tune automatiquement les hyperparamètres pour tous les symboles donnés.
     * @param symbols les symboles à tuner
     * @param grid la liste des configurations à tester
     * @param jdbcTemplate accès base
     * @param seriesProvider fournisseur de séries temporelles pour chaque symbole
     */
    public void tuneAllSymbols(List<String> symbols, List<LstmConfig> grid, JdbcTemplate jdbcTemplate, java.util.function.Function<String, BarSeries> seriesProvider) {
        long startAll = System.currentTimeMillis();
        logger.info("[TUNING] Début tuning multi-symboles ({} symboles)", symbols.size());
        for (int i = 0; i < symbols.size(); i++) {
            waitForMemory(); // Protection mémoire avant chaque tuning de symbole
            String symbol = symbols.get(i);
            long startSymbol = System.currentTimeMillis();
            logger.info("[TUNING] Début tuning symbole {}/{} : {}", i+1, symbols.size(), symbol);
            BarSeries series = seriesProvider.apply(symbol);
            tuneSymbolMultiThread(symbol, grid, series, jdbcTemplate);
            try {
                org.nd4j.linalg.factory.Nd4j.getMemoryManager().invokeGc();
                org.nd4j.linalg.api.memory.MemoryWorkspaceManager wsManager = org.nd4j.linalg.factory.Nd4j.getWorkspaceManager();
                wsManager.destroyAllWorkspacesForCurrentThread();
            } catch (Exception e) {
                logger.warn("Erreur lors du nettoyage ND4J/DL4J après le tuning du symbole {} : {}", symbol, e.getMessage());
            }
            long endSymbol = System.currentTimeMillis();
            logger.info("[TUNING] Fin tuning symbole {} | durée={} ms", symbol, (endSymbol - startSymbol));
        }
        long endAll = System.currentTimeMillis();
        logger.info("[TUNING] Fin tuning multi-symboles | durée totale={} ms", (endAll - startAll));
    }

    /** Nouveau business score V2 */
    private double computeBusinessScoreV2(double profitFactor, double winRate, double maxDrawdownPct, double expectancy, LstmConfig config){
        // Important: ne pas modifier => impact sur choix des configs si activé ailleurs
        double pfAdj = Math.min(profitFactor, config.getBusinessProfitFactorCap());
        double expPos = Math.max(expectancy, 0.0);
        double denom = 1.0 + Math.pow(Math.max(maxDrawdownPct, 0.0), config.getBusinessDrawdownGamma());
        return (expPos * pfAdj * winRate) / (denom + 1e-9);
    }

    private static class TuningResult {
        LstmConfig config; MultiLayerNetwork model; LstmTradePredictor.ScalerSet scalers; double score; double profitFactor; double winRate; double maxDrawdown; double businessScore;
        TuningResult(LstmConfig c, MultiLayerNetwork m, LstmTradePredictor.ScalerSet s, double score, double pf, double wr, double dd, double bs){ this.config=c; this.model=m; this.scalers=s; this.score=score; this.profitFactor=pf; this.winRate=wr; this.maxDrawdown=dd; this.businessScore=bs; }
    }

    /**
     * Seuil d'utilisation mémoire accepté (80%).
     * Ajuster avec prudence: trop bas => tuning ralenti; trop haut => risque OOM.
     */
    private static final double MEMORY_USAGE_THRESHOLD = 0.8;

    /**
     * Vérifie si l'utilisation mémoire actuelle dépasse le seuil défini.
     * @return true si usage mémoire > threshold, sinon false
     */
    private static boolean isMemoryUsageHigh() {
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long usedMemory = totalMemory - freeMemory;
        double usage = (double) usedMemory / (double) maxMemory;
        return usage > MEMORY_USAGE_THRESHOLD;
    }

    /**
     * Boucle d'attente active (avec sleep) tant que la mémoire est trop utilisée.
     * Raison: éviter de lancer un nouvel entraînement LSTM (coûteux en RAM) quand la JVM est proche du plafond.
     * Stratégie simple mais efficace pour environnement batch/offline.
     * Amélioration future possible: backoff exponentiel / monitoring externe.
     */
    private static void waitForMemory() {
        while (isMemoryUsageHigh()) {
            logger.warn("[TUNING] Utilisation mémoire élevée (> {}%), attente avant de lancer une nouvelle tâche...",
                    (int)(MEMORY_USAGE_THRESHOLD*100));
            try {
                Thread.sleep(5000); // Pause 5s (magique mais suffisant ici)
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }
}
