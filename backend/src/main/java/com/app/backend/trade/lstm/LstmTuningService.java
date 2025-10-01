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
     * Cette méthode orchestre le processus complet d'optimisation des hyperparamètres LSTM:
     * - Teste une grille de configurations en parallèle pour maximiser l'utilisation CPU/GPU
     * - Évalue chaque configuration via entraînement + validation walk-forward
     * - Sélectionne la meilleure configuration basée sur des métriques de trading business
     * - Persiste les résultats (hyperparamètres + modèle) pour utilisation ultérieure
     *
     * Architecture multi-thread:
     * - Pool de threads limité par effectiveMaxThreads (auto-calculé selon CPU/GPU)
     * - Chaque thread traite une configuration indépendamment
     * - Synchronisation via Future pour collecte des résultats
     * - Protection mémoire active pour éviter les OutOfMemoryError
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
     *  - businessScore combine: profitFactor, winRate, drawdown, expectancy
     *
     * Gestion d'erreurs:
     *  - Exceptions par configuration collectées dans tuningExceptionReport
     *  - Échec d'une config n'arrête pas le processus global
     *  - Si toutes les configs échouent => return null
     *
     * Important:
     *  - Ne pas modifier l'ordre des nettoyages mémoire (risque de fuite ou fragmentation).
     *  - Ne pas supprimer les catch/return null => ils préviennent les arrêts brutaux.
     *  - Les seeds garantissent la reproductibilité des résultats
     *
     * @param symbol symbole à tuner (ex: "AAPL", "BTCUSDT")
     * @param grid liste de configurations LstmConfig à tester (grille d'hyperparamètres)
     * @param series série temporelle (historique prix OHLCV) nécessaire à l'entraînement/évaluation
     * @param jdbcTemplate accès base de données pour persistance du modèle et vérifications
     * @return la meilleure configuration trouvée ou null si aucune valide ou déjà existante
     */
    public LstmConfig tuneSymbolMultiThread(String symbol, List<LstmConfig> grid, BarSeries series, JdbcTemplate jdbcTemplate) {
        // ===== PHASE 1: VÉRIFICATIONS PRÉLIMINAIRES =====

        // Vérifier si un modèle existe déjà pour ce symbole (évite duplication coûteuse)
        // Consultation rapide en base: SELECT COUNT(*) FROM lstm_models WHERE symbol = ?
        if(isSymbolAlreydyTuned(symbol, jdbcTemplate)){
            logger.info("[TUNING] Symbole {} déjà tuné, abandon du processus", symbol);
            return null; // Aucun tuning nécessaire
        }

        // Protection mémoire préventive: attendre que l'usage RAM descende sous le seuil
        // Évite de lancer un tuning coûteux quand la JVM est proche de la limite
        waitForMemory();

        // Capture du timestamp de début pour mesure de performance globale
        long startSymbol = System.currentTimeMillis();

        // ===== PHASE 2: INITIALISATION DU SUIVI DE PROGRESSION =====

        // Création de la structure de monitoring thread-safe pour ce symbole
        TuningProgress progress = new TuningProgress();
        progress.symbol = symbol;                    // Identifiant du symbole en cours
        progress.totalConfigs = grid.size();         // Nombre total de configurations à tester
        progress.testedConfigs.set(0);              // Compteur atomique des configs terminées
        progress.status = "en_cours";               // État initial du processus
        progress.startTime = startSymbol;           // Timestamp de début
        progress.lastUpdate = startSymbol;          // Dernière mise à jour (heartbeat)

        // Enregistrement dans la map globale pour monitoring externe (APIs, logs heartbeat)
        tuningProgressMap.put(symbol, progress);

        // Log informatif du démarrage avec paramètres de parallélisme
        logger.info("[TUNING] Début du tuning V2 pour le symbole {} ({} configs) | maxThreadsEffectif={} (configuré={})",
                symbol, grid.size(), effectiveMaxThreads, configuredMaxThreads);

        // ===== PHASE 3: VÉRIFICATION DOUBLE SÉCURITÉ =====

        // Variable de trace pour le meilleur score MSE (information, pas utilisée pour sélection)
        double bestScore = Double.POSITIVE_INFINITY;

        // Double vérification: chercher des hyperparamètres existants en base
        // (redondant avec isSymbolAlreydyTuned mais protection supplémentaire)
        LstmConfig existing = hyperparamsRepository.loadHyperparams(symbol);
        if(existing != null){
            logger.info("Hyperparamètres existants trouvés pour {}. Ignorer le tuning.", symbol);
            return existing; // Retourner la config existante
        }

        // ===== PHASE 4: INITIALISATION DES VARIABLES DE RÉSULTATS =====

        // Variables pour stocker la meilleure configuration trouvée
        LstmConfig bestConfig = null;              // Configuration gagnante
        MultiLayerNetwork bestModel = null;        // Modèle neuronal correspondant
        LstmTradePredictor.ScalerSet bestScalers = null; // Normalisateurs/scalers associés

        // ===== PHASE 5: CRÉATION DU POOL DE THREADS =====

        // Calcul du nombre optimal de threads pour ce tuning
        // Équilibrage entre: taille de grille, nombre de coeurs CPU, limite effective
        // Note: actuellement forcé à 4 pour stabilité (ligne commentée montre le calcul automatique)
        int numThreads = 1; // Math.min(Math.min(grid.size(), Runtime.getRuntime().availableProcessors()), effectiveMaxThreads);

        // Création du pool de threads à taille fixe pour traitement parallèle
        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(numThreads);

        // Liste des Future pour récupération asynchrone des résultats
        java.util.List<java.util.concurrent.Future<TuningResult>> futures = new java.util.ArrayList<>();

        // ===== PHASE 6: SOUMISSION DES TÂCHES DE TUNING =====

        // Itération sur chaque configuration de la grille d'hyperparamètres
        for (int i = 0; i < grid.size(); i++) {
            // Protection mémoire avant chaque nouvelle tâche (évite accumulation)
            waitForMemory();

            // Index de configuration (1-based pour logs lisibles)
            final int configIndex = i + 1;

            // Configuration courante à tester
            LstmConfig config = grid.get(i);

            // ===== ACTIVATION DES MODES AVANCÉS =====
            // Force l'utilisation des versions optimisées des algorithmes
            config.setUseScalarV2(true);        // Version améliorée du traitement scalaire
            config.setUseWalkForwardV2(true);   // Version optimisée de la validation walk-forward

            // Soumission de la tâche asynchrone au pool de threads
            futures.add(executor.submit(() -> {
                // Variable locale pour référence au modèle (nettoyage dans finally)
                MultiLayerNetwork model = null;

                try {
                    // ===== DÉBUT DE TRAITEMENT D'UNE CONFIGURATION =====

                    // Timestamp de début pour mesure de performance par config
                    long startConfig = System.currentTimeMillis();

                    // ===== REPRODUCTIBILITÉ: INITIALISATION DES SEEDS =====
                    // Fixe les générateurs aléatoires pour résultats reproductibles
                    // Critique pour pouvoir recréer exactement les mêmes résultats
                    lstmTradePredictor.setGlobalSeeds(config.getSeed());

                    // Log de début de traitement de cette configuration
                    logger.info("[TUNING][V2] [{}] Début config {}/{}", symbol, configIndex, grid.size());

                    // ===== ENTRAÎNEMENT DU MODÈLE PRINCIPAL =====
                    // Entraîne un modèle LSTM complet sur toute la série temporelle
                    // Retourne: modèle neuronal + scalers (normalisateurs) + métriques d'entraînement
                    LstmTradePredictor.TrainResult trFull = lstmTradePredictor.trainLstmScalarV2(series, config, null);
                    model = trFull.model;                    // Modèle neuronal entraîné
                    LstmTradePredictor.ScalerSet scalers = trFull.scalers; // Normalisateurs (min/max, z-score, etc.)

                    // ===== VALIDATION WALK-FORWARD =====
                    // Évalue la performance du modèle via validation temporelle séquentielle
                    // Simule le trading réel: entraîne sur passé, teste sur futur, avance dans le temps
                    // CORRECTION: Utilise le modèle pré-entraîné au lieu de re-entraîner pour chaque split
                    LstmTradePredictor.WalkForwardResultV2 wf = lstmTradePredictor.walkForwardEvaluate(series, config, model, scalers);
                    double meanMse = wf.meanMse;            // Erreur quadratique moyenne sur tous les splits

                    // ===== AGRÉGATION DES MÉTRIQUES DE TRADING =====
                    // Collecte et moyenne des métriques business sur tous les splits walk-forward
                    double sumPF=0, sumWin=0, sumExp=0, maxDDPct=0, sumBusiness=0, sumProfit=0; // Accumulateurs
                    int splits=0;           // Compteur de splits valides
                    int totalTrades=0;      // Nombre total de trades sur tous les splits

                    // Parcours de tous les résultats de split pour agrégation
                    for(LstmTradePredictor.TradingMetricsV2 m : wf.splits){
                        // Profit Factor: rapport gains/pertes (garde 0 si infini/NaN)
                        if(Double.isFinite(m.profitFactor)) sumPF += m.profitFactor; else sumPF += 0;

                        // Win Rate: pourcentage de trades gagnants
                        sumWin += m.winRate;

                        // Expectancy: gain moyen par trade
                        sumExp += m.expectancy;

                        // Max Drawdown: pire perte cumulée (garde le maximum)
                        if(m.maxDrawdownPct > maxDDPct) maxDDPct = m.maxDrawdownPct;

                        // Business Score: métrique composite (garde 0 si infini/NaN)
                        sumBusiness += (Double.isFinite(m.businessScore)? m.businessScore:0);

                        // Profit total: somme des gains/pertes
                        sumProfit += m.totalProfit;

                        // Nombre de trades: pour calcul de statistiques
                        totalTrades += m.numTrades;

                        // Compteur de splits traités
                        splits++;
                    }

                    // Sécurité: vérifier qu'au moins un split est valide
                    if(splits==0){
                        throw new IllegalStateException("Aucun split valide walk-forward");
                    }

                    // ===== CALCUL DES MOYENNES =====
                    // Moyennes des métriques sur tous les splits (normalisation)
                    double meanPF = sumPF / splits;              // Profit Factor moyen
                    double meanWinRate = sumWin / splits;        // Taux de réussite moyen
                    double meanExpectancy = sumExp / splits;     // Expectancy moyenne
                    // Score business moyen (critère de sélection)
                    double meanBusinessScore = sumBusiness / splits;


                    // ===== CALCUL DE LA RMSE =====
                    // Root Mean Square Error: racine carrée de la MSE (plus interprétable)
                    double rmse = Double.isFinite(meanMse) && meanMse>=0? Math.sqrt(meanMse): Double.NaN;

                    // ===== PERSISTANCE DES MÉTRIQUES COMPLÈTES =====
                    // Sauvegarde en base de toutes les métriques pour analyse post-tuning
                    // Permet comparaisons, debug, et optimisations futures
                    hyperparamsRepository.saveTuningMetrics(
                            symbol, config,                    // Identifiant + configuration
                            meanMse, rmse,          // Métriques de prédiction
                            sumProfit, meanPF, meanWinRate, maxDDPct, totalTrades, meanBusinessScore, // Trading metrics
                            // Métriques avancées (moyennes des splits)
                            wf.splits.stream().mapToDouble(m->m.sortino).average().orElse(0.0),      // Ratio de Sortino
                            wf.splits.stream().mapToDouble(m->m.calmar).average().orElse(0.0),       // Ratio de Calmar
                            wf.splits.stream().mapToDouble(m->m.turnover).average().orElse(0.0),     // Rotation du portefeuille
                            wf.splits.stream().mapToDouble(m->m.avgBarsInPosition).average().orElse(0.0) // Durée moyenne des positions
                    );

                    // ===== MESURE DE PERFORMANCE ET LOG =====
                    long endConfig = System.currentTimeMillis();
                    logger.info("[TUNING][V2] [{}] Fin config {}/{} | meanMSE={}, PF={}, winRate={}, DD%={}, expectancy={}, businessScore={}, trades={} durée={} ms",
                            symbol, configIndex, grid.size(), meanMse, meanPF, meanWinRate, maxDDPct,
                            meanExpectancy, meanBusinessScore, totalTrades, (endConfig-startConfig));

                    // ===== MISE À JOUR DU PROGRÈS (THREAD-SAFE) =====
                    // Incrémente le compteur de configurations terminées
                    TuningProgress p = tuningProgressMap.get(symbol);
                    if (p != null) {
                        p.testedConfigs.incrementAndGet(); // Atomique: thread-safe
                        p.lastUpdate = System.currentTimeMillis(); // Heartbeat pour monitoring
                    }

                    // ===== RETOUR DU RÉSULTAT ENCAPSULÉ =====
                    // Création d'un objet résultat contenant toutes les informations nécessaires
                    return new TuningResult(config, model, scalers, meanMse, meanPF, meanWinRate, maxDDPct, meanBusinessScore);

                } catch (Exception e){
                    // ===== GESTION CENTRALISÉE DES ERREURS =====
                    // Log l'erreur sans interrompre le processus global
                    logger.error("[TUNING][V2] Erreur config {} : {}", configIndex, e.getMessage());

                    // Capture de la stack trace complète pour debug
                    String stack = org.apache.commons.lang3.exception.ExceptionUtils.getStackTrace(e);

                    // Ajout dans le rapport d'exceptions pour analyse post-mortem
                    tuningExceptionReport.add(new TuningExceptionReportEntry(symbol, config, e.getMessage(), stack, System.currentTimeMillis()));

                    // Retourne null pour signaler l'échec (sera ignoré dans la sélection)
                    return null;

                } finally {
                    // ===== NETTOYAGE MÉMOIRE CRITIQUE =====
                    // Libération explicite des références pour aider le GC
                    model = null;

                    // Nettoyage spécifique ND4J: libération des buffers GPU/CPU
                    org.nd4j.linalg.factory.Nd4j.getMemoryManager().invokeGc();

                    // Destruction des workspaces ND4J du thread courant
                    org.nd4j.linalg.factory.Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();

                    // GC système pour libération immédiate (optionnel mais utile ici)
                    System.gc();
                }
            }));
        }

        // ===== PHASE 7: FERMETURE DU POOL DE SOUMISSIONS =====
        // Empêche la soumission de nouvelles tâches (les existantes continuent)
        executor.shutdown();

        // ===== PHASE 8: COLLECTE ET ANALYSE DES RÉSULTATS =====

        // Compteurs pour statistiques finales
        int failedConfigs = 0;                                    // Nombre de configs échouées
        double bestBusinessScore = Double.NEGATIVE_INFINITY;      // Meilleur score business trouvé

        // ===== RÉCUPÉRATION SÉQUENTIELLE DES RÉSULTATS =====
        // Parcours des Future dans l'ordre de soumission (maintien de la séquence)
        for (int i = 0; i < futures.size(); i++) {
            try {
                // Récupération bloquante du résultat (attente si pas encore terminé)
                TuningResult result = futures.get(i).get();

                // Log de progression pour monitoring
                logger.info("[TUNING][V2] [{}] Progression : {}/{} configs terminées", symbol, i+1, grid.size());

                // ===== FILTRAGE DES RÉSULTATS INVALIDES =====
                // Ignore les configs qui ont échoué ou produit des scores invalides
                if (result == null || Double.isNaN(result.businessScore) || Double.isInfinite(result.businessScore)) {
                    failedConfigs++;
                    continue; // Passe à la config suivante
                }

                // ===== SÉLECTION DU MEILLEUR CANDIDAT =====
                // Critère: maximisation du businessScore (métrique composite business)
                if (result.businessScore > bestBusinessScore) {
                    bestBusinessScore = result.businessScore;   // Nouveau record
                    bestConfig = result.config;                 // Configuration gagnante
                    bestModel = result.model;                   // Modèle associé
                    bestScalers = result.scalers;               // Scalers correspondants
                    bestScore = result.score;                   // MSE pour information
                }

            } catch (Exception e) {
                // ===== GESTION D'ERREUR DE RÉCUPÉRATION =====
                failedConfigs++;
                progress.status = "erreur"; // Marque le processus en erreur
                progress.lastUpdate = System.currentTimeMillis();

                // Log de l'erreur de récupération (différent de l'erreur d'exécution)
                logger.error("Erreur lors de la récupération du résultat de tuning : {}", e.getMessage());

                // Extraction de la stack trace (cause racine si disponible)
                String stack = e.getCause() != null
                        ? org.apache.commons.lang3.exception.ExceptionUtils.getStackTrace(e.getCause())
                        : org.apache.commons.lang3.exception.ExceptionUtils.getStackTrace(e);

                // Ajout au rapport d'exceptions (config = null car erreur de récupération)
                tuningExceptionReport.add(new TuningExceptionReportEntry(symbol, null, e.getMessage(), stack, System.currentTimeMillis()));
            }
        }

        // ===== PHASE 9: ANALYSE DU SUCCÈS/ÉCHEC GLOBAL =====

        // Timestamp de fin pour calcul de durée totale
        long endSymbol = System.currentTimeMillis();

        // ===== CAS D'ÉCHEC TOTAL =====
        // Si toutes les configurations ont échoué, abandon du processus
        if (failedConfigs == grid.size()) {
            progress.status = "failed";                 // Marquage d'échec
            progress.endTime = endSymbol;               // Timestamp de fin
            progress.lastUpdate = endSymbol;            // Dernière mise à jour

            logger.error("[TUNING][V2][EARLY STOP] Toutes les configs ont échoué pour le symbole {}.", symbol);
            return null; // Aucun résultat utilisable
        }

        // ===== PHASE 10: SUCCÈS - PERSISTANCE DES RÉSULTATS =====

        // Marquage du succès du processus
        progress.status = "termine";
        progress.endTime = endSymbol;
        progress.lastUpdate = endSymbol;

        // Vérification que nous avons un résultat valide complet
        if (bestConfig != null && bestModel != null && bestScalers != null) {

            // ===== SAUVEGARDE DES HYPERPARAMÈTRES GAGNANTS =====
            // Persistance de la configuration optimale pour réutilisation
            hyperparamsRepository.saveHyperparams(symbol, bestConfig);

            try {
                // ===== SAUVEGARDE DU MODÈLE COMPLET =====
                // Sérialisation du modèle + scalers en base pour prédictions futures
                // Comprend: architecture neuronale, poids entraînés, normalisateurs
                lstmTradePredictor.saveModelToDb(symbol, bestModel, jdbcTemplate, bestConfig, bestScalers);

            } catch (Exception e) {
                // Log d'erreur non-bloquante (les hyperparamètres sont sauvés)
                logger.error("Erreur lors de la sauvegarde du meilleur modèle : {}", e.getMessage());
            }

            // ===== LOG FINAL DE SUCCÈS =====
            // Résumé complet des meilleurs résultats trouvés
            logger.info("[TUNING][V2] Fin tuning {} | Best businessScore={} | windowSize={}, neurons={}, dropout={}, lr={}, l1={}, l2={}, meanMSE={}, durée={} ms",
                    symbol, bestBusinessScore, bestConfig.getWindowSize(), bestConfig.getLstmNeurons(),
                    bestConfig.getDropoutRate(), bestConfig.getLearningRate(), bestConfig.getL1(), bestConfig.getL2(),
                    bestScore, (endSymbol - startSymbol));

        } else {
            // ===== CAS D'ÉCHEC PARTIEL =====
            // Certaines configs ont réussi mais aucun résultat valide récupéré
            logger.warn("[TUNING][V2] Aucun modèle/scaler valide trouvé pour {}", symbol);
        }

        // ===== PHASE 11: NETTOYAGE FINAL =====
        // Nettoyage global de sécurité pour libérer toute mémoire résiduelle
        try {
            // Garbage collection ND4J forcé
            org.nd4j.linalg.factory.Nd4j.getMemoryManager().invokeGc();

            // Destruction des workspaces du thread principal
            org.nd4j.linalg.factory.Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();

            // GC système final
            System.gc();

        } catch (Exception e) {
            // Log non-critique: échec de nettoyage n'affecte pas le résultat
            logger.warn("Nettoyage ND4J échec : {}", e.getMessage());
        }

        // ===== RETOUR DU RÉSULTAT FINAL =====
        // Retourne la meilleure configuration trouvée (null si aucune valide)
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
        logger.info("[TUNING] Début tuning multi-symboles ({} symboles, parallélisé)", symbols.size());
        // Pool limité pour tuning de symboles en parallèle (max moitié des threads effectifs, au moins 1)
        int maxParallelSymbols = 1;//Math.max(1, effectiveMaxThreads / 2);
        java.util.concurrent.ExecutorService symbolExecutor = java.util.concurrent.Executors.newFixedThreadPool(maxParallelSymbols);
        java.util.List<java.util.concurrent.Future<?>> futures = new java.util.ArrayList<>();
        for (int i = 0; i < symbols.size(); i++) {
            final int symbolIndex = i;
            futures.add(symbolExecutor.submit(() -> {
                waitForMemory(); // Protection mémoire avant chaque tuning de symbole
                String symbol = symbols.get(symbolIndex);
                long startSymbol = System.currentTimeMillis();
                logger.info("[TUNING] Début tuning symbole {}/{} : {} (thread={})", symbolIndex+1, symbols.size(), symbol, Thread.currentThread().getName());
                try {
                    BarSeries series = seriesProvider.apply(symbol);
                    tuneSymbolMultiThread(symbol, grid, series, jdbcTemplate);
                } catch (Exception e) {
                    logger.error("[TUNING] Erreur tuning symbole {} : {}", symbol, e.getMessage());
                } finally {
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
            }));
        }
        // Attendre la fin de toutes les tâches
        for (java.util.concurrent.Future<?> f : futures) {
            try {
                f.get();
            } catch (Exception e) {
                logger.error("[TUNING] Erreur lors de l'attente de la fin d'un tuning symbole : {}", e.getMessage());
            }
        }
        symbolExecutor.shutdown();
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
