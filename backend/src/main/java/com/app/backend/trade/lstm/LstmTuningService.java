package com.app.backend.trade.lstm;

import com.app.backend.trade.exception.AucunSplitValideException;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.annotations.Expose;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.ta4j.core.BarSeries;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.atomic.AtomicLong;

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

    // --- Critères d'acceptation Phase 2 (configurables) ---
    private static final double REL_GAIN_EPS = 1e-6; // epsilon pour éviter division par zéro
    @Value("${lstm.tuning.twoPhase.minRelativeGain:0.05}")
    private double minRelativeGain; // ex 0.05 = +5% relatif requis
    @Value("${lstm.tuning.twoPhase.minAbsoluteGain:0.02}")
    private double minAbsoluteGain; // ex 0.02 = +0.02 absolu requis

    // Dépendances injectées (services métiers)
    public final LstmTradePredictor lstmTradePredictor;      // Service d'entraînement / prédiction LSTM
    public final LstmHyperparamsRepository hyperparamsRepository; // Accès persistence hyperparamètres + métriques

    // Verrou global sauvegarde modèle (I/O disque sérialisées)
    private final Object modelSaveLock = new Object();
    // Contrôleur de concurrence GPU (sémaphore adaptative)
    private final GpuConcurrencyController gpuController = new GpuConcurrencyController();

    // --- Paramétrage dynamique du parallélisme ---
    @Value("${lstm.tuning.maxThreads:0}")
    private int configuredMaxThreads; // Si 0 => auto-calcul
    private int effectiveMaxThreads = 12; // Valeur de secours si auto-calcul échoue
    private boolean cudaBackend = false;  // Flag détecté au démarrage (backend ND4J CUDA ou non)
    @Value("${lstm.tuning.enableTwoPhase:true}")
    private boolean enableTwoPhase; // Activation tuning deux phases dans tuneAllSymbols
    public boolean isEnableTwoPhase(){ return enableTwoPhase; }
    public void setEnableTwoPhase(boolean enableTwoPhase){ this.enableTwoPhase = enableTwoPhase; }

    // === Nouveaux paramètres d'utilisation GPU ===
    @Value("${gpu.concurrency.min:1}")
    private int gpuMinConcurrency;
    @Value("${gpu.concurrency.max:8}") // Augmenté à 8 pour pousser l'utilisation GPU
    private int gpuMaxConcurrency;
    @Value("${gpu.concurrency.scaleUpThresholdPct:70.0}")
    private double gpuScaleUpThreshold;
    @Value("${gpu.concurrency.scaleDownThresholdPct:92.0}")
    private double gpuScaleDownThreshold;
    @Value("${lstm.tuning.gpu.enableStagger:true}")
    private boolean gpuEnableStagger;
    @Value("${lstm.tuning.gpu.autoBatchScale:true}")
    private boolean gpuAutoBatchScale;
    @Value("${lstm.tuning.gpu.targetBatchSize:128}") // Augmenté à 256 pour exploiter la mémoire GPU
    private int gpuTargetBatchSize;
    @Value("${lstm.tuning.gpu.scaleLearningRateOnBatch:true}")
    private boolean gpuScaleLearningRateOnBatch; // ajuste LR proportionnellement à l'augmentation batch

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
        //logger.info("[TUNING] Parallélisme effectif (auto) = {} (configuré={}, cpuCores={}, cuda={})",  effectiveMaxThreads, configuredMaxThreads, procs, cudaBackend);
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
        // Étape 19: cumul durées individuelles des configs pour stats
        public AtomicLong cumulativeConfigDurationMs = new AtomicLong(0);
        public int threadsUsed;               // Threads utilisés pour ce symbole
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
            //logger.info("ND4J backend utilisé : {}", backendName);
            cudaBackend = backendName.toLowerCase().contains("cuda") || backendName.toLowerCase().contains("jcublas");
            if (!cudaBackend) {
                logger.warn("Le backend ND4J n'est pas CUDA. Utilisation CPU uniquement.");
                logger.warn("Pour forcer CUDA : -Dorg.nd4j.linalg.defaultbackend=org.nd4j.linalg.jcublas.JCublasBackend");
            } else {
                //logger.info("Backend CUDA détecté (optimization parallélisme ajustée).");
            }
        } catch (NoClassDefFoundError | ExceptionInInitializerError t) {
            logger.error("[BOOT][ND4J] Impossible de récupérer backend/dtype: {}", t.getMessage());
        } catch (Exception e) {
            logger.error("Erreur lors de la détection du backend ND4J : {}", e.getMessage());
        } finally {
            computeEffectiveMaxThreads();
        }

        // Configuration du contrôleur GPU avant démarrage
        try {
            gpuController.configure(gpuMinConcurrency, gpuMaxConcurrency, gpuScaleUpThreshold, gpuScaleDownThreshold);
        } catch (Exception ex) {
            logger.warn("[GPU][CONF] Échec configuration dynamique: {}", ex.getMessage());
        }

        // Heartbeat: toutes les 30 secondes
        progressLoggerExecutor = Executors.newSingleThreadScheduledExecutor();
        progressLoggerExecutor.scheduleAtFixedRate(() -> {
            try {
                if (tuningProgressMap.isEmpty()) return;
                long now = System.currentTimeMillis();
                tuningProgressMap.forEach((sym, p) -> {
                    double pct = p.totalConfigs > 0 ? (100.0 * p.testedConfigs.get() / p.totalConfigs) : 0.0;
                    long idleMs = now - p.lastUpdate;
                    String pctStr = String.format("%.2f", pct);
                    //logger.info("[TUNING][HEARTBEAT] {} status={} {}/{} ({}%) idle={}s", sym, p.status, p.testedConfigs.get(), p.totalConfigs, pctStr, idleMs/1000);
                });
            } catch (Exception ex) {
                // Log en debug pour éviter bruit excessif
                //logger.debug("Heartbeat tuning erreur: {}", ex.getMessage());
            }
        }, 10, 30, TimeUnit.SECONDS);

        // Démarrage du monitoring GPU après détection backend
        if (cudaBackend) {
            gpuController.start(cudaBackend);
            /*logger.info("[GPU] Concurrence initiale min={} max={} scaleUp<{}% scaleDown>{}% targetBatch={} autoBatchScale={} stagger={} scaleLRBatch={}",
                    gpuMinConcurrency, gpuMaxConcurrency,
                    String.format(java.util.Locale.US, "%.1f", gpuScaleUpThreshold),
                    String.format(java.util.Locale.US, "%.1f", gpuScaleDownThreshold),
                    gpuTargetBatchSize, gpuAutoBatchScale, gpuEnableStagger, gpuScaleLearningRateOnBatch);
            logger.info("[GPU][MONITOR] Utilisation VRAM initiale: {}%", String.format("%.2f", gpuController.getLastVramUsagePct()));*/
        }
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
            //logger.info("[TUNING] Symbole {} déjà tuné, abandon du processus", symbol);
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
        //logger.info("[TUNING] Début du tuning V2 pour le symbole {} ({} configs) | maxThreadsEffectif={} (configuré={})", symbol, grid.size(), effectiveMaxThreads, configuredMaxThreads);

        // ===== PHASE 3: VÉRIFICATION DOUBLE SÉCURITÉ =====

        // Variable de trace pour le meilleur score MSE (information, pas utilisée pour sélection)
        double bestScore = Double.POSITIVE_INFINITY;

        // Double vérification: chercher des hyperparamètres existants en base
        // (redondant avec isSymbolAlreydyTuned mais protection supplémentaire)
        LstmConfig existing = hyperparamsRepository.loadHyperparams(symbol);
        if(existing != null){
            //logger.info("Hyperparamètres existants trouvés pour {}. Ignorer le tuning.", symbol);
            return existing; // Retourner la config existante
        }

        // ===== PHASE 4: INITIALISATION DES VARIABLES DE RÉSULTATS =====

        // Variables pour stocker la meilleure configuration trouvée
        LstmConfig bestConfig = null;              // Configuration gagnante
        MultiLayerNetwork bestModel = null;        // Modèle neuronal correspondant
        LstmTradePredictor.ScalerSet bestScalers = null; // Normalisateurs/scalers associés
        TuningResult bestResul = null;

        // ===== PHASE 5: CRÉATION DU POOL DE THREADS =====

        // Calcul du nombre optimal de threads pour ce tuning
        // Équilibrage entre: taille de grille, nombre de coeurs CPU, limite effective
        // Note: actuellement forcé à 4 pour stabilité (ligne commentée montre le calcul automatique)
        int numThreads = Math.min(Math.min(grid.size(), Runtime.getRuntime().availableProcessors()), effectiveMaxThreads);
        if (numThreads < 1) numThreads = 1;
        progress.threadsUsed = numThreads;

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

            // Auto-scale batch pour GPU
            if (cudaBackend && gpuAutoBatchScale) {
                int originalBatch = config.getBatchSize() > 0 ? config.getBatchSize() : 32;
                if (originalBatch < gpuTargetBatchSize) {
                    int newBatch = originalBatch;
                    while (newBatch < gpuTargetBatchSize) {
                        newBatch *= 2;
                        if (newBatch >= gpuTargetBatchSize) break;
                    }
                    if (newBatch > gpuTargetBatchSize) newBatch = gpuTargetBatchSize;
                    if (newBatch != originalBatch) {
                        if (gpuScaleLearningRateOnBatch && config.getLearningRate() > 0) {
                            double scaledLr = config.getLearningRate() * ((double) originalBatch / (double) newBatch);
                            config.setLearningRate(scaledLr);
                        }
                        config.setBatchSize(newBatch);
                        //logger.debug("[GPU][BATCHSCALE] {} configIdx={} batch {} -> {} lr={}", symbol, configIndex, originalBatch, newBatch, config.getLearningRate());
                    }
                }
            }

            config.setUseScalarV2(true);
            config.setUseWalkForwardV2(true);
            futures.add(executor.submit(() -> {
                MultiLayerNetwork model = null;
                boolean permitAcquired = false;
                long staggerSleepMs = 0L;
                try {
                    if (cudaBackend) {
                        gpuController.acquirePermit();
                        permitAcquired = true;
                        int active = gpuController.getActiveTrainings();
                        if (gpuEnableStagger && active > 1) { // stagger optionnel
                            staggerSleepMs = 500L + (long)(Math.random()*500L); // réduit à 0.5-1s pour maximiser GPU
                            Thread.sleep(staggerSleepMs);
                        }
                        gpuController.markTrainingStarted();
                        //logger.info("[GPU][MONITOR] Utilisation VRAM: {}% (activeTrainings={})", String.format("%.2f", gpuController.getLastVramUsagePct()), active);
                    }
                    // ===== DÉBUT DE TRAITEMENT D'UNE CONFIGURATION =====

                    // Timestamp de début pour mesure de performance par config
                    long startConfig = System.currentTimeMillis();

                    // ===== REPRODUCTIBILITÉ: INITIALISATION DES SEEDS =====
                    // Fixe les générateurs aléatoires pour résultats reproductibles
                    // Critique pour pouvoir recréer exactement les mêmes résultats
                    lstmTradePredictor.setGlobalSeeds(config.getSeed());

                    // Log de début de traitement de cette configuration
                    //logger.info("[TUNING][V2] [{}] Début config {}/{}", symbol, configIndex, grid.size());

                    // ===== SÉPARATION TRAIN/TEST POUR ÉVITER LE DATA LEAKAGE =====
                    // PROBLÈME CORRIGÉ: éviter d'entraîner et tester sur les mêmes données

                    int totalBars = series.getBarCount();
                    int testSplitRatio = 20; // 20% pour le test (out-of-sample)
                    int trainEndBar = totalBars * (100 - testSplitRatio) / 100; // 80% pour l'entraînement

                    // Vérification que nous avons assez de données
                    if (trainEndBar < config.getWindowSize() + 50) {
                        throw new IllegalStateException("Données insuffisantes après séparation train/test");
                    }

                    // ===== ENTRAÎNEMENT SUR LA PARTIE TRAIN UNIQUEMENT =====
                    // Création d'une sous-série contenant uniquement les données d'entraînement
                    BarSeries trainSeries = series.getSubSeries(0, trainEndBar);
                    //logger.debug("[TUNING][V2] [{}] Séparation données: train=[0,{}], test=[{},{}]","BTCUSDT", trainEndBar, trainEndBar, totalBars);


                    // Entraîne le modèle UNIQUEMENT sur les données d'entraînement
                    LstmTradePredictor.TrainResult trFull = lstmTradePredictor.trainLstmScalarV2(trainSeries, config);
                    model = trFull.model;                    // Modèle neuronal entraîné
                    LstmTradePredictor.ScalerSet scalers = trFull.scalers; // Normalisateurs (min/max, z-score, etc.)

                    // ===== VALIDATION WALK-FORWARD SUR DONNÉES NON VUES =====
                    // Évalue la performance du modèle via validation temporelle séquentielle
                    // IMPORTANT: Le modèle n'a été entraîné que sur trainSeries [0, trainEndBar]
                    // Cette méthode teste UNIQUEMENT sur [trainEndBar, totalBars] (données non vues)
                    LstmTradePredictor.WalkForwardResultV2 wf = lstmTradePredictor.walkForwardEvaluateOutOfSample(
                        series,           // Série complète (pour contexte historique)
                        config,
                        model,           // Modèle entraîné UNIQUEMENT sur trainSeries
                        scalers,         // Scalers calculés UNIQUEMENT sur trainSeries
                        trainEndBar      // Point de séparation: test commence à partir d'ici (20% finaux)
                    );
                    double meanMse = wf.meanMse;            // Erreur quadratique moyenne sur tous les splits

                    // ===== AGRÉGATION DES MÉTRIQUES DE TRADING =====
                    // Collecte et moyenne des métriques business sur tous les splits walk-forward
                    double sumPF=0, sumWin=0, sumExp=0, maxDrawdownPct=0, sumBusiness=0, sumProfit=0, sumBusinessSq=0; // ajout sumBusinessSq
                    int splits=0;           // Compteur de splits valides
                    int totalTrades=0;      // Nombre total de trades sur tous les splits

                    // Parcours de tous les résultats de split pour agrégation
                    for(LstmTradePredictor.TradingMetricsV2 m : wf.splits){
                        if (m.numTrades == 0 && logger.isDebugEnabled()) {
                            /*logger.info("[TUNING][NO_TRADES][V2] symbol={} cfgNeurons={} lr={} dropout={} splitIdx={} pf={} wr={} dd={} exp={} bs={}",
                                    symbol,
                                    config.getLstmNeurons(),
                                    config.getLearningRate(),
                                    config.getDropoutRate(),
                                    splits+1,
                                    m.profitFactor,
                                    m.winRate,
                                    m.maxDrawdownPct,
                                    m.expectancy,
                                    m.businessScore);*/
                        }
                        if(Double.isFinite(m.profitFactor)) sumPF += m.profitFactor; else sumPF += 0;
                        sumWin += m.winRate;
                        sumExp += m.expectancy;
                        if(m.maxDrawdownPct > maxDrawdownPct) maxDrawdownPct = m.maxDrawdownPct;
                        double bsSplit = (Double.isFinite(m.businessScore) ? m.businessScore : 0);
                        sumBusiness += bsSplit;
                        sumBusinessSq += bsSplit * bsSplit; // accumulation variance
                        sumProfit += m.totalProfit;
                        totalTrades += m.numTrades;
                        splits++;
                    }
                    if(splits==0){
                        throw new IllegalStateException("Aucun split valide walk-forward");
                    }
                    double meanPF = sumPF / splits;
                    double meanWinRate = sumWin / splits;
                    double meanExpectancy = sumExp / splits;
                    double meanBusinessScore = sumBusiness / splits;
                    double varianceBusiness = (sumBusinessSq / splits) - (meanBusinessScore * meanBusinessScore); if(varianceBusiness < 0) varianceBusiness = 0; double stdBusinessScore = Math.sqrt(varianceBusiness);
                    // DEBUG: Log détaillé des businessScore de chaque split
                    java.util.List<Double> businessScoresDebug = new java.util.ArrayList<>();
                    for(LstmTradePredictor.TradingMetricsV2 m : wf.splits) {
                        businessScoresDebug.add(m.businessScore);
                    }
                    //logger.info("[--------][DEBUG][TUNING][V2] [{}] businessScores splits: {} | sumBusiness={} | splits={} | meanBusinessScore={}",symbol, businessScoresDebug, sumBusiness, splits, meanBusinessScore);
                    double rmse = Double.isFinite(meanMse) && meanMse>=0? Math.sqrt(meanMse): Double.NaN;
                    hyperparamsRepository.saveTuningMetrics(
                            symbol, config,
                            meanMse, rmse,
                            sumProfit, meanPF, meanWinRate, maxDrawdownPct, totalTrades, meanBusinessScore,
                            wf.splits.stream().mapToDouble(m->m.sortino).average().orElse(0.0),
                            wf.splits.stream().mapToDouble(m->m.calmar).average().orElse(0.0),
                            wf.splits.stream().mapToDouble(m->m.turnover).average().orElse(0.0),
                            wf.splits.stream().mapToDouble(m->m.avgBarsInPosition).average().orElse(0.0),
                            0, 0, 0, false
                    );
                    long endConfig = System.currentTimeMillis();
                    long cfgDuration = (endConfig-startConfig);
                    /*logger.info("[TUNING][V2] [{}] Fin config {}/{} | meanMSE={}, PF={}, winRate={}, DD%={}, expectancy={}, businessScore={}, trades={} durée={} ms",
                            symbol, configIndex, grid.size(), meanMse, meanPF, meanWinRate, maxDrawdownPct,
                            meanExpectancy, meanBusinessScore, totalTrades, cfgDuration);*/
                    TuningProgress p = tuningProgressMap.get(symbol);
                    if (p != null) {
                        p.testedConfigs.incrementAndGet();
                        p.lastUpdate = System.currentTimeMillis();
                        p.cumulativeConfigDurationMs.addAndGet(cfgDuration);
                    }
                    return new TuningResult(config, model, scalers, meanMse, meanPF, meanWinRate, maxDrawdownPct, meanBusinessScore, stdBusinessScore, rmse, sumProfit, totalTrades, series.getBarCount() - trainEndBar, -1);

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
                    if (cudaBackend) {
                        gpuController.markTrainingFinished();
                        if (permitAcquired) gpuController.releasePermit();
                    }
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
        double bestAdjScore = Double.NEGATIVE_INFINITY;           // Nouveau : suivi score ajusté (anti overfit winRate)

        // ===== RÉCUPÉRATION SÉQUENTIELLE DES RÉSULTATS =====
        // Parcours des Future dans l'ordre de soumission (maintien de la séquence)
        for (int i = 0; i < futures.size(); i++) {
            try {
                // Récupération bloquante du résultat (attente si pas encore terminé)
                TuningResult result = futures.get(i).get();

                // Log de progression pour monitoring
                //logger.info("[TUNING][V2] [{}] Progression : {}/{} configs terminées", symbol, i+1, grid.size());

                // ===== FILTRAGE DES RÉSULTATS INVALIDES =====
                // Ignore les configs qui ont échoué ou produit des scores invalides
                if (result == null || Double.isNaN(result.businessScore) || Double.isInfinite(result.businessScore)) {
                    failedConfigs++;
                    continue; // Passe à la config suivante
                }

                double adj = adjScore(result);
                if (adj > bestAdjScore) {
                    bestAdjScore = adj;
                    bestBusinessScore = result.businessScore;   // conserve la valeur brute pour reporting
                    bestConfig = result.config;
                    bestModel = result.model;
                    bestScalers = result.scalers;
                    bestScore = result.score;
                    bestResul = result;
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
            hyperparamsRepository.saveHyperparams(symbol, bestConfig, 0);

            try {
                // ===== SAUVEGARDE DU MODÈLE COMPLET =====
                // Sérialisation du modèle + scalers en base pour prédictions futures
                // Comprend: architecture neuronale, poids entraînés, normalisateurs
                synchronized (modelSaveLock) { // Sérialisation disque synchronisée
                    lstmTradePredictor.saveModelToDb(symbol, jdbcTemplate, bestModel, bestConfig, bestScalers,
                        bestScore,  bestResul.profitFactor,  bestResul.winRate,  bestResul.maxDrawdown,  bestResul.rmse,  bestResul.sumProfit,  bestResul.totalTrades,  bestResul.businessScore,
                        bestResul.totalSeriesTested, -1, -1, -1, "", false, "", 0);
                }
            } catch (Exception e) {
                // Log d'erreur non-bloquante (les hyperparamètres sont sauvés)
                logger.error("Erreur lors de la sauvegarde du meilleur modèle : {}", e.getMessage());
            }

            // ===== LOG FINAL DE SUCCÈS =====
            // Résumé complet des meilleurs résultats trouvés
            /*logger.info("[TUNING][V2] Fin tuning {} | Best businessScore={} (adj={}) | windowSize={}, neurons={}, dropout={}, lr={}, l1={}, l2={}, meanMSE={}, durée={} ms",
                    symbol, bestBusinessScore, bestAdjScore, bestConfig.getWindowSize(), bestConfig.getLstmNeurons(),
                    bestConfig.getDropoutRate(), bestConfig.getLearningRate(), bestConfig.getL1(), bestConfig.getL2(),
                    bestScore, (endSymbol - startSymbol));*/
        // Étape 19: écrire métriques JSON
            try { writeProgressMetrics(progress); } catch (Exception ex) { logger.warn("[TUNING][METRICS] Échec écriture JSON: {}", ex.getMessage()); }
        } else {
            logger.warn("[TUNING][V2] Aucun modèle/scaler valide trouvé pour {}", symbol);
            try { writeProgressMetrics(progress); } catch (Exception ex) { logger.warn("[TUNING][METRICS] Échec écriture JSON: {}", ex.getMessage()); }
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
        double[] dropoutRates = {0.2, 0.25}; // Étape 9: max 0.25
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
     * Génère une grille aléatoire de configurations adaptée au swing trade enrichie.
     * @param n nombre de configurations à générer
     * @param horizonBars tableau des horizons de prédiction à tester
     * @param numLstmLayers tableau du nombre de couches LSTM
     * @param batchSizes tableau des batch sizes
     * @param bidirectionals tableau des valeurs bidirectional
     * @param attentions tableau des valeurs attention
     * @return liste de LstmConfig à tester
     */
    public List<LstmConfig> generateRandomSwingTradeGrid(int n, int[] horizonBars, int[] numLstmLayers, int[] batchSizes, boolean[] bidirectionals, boolean[] attentions) {
        List<LstmConfig> grid = new java.util.ArrayList<>();
        int[] windowSizes = {10, 20, 30};
        int[] lstmNeurons = {64, 128};
        double[] dropoutRates = {0.2, 0.25};
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
        double[] dropoutRates = {0.2, 0.25}; // Étape 9: max 0.25
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
            config.setL2(rand.nextDouble() * 0.001); // Valeurs L2 aléatoires entre 0 et 0.001
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
        double[] dropoutRates = {0.2, 0.25}; // Étape 9: max 0.25
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
            config.setL2(rand.nextDouble() * 0.001); // Valeurs L2 aléatoires entre 0 et 0.001
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
     * Nouvelle génération optimisée pour budget réduit (n ~ 50).
     * Objectif: maximiser la diversité utile (exploration) tout en concentrant une partie des
     * configurations sur une zone déjà jugée performante (exploitation) et quelques idées plus
     * audacieuses (innovation) sans explosion du coût mémoire.
     *
     * Stratégie:
     *  - exploitation: fenêtres moyennes (30-55), neurones 128-256, faible LR log-uniform (2e-4..6e-4)
     *  - exploration: couvre fenêtres plus larges + petites, variantes horizon (3,5,7,10), couches 1-3
     *  - innovation: ajoute attention ou bidirectionnel ponctuellement + dropout plus bas (0.15) ou haut (0.28)
     *  - learningRate & l2: tirage log-uniform pour meilleur spread
     *  - numEpochs adaptatif (exploitation>exploration>innovation) pour limiter temps global
     *
     * Répartition par défaut (si n >= 15):
     *  - exploitation: 40%
     *  - exploration: 40%
     *  - innovation: 20%
     * Si n plus petit, on réduit proportionnellement mais on conserve au moins 1 par catégorie quand possible.
     *
     * NOTE: ne remplace PAS les anciennes méthodes (backward compatible). Utilisée explicitement par LstmHelper.
     */
    public List<LstmConfig> generateRandomSwingTradeGridOptimized(int n) {
        java.util.Random rand = new java.util.Random(System.currentTimeMillis());
        if (n < 3) n = 3; // sécurité
        int exploit = Math.max(1, (int) round(n * 0.4));
        int explore = Math.max(1, (int) round(n * 0.4));
        int innovate = n - exploit - explore;
        if (innovate < 1) { innovate = 1; if (explore > exploit) explore--; else exploit--; }
        List<LstmConfig> grid = new java.util.ArrayList<>(n);

        // ---- Helpers (mis à jour pour plus d'amplitude) ----
        java.util.function.DoubleSupplier lrExploitSampler = () -> round(sampleLogUniform(rand, 2e-4, 8e-4));
        java.util.function.DoubleSupplier lrExploreSampler = () -> round(sampleLogUniform(rand, 2e-4, 1.8e-3));
        java.util.function.DoubleSupplier lrInnovSampler   = () -> round(sampleLogUniform(rand, 1.5e-4, 2.0e-3));
        java.util.function.DoubleSupplier l2SamplerTight = () -> round(sampleLogUniform(rand, 5e-5, 4e-4));
        java.util.function.DoubleSupplier l2SamplerWide  = () -> round(sampleLogUniform(rand, 1e-5, 1e-3));
        java.util.function.IntSupplier horizonExploitSampler = () -> new int[]{7,9,12}[rand.nextInt(3)];
        java.util.function.IntSupplier horizonExploreSampler = () -> new int[]{5,7,9,12,15}[rand.nextInt(5)];
        java.util.function.IntSupplier horizonInnovSampler   = () -> new int[]{9,12,15}[rand.nextInt(3)];

        // ---- Exploitation (focus zone performante mais moins conservatrice) ----
        int[] winExp = {30, 35, 45, 55};
        int[] neuExp = {64, 96, 128}; // Limite à 96 pour mémoire
        int[] batchExp = {48, 64}; // batch augmenté
        for (int i = 0; i < exploit; i++) {
            LstmConfig c = baseConfigSkeleton();
            c.setWindowSize(winExp[rand.nextInt(winExp.length)]);
            c.setLstmNeurons(neuExp[rand.nextInt(neuExp.length)]);
            c.setNumLstmLayers(rand.nextDouble() < 0.55 ? 2 : 1); // max 2 couches
            double[] dChoices = {0.12,0.15,0.18,0.20}; // Dropout plus prudent
            c.setDropoutRate(dChoices[rand.nextInt(dChoices.length)]);
            c.setLearningRate(lrExploitSampler.getAsDouble());
            c.setL2(l2SamplerTight.getAsDouble());
            c.setHorizonBars(horizonExploitSampler.getAsInt());
            // Attention/bidirectional limité à 10% des configs
            boolean att = rand.nextDouble() < 0.10;
            boolean bidir = !att && rand.nextDouble() < 0.10;
            c.setAttention(att);
            c.setBidirectional(bidir);
            c.setNumEpochs(100 + rand.nextInt(21)); // 50-70 époques
            c.setPatience(5 + rand.nextInt(4)); // 5-8
            c.setMinDelta(0.00018);
            c.setBatchSize(batchExp[rand.nextInt(batchExp.length)]);
            c.setSwingTradeType(pick(rand, new String[]{"range","mean_reversion"}, 0.6));
            c.setBusinessProfitFactorCap(4.5);
            c.setBusinessDrawdownGamma(1.4);
            c.setWalkForwardSplits(4 + rand.nextInt(2)); // 4-5 splits
            grid.add(c);
        }
        // ---- Exploration (diversité structurelle) ----
        int[] winExplore = {18, 25, 30, 45, 60}; // max 30
        int[] neuExplore = {64, 96, 128, 160}; // max 96
        int[] layersExplore = {1,2}; // max 2 couches
        int[] batchExplore = {32, 48, 64}; // batch augmenté
        for (int i = 0; i < explore; i++) {
            LstmConfig c = baseConfigSkeleton();
            c.setWindowSize(winExplore[rand.nextInt(winExplore.length)]);
            c.setLstmNeurons(neuExplore[rand.nextInt(neuExplore.length)]);
            c.setNumLstmLayers(layersExplore[rand.nextInt(layersExplore.length)]);
            c.setDropoutRate(round(0.10 + rand.nextDouble()*0.10));
            c.setLearningRate(lrExploreSampler.getAsDouble());
            c.setL2(l2SamplerWide.getAsDouble());
            c.setHorizonBars(horizonExploreSampler.getAsInt());
            boolean att = rand.nextDouble() < 0.10;
            boolean bidir = !att && rand.nextDouble() < 0.10;
            c.setAttention(att);
            c.setBidirectional(bidir);
            c.setNumEpochs(40 + rand.nextInt(21)); // 40-60 époques
            c.setPatience(5 + rand.nextInt(4)); // 5-8
            c.setMinDelta(0.00022);
            c.setBatchSize(batchExplore[rand.nextInt(batchExplore.length)]);
            c.setSwingTradeType(pick(rand, new String[]{"range","breakout","mean_reversion"}, 0.34));
            if (rand.nextDouble() < 0.20) c.setNormalizationScope("global");
            c.setBusinessProfitFactorCap(5.0);
            c.setBusinessDrawdownGamma(1.3);
            c.setWalkForwardSplits(4 + rand.nextInt(2)); // 4-5 splits
            grid.add(c);
        }

        // ---- Innovation (idées agressives contrôlées) ----
        int[] winInnov = {15, 30, 60}; // max 20
        int[] neuInnov = {6, 128}; // max 64
        int[] batchInnov = {48, 64}; // batch augmenté
        for (int i = 0; i < innovate; i++) {
            LstmConfig c = baseConfigSkeleton();
            c.setWindowSize(winInnov[rand.nextInt(winInnov.length)]);
            c.setLstmNeurons(neuInnov[rand.nextInt(neuInnov.length)]);
            c.setNumLstmLayers(1); // jamais plus d'une couche
            double[] dd = {0.08, 0.12, 0.18};
            c.setDropoutRate(dd[rand.nextInt(dd.length)]);
            c.setLearningRate(lrInnovSampler.getAsDouble());
            c.setL2(l2SamplerWide.getAsDouble());
            c.setHorizonBars(horizonInnovSampler.getAsInt());
            boolean att = rand.nextDouble() < 0.10;
            boolean bidir = !att && rand.nextDouble() < 0.10;
            c.setAttention(att);
            c.setBidirectional(bidir);
            c.setNumEpochs(40 + rand.nextInt(21)); // 40-60 époques
            c.setPatience(5 + rand.nextInt(4)); // 5-8
            c.setMinDelta(0.00028);
            c.setBatchSize(batchInnov[rand.nextInt(batchInnov.length)]);
            c.setSwingTradeType(pick(rand, new String[]{"breakout","mean_reversion"}, 0.5));
            c.setBusinessProfitFactorCap(5.0);
            c.setBusinessDrawdownGamma(1.2);
            c.setWalkForwardSplits(4 + rand.nextInt(2)); // 4-5 splits
            grid.add(c);
        }

        java.util.Collections.shuffle(grid, rand);
        long baseSeed = System.currentTimeMillis();
        for (int i = 0; i < grid.size(); i++) {
            grid.get(i).setSeed(baseSeed + i);
        }
        return grid;
    }

    // Helper spécifique innovation batch 48 (évite magic number multiple)
    private int FortyEightOr(int choiceA, int choiceB, java.util.Random r){ return r.nextDouble()<0.5?choiceA:choiceB; }

    // --- Helpers ajoutés pour génération optimisée ---
    private double sampleLogUniform(java.util.Random r, double min, double max){
        if(min<=0||max<=0||max<=min) return min;
        double logMin = Math.log(min);
        double logMax = Math.log(max);
        return Math.exp(logMin + r.nextDouble() * (logMax - logMin));
    }
    private double round(double v){ return Math.round(v * 1e9)/1e9; }
    private String pick(java.util.Random r, String[] options, double pFirst){
        if(options==null||options.length==0) return "";
        if(options.length==1) return options[0];
        if(r.nextDouble() < pFirst) return options[0];
        return options[1 + r.nextInt(options.length-1)];
    }
    private LstmConfig baseConfigSkeleton(){
        LstmConfig c = new LstmConfig();
        c.setUseScalarV2(true);
        c.setUseWalkForwardV2(true);
        c.setBaselineReplica(false); // par défaut non baseline
        c.setCvMode("split");
        c.setBusinessProfitFactorCap(4.0);
        c.setBusinessDrawdownGamma(1.3);
        return c;
    }


    /**
     * Méthode de tuning pour tous les symboles donnés.
     *
     * Cette méthode utilise un parallélisme maximal pour tuner plusieurs symboles en même temps.
     * Chaque symbole est traité par un thread séparé, avec un contrôle de la mémoire pour éviter
     * les erreurs OutOfMemory.
     *
     * Flux général:
     *  1. Pour chaque symbole:
     *       - Vérifie la mémoire disponible (waitForMemory)
     *       - Démarre le tuning en appelant tuneSymbolMultiThread ou tuneSymbolTwoPhase
     *       - Gère les exceptions et nettoie les ressources
     *  2. Attente de la fin de tous les tunings
     *  3. Rapport des statistiques finales (succès/échecs)
     *
     * Important:
     *  - Ne pas modifier sans comprendre l'impact sur le parallélisme et la gestion mémoire.
     *  - Les méthodes tuneSymbolMultiThread et tuneSymbolTwoPhase gèrent déjà leur propre
     *    parallélisme interne, donc pas besoin de le gérer ici.
     *
     * @param symbols liste des symboles à tuner
     * @param grid grille d'hyperparamètres à tester
     * @param jdbcTemplate accès base de données
     * @param seriesProvider fournisseur de séries temporelles pour chaque symbole
     */
    public void tuneAllSymbols(List<String> symbols, List<LstmConfig> grid, JdbcTemplate jdbcTemplate, java.util.function.Function<String, BarSeries> seriesProvider) {
        long startAll = System.currentTimeMillis();
        //logger.info("[TUNING] Début tuning multi-symboles ({} symboles, parallélisé) | twoPhase={} ", symbols.size(), enableTwoPhase);
        int maxParallelSymbols = Math.max(1, effectiveMaxThreads);
        java.util.concurrent.ExecutorService symbolExecutor = java.util.concurrent.Executors.newFixedThreadPool(maxParallelSymbols);
        java.util.List<java.util.concurrent.Future<?>> futures = new java.util.ArrayList<>();
        for (int i = 0; i < symbols.size(); i++) {
            final int symbolIndex = i;
            futures.add(symbolExecutor.submit(() -> {
                waitForMemory(); // Protection mémoire avant chaque tuning de symbole
                String symbol = symbols.get(symbolIndex);
                long startSymbol = System.currentTimeMillis();
                //logger.info("[TUNING] Début tuning symbole {}/{} : {} (thread={})", symbolIndex+1, symbols.size(), symbol, Thread.currentThread().getName());
                try {
                    BarSeries series = seriesProvider.apply(symbol);
                    if (enableTwoPhase) {
                        // Utilise la grille comme grille coarse de phase 1
                        tuneSymbolTwoPhase(symbol, grid, series, jdbcTemplate);
                    } else {
                        tuneSymbolMultiThread(symbol, grid, series, jdbcTemplate);
                    }
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
                    //logger.info("[TUNING] Fin tuning symbole {} | durée={} ms", symbol, (endSymbol - startSymbol));
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
        //logger.info("[TUNING] Fin tuning multi-symboles | durée totale={} ms", (endAll - startAll));
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
        LstmConfig config;
        MultiLayerNetwork model;
        LstmTradePredictor.ScalerSet scalers;
        @Expose
        int top;
        @Expose
        String topLabel;
        @Expose
        double score;
        @Expose
        double profitFactor;
        @Expose
        double winRate;
        @Expose
        double maxDrawdown;
        @Expose
        double businessScore;
        @Expose
        double businessScoreStd; // nouvel écart-type des businessScore sur les splits WF
        @Expose
        double rmse;
        @Expose
        int numberGrid;
        @Expose
        int numberGridTop;
        @Expose
        double sumProfit;
        @Expose
        int totalSeriesTested;
        @Expose
        int totalTrades;
        TuningResult(LstmConfig c, MultiLayerNetwork m, LstmTradePredictor.ScalerSet s, double score, double pf, double wr, double dd, double bs, double bsStd, double rmse, double sumProfit, int totalTrades, int totalSeriesTested, int numberGrid)
        { this.numberGrid = numberGrid;this.config=c; this.model=m; this.scalers=s; this.score=score; this.profitFactor=pf; this.winRate=wr; this.maxDrawdown=dd; this.businessScore=bs; this.businessScoreStd=bsStd; this.rmse = rmse; this.sumProfit=sumProfit; this.totalTrades=totalTrades; this.totalSeriesTested=totalSeriesTested;}
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

    private static final Object METRICS_FILE_LOCK = new Object();
    private static final Path METRICS_PATH = Path.of("tuning_progress_metrics.json");
    private static final DateTimeFormatter ISO_FMT = DateTimeFormatter.ISO_OFFSET_DATE_TIME;

    // --- Ajout Hold-Out Final (jamais vu pendant phase1/phase2) ---
    private static final double HOLD_OUT_FRACTION = 0.10; // 10% des données réservées
    private static final int MIN_HOLD_OUT_BARS = 200;     // minimum absolu
    private static final int FINAL_HOLD_OUT_PHASE = 100;   // identifiant de phase pour persistance finale

    private void writeProgressMetrics(TuningProgress progress) throws Exception {
        if (progress == null) return;
        double durationMs = (progress.endTime > 0 ? progress.endTime : System.currentTimeMillis()) - progress.startTime;
        double configsDone = Math.max(1, progress.testedConfigs.get());
        double cfgPerSec = (durationMs > 0) ? (configsDone / (durationMs/1000.0)) : 0.0;
        double meanCfgMs = progress.cumulativeConfigDurationMs.get() > 0 ? (progress.cumulativeConfigDurationMs.get() / configsDone) : 0.0;
        String startIso = ISO_FMT.format(Instant.ofEpochMilli(progress.startTime).atOffset(ZoneOffset.UTC));
        String endIso = (progress.endTime>0)? ISO_FMT.format(Instant.ofEpochMilli(progress.endTime).atOffset(ZoneOffset.UTC)) : "";
        String entry = "{"+
                "\"symbol\":\""+progress.symbol+"\","+
                "\"status\":\""+progress.status+"\","+
                "\"totalConfigs\":"+progress.totalConfigs+","+
                "\"testedConfigs\":"+progress.testedConfigs.get()+","+
                "\"durationMs\":"+ (long)durationMs +","+
                "\"configsPerSecond\":"+String.format(java.util.Locale.US,"%.4f",cfgPerSec)+","+
                "\"meanConfigDurationMs\":"+ (long)meanCfgMs +","+
                "\"threadsUsed\":"+progress.threadsUsed+","+
                "\"effectiveMaxThreads\":"+effectiveMaxThreads+"," +
                "\"startTime\":\""+startIso+"\","+
                "\"endTime\":\""+endIso+"\"}";
        synchronized (METRICS_FILE_LOCK) {
            if (!Files.exists(METRICS_PATH)) {
                Files.writeString(METRICS_PATH, "[\n"+entry+"\n]", StandardOpenOption.CREATE, StandardOpenOption.WRITE);
            } else {
                // Append inside JSON array (simple approach: read, strip ending, append)
                String content = Files.readString(METRICS_PATH);
                String trimmed = content.trim();
                if (trimmed.endsWith("]")) {
                    int idx = trimmed.lastIndexOf(']');
                    String head = trimmed.substring(0, idx).trim();
                    if (head.endsWith("[")) {
                        // empty array
                        String newJson = "[\n"+entry+"\n]";
                        Files.writeString(METRICS_PATH, newJson, StandardOpenOption.TRUNCATE_EXISTING);
                    } else {
                        if (head.endsWith("}")) head += ","; // ensure comma
                        String newJson = head + "\n" + entry + "\n]";
                        Files.writeString(METRICS_PATH, newJson, StandardOpenOption.TRUNCATE_EXISTING);
                    }
                } else {
                    // Corrupted file -> recreate
                    Files.writeString(METRICS_PATH, "[\n"+entry+"\n]", StandardOpenOption.TRUNCATE_EXISTING);
                }
            }
        }
        //logger.info("[TUNING][METRICS] Mise à jour tuning_progress_metrics.json pour {} (configsPerSec={})", progress.symbol, String.format(java.util.Locale.US, "%.3f", cfgPerSec));
    }

    /**
     * Tuning en deux phases (Étape 20):
     * Phase 1 : grille coarse fournie (random ou déterministe). On évalue toutes les configs sans persister immédiatement le meilleur modèle.
     * Phase 2 : micro-grille générée autour des 5 meilleures configs (businessScore ajusté = businessScore * (1 - maxDrawdown)).
     * Variations (affinage réduit):
     *   - lstmNeurons ±16
     *   - learningRate * {0.9, 1.0, 1.1} (borné [1e-5, 0.01])
     *   - dropout ±0.03 (borné [0.05, 0.40])
     * Acceptation : phase 2 retenue si relativeGain >= minRelativeGain ET absoluteGain >= minAbsoluteGain
     * (paramétrables via properties: lstm.tuning.twoPhase.minRelativeGain / minAbsoluteGain)
     * Sinon persistance du meilleur de la phase 1.
     *
     * Remarque :
     *  - On réutilise largement la logique de tuneSymbolMultiThread (duplication contrôlée pour limiter refactor risqué)
     *  - Pas de sauvegarde hyperparams / modèle avant la décision finale => évite blocage par isSymbolAlreydyTuned
     */
    public LstmConfig tuneSymbolTwoPhase(String symbol, List<LstmConfig> coarseGrid, BarSeries series, JdbcTemplate jdbcTemplate) {
        boolean isOnlyPhase1 = true;

        if (isSymbolAlreydyTuned(symbol, jdbcTemplate)) {
            //logger.info("[TUNING-2PH] Symbole {} déjà tuné – abandon", symbol);
            return null;
        }
        if (coarseGrid == null || coarseGrid.isEmpty()) {
            logger.warn("[TUNING-2PH] Grille initiale vide pour {}", symbol);
            return null;
        }
        // Détermination du segment hold-out final (non utilisé pour la sélection hyperparams)
        int totalBars = series.getBarCount();
        int requestedHoldOut = Math.max((int)(totalBars * HOLD_OUT_FRACTION), MIN_HOLD_OUT_BARS);
        if (requestedHoldOut > totalBars / 3) requestedHoldOut = totalBars / 3; // limite sécurité
        int holdOutStart = totalBars - requestedHoldOut;
        boolean enableHoldOut = holdOutStart > (coarseGrid.get(0).getWindowSize() + 60);
        BarSeries phaseSeries = enableHoldOut ? series.getSubSeries(0, holdOutStart) : series;
        if (enableHoldOut) {
            //logger.info("[TUNING-2PH][HOLDOUT] Activation hold-out: {} barres ({}..{} exclus pour phases 1/2)", requestedHoldOut, holdOutStart, totalBars-1);
        } else {
            logger.warn("[TUNING-2PH][HOLDOUT] Données insuffisantes – pas de hold-out (fallback classique)");
        }

        long startSymbol = System.currentTimeMillis();
        TuningProgress progress = new TuningProgress();
        progress.symbol = symbol;
        progress.totalConfigs = coarseGrid.size();
        progress.startTime = startSymbol;
        progress.lastUpdate = startSymbol;
        progress.status = "phase1";
        tuningProgressMap.put(symbol, progress);
        try {
            // Contrôle VRAM avant lancement du job LSTM (phase 1)
            double vramPct = gpuController.getLastVramUsagePct();
            if (vramPct > 90.0) {
                logger.error("[GPU][VRAM] Saturation détectée ({}%) : lancement LSTM bloqué.", vramPct);
                progress.status = "vram_failed";
                progress.endTime = System.currentTimeMillis();
                progress.lastUpdate = progress.endTime;
                try { writeProgressMetrics(progress); } catch (Exception ignored) {}
                return null;
            }
            // Phase 1
            //logger.info("[TUNING-2PH][PHASE1] Début phase1 ({} configs) série utilisée={} (holdOutStart={})", coarseGrid.size(), phaseSeries.getBarCount(), enableHoldOut?holdOutStart:-1);
            PhaseAggregate phase1 = runPhaseNoPersist(symbol, coarseGrid, phaseSeries, 1, "PHASE1", progress);
            if (phase1 == null || phase1.bestConfig == null || phase1.allResults == null || phase1.allResults.isEmpty()) {
                progress.status = "failed"; progress.endTime = System.currentTimeMillis(); progress.lastUpdate = progress.endTime;
                logger.error("[TUNING-2PH] Erreur globale {} : {}", symbol, "aucun résultat de phase 1");
                try { writeProgressMetrics(progress); } catch (Exception ignored) {}
                return null;
            }

            // Top N pour micro-grille
            java.util.List<TuningResult> sorted = new java.util.ArrayList<>(phase1.allResults);
            sorted.sort((a,b)->Double.compare(adjScore(b), adjScore(a)));

            java.util.List<TuningResult> sortedRendement = new java.util.ArrayList<>(phase1.allResults);
            sortedRendement.sort((a,b)->Double.compare(adjScoreRendement(b), adjScoreRendement(a)));

            int topN = Math.min(5, sorted.size());
            java.util.List<TuningResult> top = new java.util.ArrayList<>();
            for(int it = 0; it < topN; it++){
                TuningResult trClone = cloneTuningResult(sorted.get(it));
                trClone.top = it+1;
                trClone.topLabel = "BS";
                top.add(trClone);
            }

            int topRN = Math.min(5, sortedRendement.size());
            java.util.List<TuningResult> topRendement = new java.util.ArrayList<>();
            for(int itr = 0; itr < topRN; itr++){
                TuningResult trClone = cloneTuningResult(sortedRendement.get(itr));
                trClone.top = itr+1;
                trClone.topLabel = "BSR";
                topRendement.add(trClone);
            }
            // Fusionne les deux listes sans doublons (basé sur l'objet TuningResult)
            //java.util.Set<TuningResult> fusion = new java.util.LinkedHashSet<>();
            List<TuningResult> fusionList = new ArrayList<>();
            fusionList.addAll(top);
            fusionList.addAll(topRendement);

            java.util.Set<String> dedup = new java.util.HashSet<>();
            java.util.List<LstmConfig> microGrid = new java.util.ArrayList<>();
            if(!isOnlyPhase1){
                int indexTop = 0;
                int cpt = 0;
                for (TuningResult tr : top) {
                    indexTop++;
                    LstmConfig base = tr.config; int baseNeu = base.getLstmNeurons();
                    int[] neuVar = {baseNeu-16, baseNeu, baseNeu+16};
                    double[] lrVar = {base.getLearningRate()*0.9, base.getLearningRate(), base.getLearningRate()*1.15}; // élargir 1.15
                    double[] drVar = {base.getDropoutRate()-0.05, base.getDropoutRate(), base.getDropoutRate()+0.04}; // variations plus larges
                    for (int nv : neuVar) {
                        if (nv<16||nv>512) continue;
                        for (double lr: lrVar){
                            cpt++;
                            lr=Math.max(1e-5, Math.min(0.02, lr)); // plafond un peu plus haut pour dynamisme
                            for(double dr:drVar){
                                dr=Math.max(0.05, Math.min(0.40, dr));
                                LstmConfig c=cloneConfig(base);
                                c.setIndexTop(indexTop);
                                c.setLstmNeurons(nv);
                                c.setLearningRate(lr);
                                c.setDropoutRate(dr);
                                if(dedup.add(keyOf(c))) microGrid.add(c);
                            }
                        }
                    }
                }
                // Ajustements agressifs micro-grille (sauf baseline)
                for (LstmConfig c : microGrid) {
                    if (!c.isBaselineReplica()) {
                        c.setAggressivenessBoost(1.3);
                        c.setLimitPredictionPct(0.0);
                        c.setEntryPercentileQuantile(Math.max(0.45, c.getEntryPercentileQuantile() - 0.05));
                        c.setDeadzoneFactor(Math.max(0.12, c.getDeadzoneFactor() * 0.6));
                        c.setL2(Math.max(0.0, c.getL2() * 0.5)); // réduire régularisation
                    }
                }
            }

            progress.totalConfigs += microGrid.size();
            try { writeProgressMetrics(progress); } catch (Exception ignored) {}
            if(isOnlyPhase1){
                for(TuningResult tuR : fusionList){
                    HoldOutEval holdOutEval = evaluateHoldOutFull(symbol, tuR.config, tuR.model, tuR.scalers, series, holdOutStart);
                    if (holdOutEval != null) {
                        PhaseAggregate finalAg = new PhaseAggregate();
                        finalAg.bestConfig = tuR.config;
                        finalAg.bestModel = holdOutEval.model;
                        finalAg.bestScalers = holdOutEval.scalers;
                        finalAg.bestBusinessScore = holdOutEval.businessScore;
                        finalAg.bestMse = holdOutEval.meanMse;
                        finalAg.profitFactor = holdOutEval.profitFactor;
                        finalAg.winRate = holdOutEval.winRate;
                        finalAg.maxDrawdown = holdOutEval.maxDrawdown;
                        finalAg.rmse = holdOutEval.rmse;
                        finalAg.sumProfit = holdOutEval.sumProfit;
                        finalAg.totalTrades = holdOutEval.totalTrades;
                        finalAg.totalSeriesTested = holdOutEval.totalSeriesTested;
                        finalAg.phaseGrid = 1;
                        finalAg.numberGrid = tuR.numberGrid;
                        finalAg.phase1TopN = tuR.top;
                        finalAg.holdOut = true;
                        finalAg.phase1TopNLabel = tuR.topLabel;
                        Gson gson = new GsonBuilder().excludeFieldsWithoutExposeAnnotation().create();
                        finalAg.tuningResult = gson.toJson(tuR);
                        persistBest(symbol, finalAg, jdbcTemplate, 0);
                        progress.status = "termine"; progress.endTime = System.currentTimeMillis(); progress.lastUpdate = progress.endTime; try { writeProgressMetrics(progress);} catch(Exception ignored) {}

                    }
                }
                return null;
            }else if (microGrid.isEmpty()) {
                logger.warn("[TUNING-2PH][PHASE2] Micro-grille vide – validation directe phase1");
                if (enableHoldOut) {
                    HoldOutEval ho1 = evaluateHoldOutFull(symbol, phase1.bestConfig, phase1.bestModel, phase1.bestScalers, series, holdOutStart);
                    if (ho1 != null) {
                        PhaseAggregate finalAg = new PhaseAggregate();
                        finalAg.bestConfig = phase1.bestConfig;
                        finalAg.bestModel = ho1.model;
                        finalAg.bestScalers = ho1.scalers;
                        finalAg.bestBusinessScore = ho1.businessScore;
                        finalAg.bestMse = ho1.meanMse;
                        finalAg.profitFactor = ho1.profitFactor;
                        finalAg.winRate = ho1.winRate;
                        finalAg.maxDrawdown = ho1.maxDrawdown;
                        finalAg.rmse = ho1.rmse;
                        finalAg.sumProfit = ho1.sumProfit;
                        finalAg.totalTrades = ho1.totalTrades;
                        finalAg.totalSeriesTested = ho1.totalSeriesTested;
                        finalAg.phaseGrid = 1;
                        finalAg.numberGrid = phase1.numberGrid;
                        finalAg.phase1TopN = 1;
                        finalAg.holdOut = true;
                        persistBest(symbol, finalAg, jdbcTemplate, 0);
                        progress.status = "termine"; progress.endTime = System.currentTimeMillis(); progress.lastUpdate = progress.endTime; try { writeProgressMetrics(progress);} catch(Exception ignored) {}
                        return phase1.bestConfig;
                    }
                }
                persistBest(symbol, phase1, jdbcTemplate, 0);
                progress.status = "termine"; progress.endTime = System.currentTimeMillis(); progress.lastUpdate = progress.endTime; try { writeProgressMetrics(progress);} catch(Exception ignored) {}
                return phase1.bestConfig;
            }

            progress.status = "phase2"; progress.lastUpdate = System.currentTimeMillis();
            try { writeProgressMetrics(progress); } catch (Exception ignored) {}

            PhaseAggregate phase2 = runPhaseNoPersist(symbol, microGrid, phaseSeries, 2, "PHASE2", progress);
            // --- LOG DISTRIBUTION PHASE 2 ---
            if (phase2 != null && phase2.allResults != null && !phase2.allResults.isEmpty()) {
                java.util.List<Double> scoresPhase2 = new java.util.ArrayList<>();
                for (TuningResult r : phase2.allResults) {
                    if (r != null && Double.isFinite(r.businessScore)) scoresPhase2.add(r.businessScore);
                }
                if (!scoresPhase2.isEmpty()) {
                    double sumP2 = 0d; for (double v : scoresPhase2) sumP2 += v;
                    double meanP2 = sumP2 / scoresPhase2.size();
                    double varP2 = 0d; for (double v : scoresPhase2) { double d = v - meanP2; varP2 += d*d; }
                    varP2 /= scoresPhase2.size();
                    double stdP2 = Math.sqrt(varP2);
                    scoresPhase2.sort(java.util.Comparator.reverseOrder());
                    java.util.List<Double> top5P2 = scoresPhase2.subList(0, Math.min(5, scoresPhase2.size()));
                    double maxP2 = scoresPhase2.get(0);
                    double minP2 = scoresPhase2.get(scoresPhase2.size()-1);
                    /*logger.info("[TUNING-2PH][PHASE2][DISTRIB] {} n={} mean={} std={} min={} max={} top5={} (dispersionRatio={})", symbol,
                            scoresPhase2.size(),
                            String.format(java.util.Locale.US, "%.6f", meanP2),
                            String.format(java.util.Locale.US, "%.6f", stdP2),
                            String.format(java.util.Locale.US, "%.6f", minP2),
                            String.format(java.util.Locale.US, "%.6f", maxP2),
                            top5P2,
                            String.format(java.util.Locale.US, "%.3f", (meanP2 != 0 ? stdP2 / Math.abs(meanP2) : Double.NaN)));*/
                    // Heuristique: alerter si écart-type > 40% de la valeur absolue moyenne
                    if (meanP2 != 0 && stdP2 > Math.abs(meanP2) * 0.40) {
                        logger.warn("[TUNING-2PH][PHASE2][DISTRIB] Dispersion anormale détectée (std > 40% | mean={} std={})", String.format("%.6f", meanP2), String.format("%.6f", stdP2));
                    }
                } else {
                    logger.warn("[TUNING-2PH][PHASE2][DISTRIB] Aucune valeur de businessScore exploitable pour la distribution {}", symbol);
                }
            }
            if (phase2 == null || phase2.bestConfig == null) {
                logger.warn("[TUNING-2PH][PHASE2] Aucun résultat – retour phase1 (hold-out si actif)");
                if (enableHoldOut) {
                    HoldOutEval ho1 = evaluateHoldOutFull(symbol, phase1.bestConfig, phase1.bestModel, phase1.bestScalers, series, holdOutStart);
                    if (ho1 != null) {
                        PhaseAggregate finalAg = new PhaseAggregate();
                        finalAg.bestConfig = phase1.bestConfig; finalAg.bestModel = ho1.model; finalAg.bestScalers = ho1.scalers;
                        finalAg.bestBusinessScore = ho1.businessScore; finalAg.bestMse = ho1.meanMse; finalAg.profitFactor = ho1.profitFactor;
                        finalAg.winRate = ho1.winRate; finalAg.maxDrawdown = ho1.maxDrawdown; finalAg.rmse = ho1.rmse; finalAg.sumProfit = ho1.sumProfit;
                        finalAg.totalTrades = ho1.totalTrades; finalAg.totalSeriesTested = ho1.totalSeriesTested;
                        finalAg.phaseGrid = 1;
                        finalAg.numberGrid = phase1.numberGrid;
                        finalAg.phase1TopN = 1;
                        finalAg.holdOut = true;
                        persistBest(symbol, finalAg, jdbcTemplate, 0);
                        progress.status = "termine"; progress.endTime = System.currentTimeMillis(); progress.lastUpdate = progress.endTime; try { writeProgressMetrics(progress);} catch(Exception ignored) {}
                        return phase1.bestConfig;
                    }
                }
                persistBest(symbol, phase1, jdbcTemplate, 0);
                progress.status = "termine"; progress.endTime = System.currentTimeMillis(); progress.lastUpdate = progress.endTime; try { writeProgressMetrics(progress);} catch(Exception ignored) {}
                return phase1.bestConfig;
            }
            // --- Nouvelle métrique d'amélioration: (improved - baseline) / max(eps, |baseline|)
            double baselineScore = phase1.bestBusinessScore; // redéfini localement ici pour clarté
            double improvedScore = phase2.bestBusinessScore;
            double baselineStd = phase1.bestBusinessScoreStd > 0 ? phase1.bestBusinessScoreStd : 0.0; // écart-type splits WF baseline
            double absoluteGain = improvedScore - baselineScore;
            double denom = Math.max(REL_GAIN_EPS, Math.abs(baselineScore));
            double relativeGain = absoluteGain / denom;
            // Nouvelle condition: amélioration doit dépasser baseline + 1 * std (robustesse statistique simple)
            boolean stdCondition = improvedScore > baselineScore + baselineStd;
            boolean acceptPhase2 = relativeGain >= minRelativeGain && absoluteGain >= minAbsoluteGain && stdCondition;
            if (logger.isInfoEnabled()) {
                /*logger.info("[TUNING-2PH][COMPARE] baseline={} (std={}) improved={} absGain={} relGain={} thrBaselinePlusStd={} acceptPhase2={} (seuils: rel>={} abs>= {})",
                        String.format("%.6f", baselineScore), String.format("%.6f", baselineStd), String.format("%.6f", improvedScore),
                        String.format("%.6f", absoluteGain), String.format("%.6f", relativeGain), String.format("%.6f", baselineScore + baselineStd), acceptPhase2,
                        String.format("%.3f", minRelativeGain), String.format("%.3f", minAbsoluteGain));*/
            }
            PhaseAggregate provisional = acceptPhase2 ? phase2 : phase1;
            boolean fromPhase2 = provisional == phase2;
            double ratioPersist = fromPhase2 ? relativeGain : 0; // on conserve le gain relatif accepté

            PhaseAggregate finalAggregate;
            if (enableHoldOut) {
                //logger.info("[TUNING-2PH][HOLDOUT] Validation hold-out des candidats");
                HoldOutEval ho1 = evaluateHoldOutFull(symbol, phase1.bestConfig,
                        provisional.bestModel, provisional.bestScalers,
                        series, holdOutStart);
                HoldOutEval hoProv = fromPhase2 ? evaluateHoldOutFull(symbol, provisional.bestConfig,
                        provisional.bestModel, provisional.bestScalers,
                        series, holdOutStart) : ho1;
                if (ho1 != null && hoProv != null) {
                    boolean acceptProv = fromPhase2 && hoProv.businessScore >= ho1.businessScore * 1.00; // au moins égal en hold-out
                    HoldOutEval chosen = acceptProv ? hoProv : ho1;
                    if (!acceptProv && fromPhase2) ratioPersist = 0; // gain non confirmé
                    finalAggregate = new PhaseAggregate();
                    finalAggregate.bestConfig = acceptProv ? provisional.bestConfig : phase1.bestConfig;
                    finalAggregate.bestModel = chosen.model;
                    finalAggregate.bestScalers = chosen.scalers;
                    finalAggregate.bestBusinessScore = chosen.businessScore;
                    finalAggregate.bestMse = chosen.meanMse;
                    finalAggregate.profitFactor = chosen.profitFactor;
                    finalAggregate.winRate = chosen.winRate;
                    finalAggregate.maxDrawdown = chosen.maxDrawdown;
                    finalAggregate.rmse = chosen.rmse;
                    finalAggregate.sumProfit = chosen.sumProfit;
                    finalAggregate.totalTrades = chosen.totalTrades;
                    finalAggregate.totalSeriesTested = chosen.totalSeriesTested;
                    finalAggregate.phaseGrid = fromPhase2 ? provisional.phaseGrid : phase1.phaseGrid;
                    finalAggregate.numberGrid = fromPhase2 ? provisional.numberGrid : phase1.numberGrid;
                    finalAggregate.phase1TopN = fromPhase2 ? provisional.phase1TopN : phase1.phase1TopN;
                    finalAggregate.holdOut = true;
                } else {
                    logger.warn("[TUNING-2PH][HOLDOUT] Échec évaluation hold-out – on persiste sélection provisoire");
                    finalAggregate = provisional;
                }
            } else {
                finalAggregate = provisional;
            }

            persistBest(symbol, finalAggregate, jdbcTemplate, ratioPersist);
            progress.status = "termine"; progress.endTime = System.currentTimeMillis(); progress.lastUpdate = progress.endTime; try { writeProgressMetrics(progress);} catch(Exception ignored) {}
            return finalAggregate.bestConfig;
        } catch (Exception e) {
            progress.status = "erreur"; progress.endTime = System.currentTimeMillis(); progress.lastUpdate = progress.endTime; try { writeProgressMetrics(progress);} catch(Exception ignored) {}
            logger.error("[TUNING-2PH] Erreur globale {} : {}", symbol, e.getMessage());
            return null;
        }
    }

    // ---- Structures internes pour la phase deux (restaurées) ----
    private static class PhaseAggregate {
        LstmConfig bestConfig; MultiLayerNetwork bestModel; LstmTradePredictor.ScalerSet bestScalers;
        double bestBusinessScore; double bestBusinessScoreStd; double bestMse; double profitFactor; double winRate; double maxDrawdown;
        double rmse; double sumProfit; int totalTrades; int totalSeriesTested;
        int phaseGrid;
        int numberGrid;
        int phase1TopN;
        String tuningResult;
        String phase1TopNLabel;
        boolean holdOut; java.util.List<TuningResult> allResults; }

    // --- Structure évaluation hold-out ---
    private static class HoldOutEval { LstmConfig config; MultiLayerNetwork model; LstmTradePredictor.ScalerSet scalers; double businessScore; double meanMse; double rmse; double profitFactor; double winRate; double maxDrawdown; double sumProfit; int totalTrades; int totalSeriesTested; }

    private HoldOutEval evaluateHoldOut(String symbol, LstmConfig config, BarSeries fullSeries, int holdOutStart) {
        try {
            waitForMemory();
            if (holdOutStart <= 0 || holdOutStart >= fullSeries.getBarCount()-config.getWindowSize()-10) return null;
            BarSeries trainSeries = fullSeries.getSubSeries(0, holdOutStart);
            lstmTradePredictor.setGlobalSeeds(config.getSeed());
            LstmTradePredictor.TrainResult tr = lstmTradePredictor.trainLstmScalarV2(trainSeries, config);
            MultiLayerNetwork model = tr.model; LstmTradePredictor.ScalerSet scalers = tr.scalers;
            LstmTradePredictor.WalkForwardResultV2 wf = lstmTradePredictor.walkForwardEvaluateOutOfSample(fullSeries, config, model, scalers, holdOutStart);
            double sumB=0,sumPF=0,sumWin=0,maxDrawdownPct=0,sumProfit=0; int splits=0,trades=0; for(var m: wf.splits){ sumB += Double.isFinite(m.businessScore)?m.businessScore:0; sumPF += Double.isFinite(m.profitFactor)?m.profitFactor:0; sumWin+=m.winRate; if(m.maxDrawdownPct>maxDrawdownPct) maxDrawdownPct=m.maxDrawdownPct; sumProfit+=m.totalProfit; trades+=m.numTrades; splits++; }
            if (splits==0) return null;
            HoldOutEval ho=new HoldOutEval(); ho.config=config; ho.model=model; ho.scalers=scalers; ho.businessScore=sumB/splits; ho.meanMse=wf.meanMse; ho.rmse=Double.isFinite(wf.meanMse)&&wf.meanMse>=0?Math.sqrt(wf.meanMse):Double.NaN; ho.profitFactor=sumPF/splits; ho.winRate=sumWin/splits; ho.maxDrawdown=maxDrawdownPct; ho.sumProfit=sumProfit; ho.totalTrades=trades; ho.totalSeriesTested=wf.totalTestedBars; 
            //logger.info("[TUNING-2PH][HOLDOUT] {} neurons={} lr={} dropout={} bs={} pf={} wr={} dd={}", symbol, config.getLstmNeurons(), config.getLearningRate(), config.getDropoutRate(), String.format("%.5f", ho.businessScore), String.format("%.4f", ho.profitFactor), String.format("%.4f", ho.winRate), String.format("%.4f", ho.maxDrawdown)); 
            return ho;
        } catch (Exception e) { logger.warn("[TUNING-2PH][HOLDOUT] Échec {} : {}", symbol, e.getMessage()); return null; }
        finally { try { org.nd4j.linalg.factory.Nd4j.getMemoryManager().invokeGc(); } catch (Exception ignored) {} try { org.nd4j.linalg.factory.Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread(); } catch (Exception ignored) {} System.gc(); }
    }
    private HoldOutEval evaluateHoldOutFull(String symbol, LstmConfig config, MultiLayerNetwork model, LstmTradePredictor.ScalerSet scalers, BarSeries fullSeries, int holdOutStart) {
        try {
            waitForMemory();
            if (holdOutStart <= 0 || holdOutStart >= fullSeries.getBarCount()-config.getWindowSize()-10) return null;
            lstmTradePredictor.setGlobalSeeds(config.getSeed());
            LstmTradePredictor.WalkForwardResultV2 wf = lstmTradePredictor.walkForwardEvaluateOutOfSample(fullSeries, config, model, scalers, holdOutStart);
            double sumB=0,sumPF=0,sumWin=0,maxDrawdownPct=0,sumProfit=0; int splits=0,trades=0; for(var m: wf.splits){ sumB += Double.isFinite(m.businessScore)?m.businessScore:0; sumPF += Double.isFinite(m.profitFactor)?m.profitFactor:0; sumWin+=m.winRate; if(m.maxDrawdownPct>maxDrawdownPct) maxDrawdownPct=m.maxDrawdownPct; sumProfit+=m.totalProfit; trades+=m.numTrades; splits++; }
            if (splits==0) return null;
            HoldOutEval ho=new HoldOutEval(); ho.config=config; ho.model=model; ho.scalers=scalers; ho.businessScore=sumB/splits; ho.meanMse=wf.meanMse; ho.rmse=Double.isFinite(wf.meanMse)&&wf.meanMse>=0?Math.sqrt(wf.meanMse):Double.NaN; ho.profitFactor=sumPF/splits; ho.winRate=sumWin/splits; ho.maxDrawdown=maxDrawdownPct; ho.sumProfit=sumProfit; ho.totalTrades=trades; ho.totalSeriesTested=wf.totalTestedBars;
            //logger.info("[TUNING-2PH][HOLDOUT] {} neurons={} lr={} dropout={} bs={} pf={} wr={} dd={}", symbol, config.getLstmNeurons(), config.getLearningRate(), config.getDropoutRate(), String.format("%.5f", ho.businessScore), String.format("%.4f", ho.profitFactor), String.format("%.4f", ho.winRate), String.format("%.4f", ho.maxDrawdown));
            return ho;
        } catch (Exception e) { logger.warn("[TUNING-2PH][HOLDOUT] Échec {} : {}", symbol, e.getMessage()); return null; }
        finally { try { org.nd4j.linalg.factory.Nd4j.getMemoryManager().invokeGc(); } catch (Exception ignored) {} try { org.nd4j.linalg.factory.Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread(); } catch (Exception ignored) {} System.gc(); }
    }
    private PhaseAggregate runPhaseNoPersist(String symbol,
                                             java.util.List<LstmConfig> grid,
                                             BarSeries series,
                                             int phase,
                                             String phaseTag,
                                             TuningProgress progress) {
        waitForMemory();
        long start = System.currentTimeMillis();
        int numThreads = 2; // Désormais séquentiel
        if (numThreads < 1) numThreads = 1;
        if (progress != null && progress.threadsUsed == 0) progress.threadsUsed = numThreads;
        // Suppression du pool de threads
        // var executor = java.util.concurrent.Executors.newFixedThreadPool(numThreads);
        // var futures = new java.util.ArrayList<java.util.concurrent.Future<TuningResult>>();
        var results = new java.util.ArrayList<TuningResult>(); // plus besoin de synchronizedList
        for (int i=0;i<grid.size();i++) {
            final int idx=i; LstmConfig cfg=grid.get(i); cfg.setUseScalarV2(true); cfg.setUseWalkForwardV2(true); // fix parenthèse
            // Auto-scale batch GPU phase1/2
            if (cudaBackend && gpuAutoBatchScale) {
                int originalBatch = cfg.getBatchSize()>0?cfg.getBatchSize():32;
                if (originalBatch < gpuTargetBatchSize) {
                    int newBatch = originalBatch;
                    while (newBatch < gpuTargetBatchSize) { newBatch *= 2; if (newBatch >= gpuTargetBatchSize) break; }
                    if (newBatch > gpuTargetBatchSize) newBatch = gpuTargetBatchSize;
                    if (newBatch != originalBatch) {
                        if (gpuScaleLearningRateOnBatch && cfg.getLearningRate() > 0) {
                            double scaledLr = cfg.getLearningRate() * ((double) originalBatch / (double) newBatch);
                            cfg.setLearningRate(scaledLr);
                        }
                        cfg.setBatchSize(newBatch);
                        //logger.debug("[GPU][BATCHSCALE][{}] phase={} idx={} batch {} -> {} lr={}", symbol, phase, idx+1, originalBatch, newBatch, cfg.getLearningRate());
                    }
                }
            }
            // --- Phase 2: diversification supplémentaire pour réduire corrélation ---
            if (phase == 2 && !cfg.isBaselineReplica()) {
                cfg.setSeed(cfg.getSeed() + 777 + idx); // seed décalé (sauf baseline)
                cfg.setWalkForwardSplits(Math.min(6, cfg.getWalkForwardSplits() + 1)); // plus de splits (sauf baseline)
            }
            // Exécution séquentielle de la tâche (remplace submit)
            MultiLayerNetwork model=null; boolean permit=false; long stagger=0;
            try {
                if (cudaBackend){ gpuController.acquirePermit(); permit=true; int active=gpuController.getActiveTrainings(); if(gpuEnableStagger && active>1){ stagger=1000+(long)(Math.random()*1500); Thread.sleep(stagger);} gpuController.markTrainingStarted();
                    //logger.info("[GPU][MONITOR] Utilisation VRAM: {}% (activeTrainings={})", String.format("%.2f", gpuController.getLastVramUsagePct()), active);
                }
                lstmTradePredictor.setGlobalSeeds(cfg.getSeed());
                int totalBars=series.getBarCount();
                int testSplitRatio = (phase == 2 ? 25 : 20);
                int trainEnd= totalBars*(100-testSplitRatio)/100;
                if (phase == 2) {
                    int jitterRange = Math.max(5, (int)(totalBars * 0.01));
                    int jitterSeed = (int)((cfg.getSeed() ^ (idx * 0x9E3779B97F4A7C15L)) & 0x7fffffff);
                    java.util.Random jitterRand = new java.util.Random(jitterSeed);
                    int jitter = jitterRand.nextInt(jitterRange * 2 + 1) - jitterRange;
                    trainEnd = trainEnd + jitter;
                    int minTrain = cfg.getWindowSize() + 60;
                    int maxTrain = totalBars - (cfg.getWindowSize() + 60);
                    if (trainEnd < minTrain) trainEnd = minTrain;
                    if (trainEnd > maxTrain) trainEnd = maxTrain;
                    if (logger.isDebugEnabled()) {
                        //logger.debug("[TUNING-2PH][{}] phase=2 configIdx={} ratioTest={} jitter={} trainEnd={}/{}", symbol, idx+1, testSplitRatio, jitter, trainEnd, totalBars);
                    }
                }
                if(trainEnd < cfg.getWindowSize()+50) throw new IllegalStateException("Données insuffisantes");
                BarSeries trainSeries=series.getSubSeries(0, trainEnd);
                var tr = lstmTradePredictor.trainLstmScalarV2(trainSeries, cfg);
                model=tr.model; var scalers=tr.scalers;
                var wf = lstmTradePredictor.walkForwardEvaluateOutOfSample(series, cfg, model, scalers, trainEnd);
                double sumPF=0,sumWin=0,maxDrawdownPct=0,sumBusiness=0,sumProfit=0,sumBusinessSq=0; // ajout sumBusinessSq
                int splits=0,trades=0;
                for(var m: wf.splits){
                    double bsSplit = Double.isFinite(m.businessScore)?m.businessScore:0;
                    sumPF+=Double.isFinite(m.profitFactor)?m.profitFactor:0;
                    sumWin+=m.winRate;
                    if(m.maxDrawdownPct>maxDrawdownPct) maxDrawdownPct=m.maxDrawdownPct;
                    sumBusiness+=bsSplit;
                    sumBusinessSq+=bsSplit*bsSplit; // accumulation pour variance
                    sumProfit+=m.totalProfit;
                    trades+=m.numTrades; splits++; }
                if (splits==0) throw new AucunSplitValideException("Aucun split valide");
                double meanMse=wf.meanMse; double meanBusiness=sumBusiness/splits; double variance = (sumBusinessSq / splits) - (meanBusiness*meanBusiness); if(variance<0) variance=0; double stdBusiness = Math.sqrt(variance);
                double rmse=(Double.isFinite(meanMse)&&meanMse>=0)?Math.sqrt(meanMse):Double.NaN;
                hyperparamsRepository.saveTuningMetrics(symbol,cfg,meanMse,rmse,sumProfit,sumPF/splits,sumWin/splits,maxDrawdownPct,trades,meanBusiness,
                        wf.splits.stream().mapToDouble(m->m.sortino).average().orElse(0.0),
                        wf.splits.stream().mapToDouble(m->m.calmar).average().orElse(0.0),
                        wf.splits.stream().mapToDouble(m->m.turnover).average().orElse(0.0),
                        wf.splits.stream().mapToDouble(m->m.avgBarsInPosition).average().orElse(0.0),
                        phase, i + 1, -1, false);
                TuningResult trRes=new TuningResult(cfg, model, scalers, meanMse, sumPF/splits, sumWin/splits, maxDrawdownPct, meanBusiness, stdBusiness, rmse, sumProfit, trades, wf.totalTestedBars, i + 1);
                results.add(trRes);
                if(progress!=null){ progress.testedConfigs.incrementAndGet(); progress.lastUpdate=System.currentTimeMillis(); }
                //logger.info("[TUNING-2PH][{}] {} config {}/{} bs={} adj={} dd={} trades={} phase={}", phaseTag, symbol, idx+1, grid.size(), String.format("%.5f", meanBusiness), String.format("%.5f", adjScore(trRes)), String.format("%.4f", maxDrawdownPct), trades, phase);
                trRes.numberGridTop = grid.get(idx).getIndexTop();
                trRes.numberGrid = i + 1;
            } catch(AucunSplitValideException ex){
                if(progress!=null){ progress.testedConfigs.incrementAndGet(); progress.lastUpdate=System.currentTimeMillis(); }
                logger.warn("[TUNING-2PH][{}][{}] AucunSplitValideException {}/{} : {}", phaseTag, symbol, idx+1, grid.size(), ex.getMessage());
            } catch(Exception ex){
                if(progress!=null){ progress.testedConfigs.incrementAndGet(); progress.lastUpdate=System.currentTimeMillis(); }
                logger.error("[TUNING-2PH]Exception[{}][{}] Erreur config {}/{} : {}", phaseTag, symbol, idx+1, grid.size(), ex.getMessage());
            }
            finally { model=null; try{ org.nd4j.linalg.factory.Nd4j.getMemoryManager().invokeGc(); org.nd4j.linalg.factory.Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread(); }catch(Exception ignore){} System.gc(); if(cudaBackend){ gpuController.markTrainingFinished(); if(permit) gpuController.releasePermit(); } }
        }
        // Suppression de la récupération des futures
        // executor.shutdown(); for(var f: futures){ try{ f.get(); }catch(Exception ignored){} }
        if(results.isEmpty()) return null;
        TuningResult best=null; double bestAdj=Double.NEGATIVE_INFINITY; double bestRaw=Double.NEGATIVE_INFINITY; double bestStd=0;
        for(var r: results){ if(r==null) continue; double a=adjScore(r); if(a>bestAdj){ bestAdj=a; bestRaw=r.businessScore; bestStd=r.businessScoreStd; best=r; } }
        PhaseAggregate ag=new PhaseAggregate();
        ag.bestConfig=best!=null?best.config:null; ag.bestModel=best!=null?best.model:null; ag.bestScalers=best!=null?best.scalers:null;
        ag.bestBusinessScore=bestRaw; ag.bestBusinessScoreStd=bestStd; ag.bestMse=best!=null?best.score:Double.NaN; ag.profitFactor=best!=null?best.profitFactor:0;
        ag.winRate=best!=null?best.winRate:0; ag.maxDrawdown=best!=null?best.maxDrawdown:0; ag.rmse=best!=null?best.rmse:0; ag.sumProfit=best!=null?best.sumProfit:0;
        ag.totalTrades=best!=null?best.totalTrades:0; ag.totalSeriesTested=best!=null?best.totalSeriesTested:0;
        ag.phaseGrid = phase;
        ag.numberGrid = best!=null?best.numberGrid:-1;
        ag.phase1TopN = best!=null?best.numberGridTop:-1;
        ag.holdOut = true;
        ag.allResults=results;
        //logger.info("[TUNING-2PH][{}] Fin phase {} | bestRawBS={} bestAdjBS={} std={} trades={} ", symbol, phaseTag, String.format("%.6f", bestRaw), String.format("%.6f", bestAdj), String.format("%.6f", bestStd), best!=null?best.totalTrades:0); 
        return ag;
    }
    private void persistBest(String symbol, PhaseAggregate pa, JdbcTemplate jdbcTemplate, double ratio){
        if(pa.bestConfig==null||pa.bestModel==null||pa.bestScalers==null){ logger.warn("[TUNING-2PH][PERSIST] Objets null – skip"); return; }
        try { hyperparamsRepository.saveHyperparams(symbol, pa.bestConfig, pa.phaseGrid); } catch(Exception e){ logger.error("[TUNING-2PH][PERSIST] saveHyperparams échec: {}", e.getMessage()); }
        try { synchronized (modelSaveLock){ lstmTradePredictor.saveModelToDb(symbol, jdbcTemplate, pa.bestModel, pa.bestConfig, pa.bestScalers, pa.bestMse,
                pa.profitFactor, pa.winRate, pa.maxDrawdown, pa.rmse, pa.sumProfit, pa.totalTrades, pa.bestBusinessScore, pa.totalSeriesTested,
            pa.phaseGrid,
            pa.numberGrid,
            pa.phase1TopN,
            pa.phase1TopNLabel,
            pa.holdOut, pa.tuningResult, ratio); } }
        catch(Exception e){ logger.error("[TUNING-2PH][PERSIST] saveModelToDb échec: {}", e.getMessage()); }
    }

    // --- Utilitaires micro-grille ---
    private static LstmConfig cloneConfig(LstmConfig src){
        LstmConfig c = new LstmConfig();
        c.setWindowSize(src.getWindowSize()); c.setLstmNeurons(src.getLstmNeurons()); c.setDropoutRate(src.getDropoutRate()); c.setLearningRate(src.getLearningRate());
        c.setNumEpochs(src.getNumEpochs()); c.setPatience(src.getPatience()); c.setMinDelta(src.getMinDelta()); c.setKFolds(src.getKFolds());
        c.setOptimizer(src.getOptimizer()); c.setL1(src.getL1()); c.setL2(src.getL2()); c.setNormalizationScope(src.getNormalizationScope());
        c.setNormalizationMethod(src.getNormalizationMethod()); c.setSwingTradeType(src.getSwingTradeType()); c.setUseScalarV2(src.isUseScalarV2()); c.setUseWalkForwardV2(src.isUseWalkForwardV2());
        c.setNumLstmLayers(src.getNumLstmLayers()); c.setBidirectional(src.isBidirectional()); c.setAttention(src.isAttention()); c.setHorizonBars(src.getHorizonBars());
        c.setBatchSize(src.getBatchSize()); c.setWalkForwardSplits(src.getWalkForwardSplits()); c.setEmbargoBars(src.getEmbargoBars()); c.setCapital(src.getCapital());
        c.setRiskPct(src.getRiskPct()); c.setSizingK(src.getSizingK()); c.setFeePct(src.getFeePct()); c.setSlippagePct(src.getSlippagePct());
        c.setBusinessProfitFactorCap(src.getBusinessProfitFactorCap()); c.setBusinessDrawdownGamma(src.getBusinessDrawdownGamma()); c.setSeed(src.getSeed());
        c.setCvMode(src.getCvMode());
        return c;
    }
    private static String keyOf(LstmConfig c){ return c.getLstmNeurons()+"|"+String.format(java.util.Locale.US,"%.6f",c.getLearningRate())+"|"+String.format(java.util.Locale.US,"%.4f",c.getDropoutRate()); }
    // Nouveau score ajusté: favorise modèles avec plus de trades et évite les modèles trop plats
    private static double adjScore(TuningResult r){
        if(r==null) return Double.NEGATIVE_INFINITY;
        double base = r.businessScore;
        if(r.totalTrades < 3) base *= 0.40; else if(r.totalTrades < 8) base *= 0.70; else if(r.totalTrades < 12) base *= 0.85;
        double tradeRef = 30.0; double boost = Math.min(r.totalTrades / tradeRef, 1.6); // boost max +60%
        base *= (0.9 + 0.12*boost);
        if(r.businessScoreStd < 0.0005) base *= 0.70; else if(r.businessScoreStd < 0.001) base *= 0.88;
        // Pénalités winRate extrêmes (soupçon sur-fit ou manque de robustesse)
        if(r.winRate > 0.95) base *= 0.55; else if(r.winRate > 0.90) base *= 0.75; // trop parfait => suspect
        if(r.winRate < 0.25) base *= 0.60; else if(r.winRate < 0.35) base *= 0.80;  // trop faible => faible généralisabilité
        return base;
    }
    private static double adjScoreRendement(TuningResult r){
        if(r == null) return Double.NEGATIVE_INFINITY;
        double rendement = r.sumProfit;
        double base = r.businessScore * 0.6 + rendement * 0.4; // pondération : 40% rendement, 60% businessScore

        // Bonus si rendement élevé
        if(rendement > 0) base *= (1.0 + Math.min(rendement / 100.0, 0.5)); // max +50% si rendement > 50

        // Pénalités sur le nombre de trades
        if(r.totalTrades < 3) base *= 0.40;
        else if(r.totalTrades < 8) base *= 0.70;
        else if(r.totalTrades < 12) base *= 0.85;

        // Boost sur le nombre de trades
        double tradeRef = 30.0;
        double boost = Math.min(r.totalTrades / tradeRef, 1.6);
        base *= (0.9 + 0.12 * boost);

        // Robustesse (écart-type)
        if(r.businessScoreStd < 0.0005) base *= 0.70;
        else if(r.businessScoreStd < 0.001) base *= 0.88;

        // Pénalités winRate extrêmes
        if(r.winRate > 0.95) base *= 0.55;
        else if(r.winRate > 0.90) base *= 0.75;
        if(r.winRate < 0.25) base *= 0.60;
        else if(r.winRate < 0.35) base *= 0.80;

        return base;
    }

    // Java
    private TuningResult cloneTuningResult(TuningResult src) {

        TuningResult clone = new TuningResult(
                src.config,
                src.model,
                src.scalers,
                src.score,
                src.profitFactor,
                src.winRate,
                src.maxDrawdown,
                src.businessScore,
                src.businessScoreStd,
                src.rmse,
                src.sumProfit,
                src.totalTrades,
                src.totalSeriesTested,
                src.numberGrid
        );
        clone.top = src.top;
        clone.topLabel = src.topLabel;
        clone.numberGridTop = src.numberGridTop;
        return clone;
    }
}
