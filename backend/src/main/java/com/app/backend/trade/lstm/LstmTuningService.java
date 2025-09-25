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

import static com.app.backend.trade.strategy.StrategieBackTest.FEE_PCT;
import static com.app.backend.trade.strategy.StrategieBackTest.SLIP_PAGE_PCT;

/**
 * Service de tuning des hyperparamètres LSTM pour le trading.
 */
@Service
public class LstmTuningService {
    private static final Logger logger = LoggerFactory.getLogger(LstmTuningService.class);
    public final LstmTradePredictor lstmTradePredictor;
    public final LstmHyperparamsRepository hyperparamsRepository;

    // Structure de suivi d'avancement
    public static class TuningProgress {
        public String symbol;
        public int totalConfigs;
        public AtomicInteger testedConfigs = new AtomicInteger(0);
        public String status = "en_cours";
        public long startTime;
        public long endTime;
        public long lastUpdate;
    }

    private final ConcurrentHashMap<String, TuningProgress> tuningProgressMap = new ConcurrentHashMap<>();

    @PostConstruct
    public void initNd4jCuda() {
        try {
            // Activation cuDNN si disponible
            System.setProperty("org.deeplearning4j.cudnn.enabled", "true");
            org.nd4j.linalg.factory.Nd4jBackend backend = org.nd4j.linalg.factory.Nd4j.getBackend();
            logger.info("ND4J backend utilisé : {}", backend.getClass().getSimpleName());
            if (!backend.getClass().getSimpleName().toLowerCase().contains("cuda")) {
                logger.warn("Le backend ND4J n'est pas CUDA ! Le GPU ne sera pas utilisé.");
                logger.warn("Pour forcer CUDA, lancez la JVM avec : -Dorg.nd4j.linalg.defaultbackend=org.nd4j.linalg.jcublas.JCublasBackend");
            } else {
                logger.info("Backend CUDA détecté. Pour de meilleures performances, vérifiez que cuDNN est activé et que la version CUDA/cuDNN est compatible avec ND4J/DL4J.");
            }
        } catch (Exception e) {
            logger.error("Erreur lors de la détection du backend ND4J : {}", e.getMessage());
        }
    }

    // Structure de reporting centralisé des exceptions tuning
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

    public ConcurrentHashMap<String, TuningProgress> getTuningProgressMap() {
        return tuningProgressMap;
    }

    public java.util.List<TuningExceptionReportEntry> getTuningExceptionReport() {
        return new java.util.ArrayList<>(tuningExceptionReport);
    }

    /**
     * Tune automatiquement les hyperparamètres pour un symbole donné.
     * @param symbol le symbole à tuner
     * @param grid la liste des configurations à tester
     * @param series les données historiques du symbole
     * @param jdbcTemplate accès base
     * @return la meilleure configuration trouvée
     */
    public LstmConfig tuneSymbolMultiThread(String symbol, List<LstmConfig> grid, BarSeries series, JdbcTemplate jdbcTemplate) {
        if(isSymbolAlreydyTuned(symbol, jdbcTemplate)){
            return null;
        }
        waitForMemory();
        long startSymbol = System.currentTimeMillis();
        // Suivi d'avancement
        TuningProgress progress = new TuningProgress();
        progress.symbol = symbol;
        progress.totalConfigs = grid.size();
        progress.testedConfigs.set(0);
        progress.status = "en_cours";
        progress.startTime = startSymbol;
        progress.lastUpdate = startSymbol;
        tuningProgressMap.put(symbol, progress);
        logger.info("[TUNING] Début du tuning V2 pour le symbole {} ({} configs)", symbol, grid.size());
        double bestScore = Double.POSITIVE_INFINITY; // MSE moyen walk-forward
        LstmConfig existing = hyperparamsRepository.loadHyperparams(symbol);
        if(existing != null){
            logger.info("Hyperparamètres existants trouvés pour {}. Ignorer le tuning.", symbol);
            return existing;
        }
        LstmConfig bestConfig = null;
        MultiLayerNetwork bestModel = null;
        LstmTradePredictor.ScalerSet bestScalers = null;
        int numThreads = Math.min(Math.min(grid.size(), Runtime.getRuntime().availableProcessors()), MAX_THREADS);
        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(numThreads);
        java.util.List<java.util.concurrent.Future<TuningResult>> futures = new java.util.ArrayList<>();
        for (int i = 0; i < grid.size(); i++) {
            waitForMemory();
            final int configIndex = i + 1;
            LstmConfig config = grid.get(i);
            // Forcer pipeline V2
            config.setUseScalarV2(true);
            config.setUseWalkForwardV2(true);
            futures.add(executor.submit(() -> {
                MultiLayerNetwork model = null;
                try {
                    long startConfig = System.currentTimeMillis();
                    lstmTradePredictor.setGlobalSeeds(config.getSeed());
                    logger.info("[TUNING][V2] [{}] Début config {}/{}", symbol, configIndex, grid.size());
                    // Entraînement sur toute la série pour la prédiction finale (après WF pour sélection)
                    LstmTradePredictor.TrainResult trFull = lstmTradePredictor.trainLstmScalarV2(series, config, null);
                    model = trFull.model;
                    LstmTradePredictor.ScalerSet scalers = trFull.scalers;
                    // Walk-forward evaluation
                    LstmTradePredictor.WalkForwardResultV2 wf = lstmTradePredictor.walkForwardEvaluate(series, config);
                    double meanMse = wf.meanMse;
                    // Agrégation métriques trading
                    double sumPF=0, sumWin=0, sumExp=0, maxDDPct=0, sumBusiness=0, sumProfit=0; int splits=0; int totalTrades=0;
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
                        // Pas de splits valides; on marque échec
                        throw new IllegalStateException("Aucun split valide walk-forward");
                    }
                    double meanPF = sumPF / splits;
                    double meanWinRate = sumWin / splits;
                    double meanExpectancy = sumExp / splits;
                    double meanBusinessScore = sumBusiness / splits;
                    // Direction instantanée sur la série complète (optionnel)
                    double predicted = lstmTradePredictor.predictNextCloseScalarV2(series, config, model, scalers);
                    double lastClose = series.getLastBar().getClosePrice().doubleValue();
                    double th = lstmTradePredictor.computeSwingTradeThreshold(series, config);
                    String direction = (predicted - lastClose) > th ? "up" : ((predicted - lastClose) < -th ? "down" : "stable");
                    // Sauvegarde métriques tuning (mse, rmse, profitTotal= somme profits splits, profitFactor moyen, winRate moyen, drawdown max pct, numTrades total)
                    double rmse = Double.isFinite(meanMse) && meanMse>=0? Math.sqrt(meanMse): Double.NaN;
                    hyperparamsRepository.saveTuningMetrics(symbol, config, meanMse, rmse, direction,
                            sumProfit, meanPF, meanWinRate, maxDDPct, totalTrades, meanBusinessScore,
                            // Aggregation des nouvelles métriques
                            wf.splits.stream().mapToDouble(m->m.sortino).average().orElse(0.0),
                            wf.splits.stream().mapToDouble(m->m.calmar).average().orElse(0.0),
                            wf.splits.stream().mapToDouble(m->m.turnover).average().orElse(0.0),
                            wf.splits.stream().mapToDouble(m->m.avgBarsInPosition).average().orElse(0.0));
                    long endConfig = System.currentTimeMillis();
                    logger.info("[TUNING][V2] [{}] Fin config {}/{} | meanMSE={}, PF={}, winRate={}, DD%={}, expectancy={}, businessScore={}, trades={} durée={} ms", symbol, configIndex, grid.size(), meanMse, meanPF, meanWinRate, maxDDPct, meanExpectancy, meanBusinessScore, totalTrades, (endConfig-startConfig));
                    TuningProgress p = tuningProgressMap.get(symbol);
                    if (p != null) { p.testedConfigs.incrementAndGet(); p.lastUpdate = System.currentTimeMillis(); }
                    return new TuningResult(config, model, scalers, meanMse, meanPF, meanWinRate, maxDDPct, meanBusinessScore);
                } catch (Exception e){
                    logger.error("[TUNING][V2] Erreur config {} : {}", configIndex, e.getMessage());
                    String stack = org.apache.commons.lang3.exception.ExceptionUtils.getStackTrace(e);
                    tuningExceptionReport.add(new TuningExceptionReportEntry(symbol, config, e.getMessage(), stack, System.currentTimeMillis()));
                    return null;
                } finally {
                    model = null;
                    org.nd4j.linalg.factory.Nd4j.getMemoryManager().invokeGc();
                    org.nd4j.linalg.factory.Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
                    System.gc();
                }
            }));
        }
        executor.shutdown();
        int failedConfigs = 0;
        double bestBusinessScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < futures.size(); i++) {
            try {
                TuningResult result = futures.get(i).get();
                logger.info("[TUNING][V2] [{}] Progression : {}/{} configs terminées", symbol, i+1, grid.size());
                if (result == null || Double.isNaN(result.businessScore) || Double.isInfinite(result.businessScore)) {
                    failedConfigs++;
                    continue;
                }
                if (result.businessScore > bestBusinessScore) {
                    bestBusinessScore = result.businessScore;
                    bestConfig = result.config;
                    bestModel = result.model;
                    bestScalers = result.scalers;
                    bestScore = result.score; // meanMSE
                }
            } catch (Exception e) {
                failedConfigs++;
                progress.status = "erreur";
                progress.lastUpdate = System.currentTimeMillis();
                logger.error("Erreur lors de la récupération du résultat de tuning : {}", e.getMessage());
                String stack = e.getCause() != null ? org.apache.commons.lang3.exception.ExceptionUtils.getStackTrace(e.getCause()) : org.apache.commons.lang3.exception.ExceptionUtils.getStackTrace(e);
                tuningExceptionReport.add(new TuningExceptionReportEntry(symbol, null, e.getMessage(), stack, System.currentTimeMillis()));
            }
        }
        long endSymbol = System.currentTimeMillis();
        if (failedConfigs == grid.size()) {
            progress.status = "failed";
            progress.endTime = endSymbol;
            progress.lastUpdate = endSymbol;
            logger.error("[TUNING][V2][EARLY STOP] Toutes les configs ont échoué pour le symbole {}.", symbol);
            return null;
        }
        progress.status = "termine";
        progress.endTime = endSymbol;
        progress.lastUpdate = endSymbol;
        if (bestConfig != null && bestModel != null && bestScalers != null) {
            hyperparamsRepository.saveHyperparams(symbol, bestConfig);
            try {
                lstmTradePredictor.saveModelToDb(symbol, bestModel, jdbcTemplate, bestConfig, bestScalers);
            } catch (Exception e) {
                logger.error("Erreur lors de la sauvegarde du meilleur modèle : {}", e.getMessage());
            }
            logger.info("[TUNING][V2] Fin tuning {} | Best businessScore={} | windowSize={}, neurons={}, dropout={}, lr={}, l1={}, l2={}, meanMSE={}, durée={} ms", symbol, bestBusinessScore, bestConfig.getWindowSize(), bestConfig.getLstmNeurons(), bestConfig.getDropoutRate(), bestConfig.getLearningRate(), bestConfig.getL1(), bestConfig.getL2(), bestScore, (endSymbol - startSymbol));
        } else {
            logger.warn("[TUNING][V2] Aucun modèle/scaler valide trouvé pour {}", symbol);
        }
        try {
            org.nd4j.linalg.factory.Nd4j.getMemoryManager().invokeGc();
            org.nd4j.linalg.factory.Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
            System.gc();
        } catch (Exception e) { logger.warn("Nettoyage ND4J échec : {}", e.getMessage()); }
        return bestConfig;
    }

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
     * Génère automatiquement une grille de configurations adaptée au swing trade.
     * @return liste de LstmConfig à tester
     */
    public List<LstmConfig> generateSwingTradeGrid() {
        List<LstmConfig> grid = new java.util.ArrayList<>();
        int[] windowSizes = {20, 30, 40}; // Fenêtres typiques swing
        int[] lstmNeurons = {64, 128, 256}; // Taille raisonnable pour swing
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
    public List<LstmConfig> generateSwingTradeGrid(List<String> features, int[] horizonBars, int[] numLstmLayers, int[] batchSizes, boolean[] bidirectionals, boolean[] attentions) {
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
                                                            config.setFeatures(features);
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
    public List<LstmConfig> generateRandomSwingTradeGrid(int n, List<String> features, int[] horizonBars, int[] numLstmLayers, int[] batchSizes, boolean[] bidirectionals, boolean[] attentions) {
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
            config.setFeatures(features);
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
        double pfAdj = Math.min(profitFactor, config.getBusinessProfitFactorCap());
        double expPos = Math.max(expectancy, 0.0);
        double denom = 1.0 + Math.pow(Math.max(maxDrawdownPct, 0.0), config.getBusinessDrawdownGamma());
        return (expPos * pfAdj * winRate) / (denom + 1e-9);
    }

    private static class TuningResult {
        LstmConfig config; MultiLayerNetwork model; LstmTradePredictor.ScalerSet scalers; double score; double profitFactor; double winRate; double maxDrawdown; double businessScore;
        TuningResult(LstmConfig c, MultiLayerNetwork m, LstmTradePredictor.ScalerSet s, double score, double pf, double wr, double dd, double bs){ this.config=c; this.model=m; this.scalers=s; this.score=score; this.profitFactor=pf; this.winRate=wr; this.maxDrawdown=dd; this.businessScore=bs; }
    }

    // Limite de threads pour éviter OOM (configurable)
    private static final int MAX_THREADS = 2; // Peut être ajusté selon la machine
    // Seuil mémoire (80% de la mémoire max JVM)
    private static final double MEMORY_USAGE_THRESHOLD = 0.8;

    /**
     * Vérifie si l'utilisation mémoire approche du seuil critique.
     * Retourne true si la mémoire est trop utilisée.
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
     * Attend que la mémoire soit sous le seuil critique, sinon sleep et log.
     */
    private static void waitForMemory() {
        while (isMemoryUsageHigh()) {
            logger.warn("[TUNING] Utilisation mémoire élevée (> {}%), attente avant de lancer une nouvelle tâche...", (int)(MEMORY_USAGE_THRESHOLD*100));
            try {
                Thread.sleep(5000); // Attendre 5s avant de réessayer
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }
}
