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
        logger.info("[TUNING] Début du tuning pour le symbole {} ({} configs)", symbol, grid.size());
        double bestScore = Double.MAX_VALUE;
        LstmConfig conf = hyperparamsRepository.loadHyperparams(symbol);
        if(conf != null){
            logger.info("Hyperparamètres existants trouvés pour {}. Ignorer le tuning.", symbol);
            return conf;
        }
        LstmConfig bestConfig = null;
        MultiLayerNetwork bestModel = null;
        LstmTradePredictor.ScalerSet bestScalers = null;
        int numThreads = Math.min(grid.size(), Runtime.getRuntime().availableProcessors());
        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(numThreads);
        List<java.util.concurrent.Future<TuningResult>> futures = new java.util.ArrayList<>();
        for (int i = 0; i < grid.size(); i++) {
            final int configIndex = i + 1;
            LstmConfig config = grid.get(i);
            futures.add(executor.submit(() -> {
                long startConfig = System.currentTimeMillis();
                logger.info("[TUNING] [{}] | Début config {}/{}", symbol, Thread.currentThread().getName(), configIndex, grid.size());
                int numFeatures = config.getFeatures() != null ? config.getFeatures().size() : 1;
                MultiLayerNetwork model = lstmTradePredictor.initModel(
                    numFeatures,
                    1,
                    config.getLstmNeurons(),
                    config.getDropoutRate(),
                    config.getLearningRate(),
                    config.getOptimizer(),
                    config.getL1(),
                    config.getL2(),
                    config, true
                );
                LstmTradePredictor.TrainResult trainResult = lstmTradePredictor.trainLstmWithScalers(series, config, model);
                model = trainResult.model;
                LstmTradePredictor.ScalerSet scalers = trainResult.scalers;
                double score = Double.POSITIVE_INFINITY;
                String cvMode = config.getCvMode() != null ? config.getCvMode() : "split";
                if ("timeseries".equalsIgnoreCase(cvMode)) {
                    score = lstmTradePredictor.crossValidateLstmTimeSeriesSplit(series, config);
                } else if ("kfold".equalsIgnoreCase(cvMode)) {
                    score = lstmTradePredictor.crossValidateLstm(series, config);
                } else { // split simple
                    score = lstmTradePredictor.splitScoreLstm(series, config);
                }
                double rmse = Math.sqrt(score);
                double predicted = lstmTradePredictor.predictNextClose(symbol, series, config, model);
                double[] closes = lstmTradePredictor.extractCloseValues(series);
                double lastClose = closes[closes.length - 1];
                double delta = predicted - lastClose;
                double th = lstmTradePredictor.computeSwingTradeThreshold(series, config);
                String direction;
                if (delta > th) {
                    direction = "up";
                } else if (delta < -th) {
                    direction = "down";
                } else {
                    direction = "stable";
                }
                // Calcul des métriques trading sur le jeu de test
                double[] tradingMetrics = lstmTradePredictor.calculateTradingMetricsAdvanced(series, config, model,  FEE_PCT, SLIP_PAGE_PCT);
                double profitTotal = tradingMetrics[0];
                int numTrades = (int) tradingMetrics[1];
                double profitFactor = tradingMetrics[2];
                double winRate = tradingMetrics[3];
                double maxDrawdown = tradingMetrics[4];
                // Sauvegarde des métriques
                hyperparamsRepository.saveTuningMetrics(symbol, config, score, rmse, direction,
                    profitTotal, profitFactor, winRate, maxDrawdown, numTrades);
                long endConfig = System.currentTimeMillis();
                logger.info("[TUNING] [{}] Fin config {}/{} | MSE={}, RMSE={}, direction={}, durée={} ms", symbol, configIndex, grid.size(), score, rmse, direction, (endConfig - startConfig));
                // Mise à jour du suivi d'avancement
                TuningProgress p = tuningProgressMap.get(symbol);
                if (p != null) {
                    p.testedConfigs.incrementAndGet();
                    p.lastUpdate = System.currentTimeMillis();
                }
                // Calcul du score métier
                double businessScore = computeBusinessScore(profitFactor, winRate, maxDrawdown);
                return new TuningResult(config, model, score, scalers, profitFactor, winRate, maxDrawdown, businessScore);
            }));
        }
        executor.shutdown();
        int failedConfigs = 0;
        double bestBusinessScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < futures.size(); i++) {
            try {
                TuningResult result = futures.get(i).get();
                logger.info("[TUNING] [{}] Progression : {}/{} configs terminées", symbol, i+1, grid.size());
                if (result == null || Double.isNaN(result.score) || Double.isInfinite(result.score)) {
                    failedConfigs++;
                    continue;
                }
                if (result.businessScore > bestBusinessScore) {
                    bestBusinessScore = result.businessScore;
                    bestConfig = result.config;
                    bestModel = result.model;
                    bestScalers = result.scalers;
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
            logger.error("[TUNING][EARLY STOP GLOBAL] Toutes les configs ont échoué pour le symbole {}. Tuning stoppé.", symbol);
            tuningExceptionReport.add(new TuningExceptionReportEntry(symbol, null, "Toutes les configs ont échoué (early stopping global)", "", System.currentTimeMillis()));
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
            logger.info("[TUNING] Fin tuning pour {} | Meilleure config : windowSize={}, neurons={}, dropout={}, lr={}, l1={}, l2={}, Test MSE={}, durée totale={} ms", symbol, bestConfig.getWindowSize(), bestConfig.getLstmNeurons(), bestConfig.getDropoutRate(), bestConfig.getLearningRate(), bestConfig.getL1(), bestConfig.getL2(), bestScore, (endSymbol - startSymbol));
        } else {
            logger.warn("[TUNING] Aucun modèle/scaler valide trouvé pour {}", symbol);
        }
        // Nettoyage mémoire ND4J/DL4J pour libérer le GPU
        try {
            org.nd4j.linalg.factory.Nd4j.getMemoryManager().invokeGc();
            org.nd4j.linalg.api.memory.MemoryWorkspaceManager wsManager = org.nd4j.linalg.factory.Nd4j.getWorkspaceManager();
            wsManager.destroyAllWorkspacesForCurrentThread();
            logger.info("Nettoyage mémoire ND4J/DL4J effectué après tuning.");
        } catch (Exception e) {
            logger.warn("Erreur lors du nettoyage ND4J/DL4J : {}", e.getMessage());
        }
        return bestConfig;
    }

    public LstmConfig tuneSymbol(String symbol, List<LstmConfig> grid, BarSeries series, JdbcTemplate jdbcTemplate) {
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
        logger.info("[TUNING] Début du tuning pour le symbole {} ({} configs)", symbol, grid.size());
        double bestScore = Double.MAX_VALUE;
        LstmConfig conf = hyperparamsRepository.loadHyperparams(symbol);
        if(conf != null){
            logger.info("Hyperparamètres existants trouvés pour {}. Ignorer le tuning.", symbol);
            return conf;
        }
        LstmConfig bestConfig = null;
        MultiLayerNetwork bestModel = null;
        LstmTradePredictor.ScalerSet bestScalers = null;
        int failedConfigs = 0;
        double bestBusinessScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < grid.size(); i++) {
            long startConfig = System.currentTimeMillis();
            LstmConfig config = grid.get(i);
            logger.info("[TUNING] [{}] Début config {}/{}", symbol, i+1, grid.size());
            double score = Double.POSITIVE_INFINITY;
            MultiLayerNetwork model = null;
            LstmTradePredictor.TrainResult trainResult = null;
            try {
                // Sélection du mode de validation selon cvMode
                String cvMode = config.getCvMode() != null ? config.getCvMode() : "split";
                if ("timeseries".equalsIgnoreCase(cvMode)) {
                    score = lstmTradePredictor.crossValidateLstmTimeSeriesSplit(series, config);
                } else if ("kfold".equalsIgnoreCase(cvMode)) {
                    score = lstmTradePredictor.crossValidateLstm(series, config);
                } else { // split simple
                    score = lstmTradePredictor.splitScoreLstm(series, config);
                }
                int numFeatures = config.getFeatures() != null ? config.getFeatures().size() : 1;
                model = lstmTradePredictor.initModel(
                    numFeatures,
                    1,
                    config.getLstmNeurons(),
                    config.getDropoutRate(),
                    config.getLearningRate(),
                    config.getOptimizer(),
                    config.getL1(),
                    config.getL2(),
                    config, true
                );
                trainResult = lstmTradePredictor.trainLstmWithScalers(series, config, model);
                model = trainResult.model;
                if (Double.isNaN(score) || Double.isInfinite(score)) {
                    failedConfigs++;
                    continue;
                }
            } catch (Exception e) {
                failedConfigs++;
                logger.error("Erreur lors du tuning config {}/{} : {}", i+1, grid.size(), e.getMessage());
                String stack = org.apache.commons.lang3.exception.ExceptionUtils.getStackTrace(e);
                tuningExceptionReport.add(new TuningExceptionReportEntry(symbol, config, e.getMessage(), stack, System.currentTimeMillis()));
                continue;
            }
            double rmse = Math.sqrt(score);
            double predicted = lstmTradePredictor.predictNextClose(symbol, series, config, model);
            double[] closes = lstmTradePredictor.extractCloseValues(series);
            double lastClose = closes[closes.length - 1];
            double delta = predicted - lastClose;
            double th = lstmTradePredictor.computeSwingTradeThreshold(series, config);
            String direction;
            if (delta > th) {
                direction = "up";
            } else if (delta < -th) {
                direction = "down";
            } else {
                direction = "stable";
            }
            // Calcul des métriques trading sur le jeu de test
            double[] tradingMetrics = lstmTradePredictor.calculateTradingMetricsAdvanced(series, config, model,  FEE_PCT, SLIP_PAGE_PCT);
            double profitTotal = tradingMetrics[0];
            int numTrades = (int) tradingMetrics[1];
            double profitFactor = tradingMetrics[2];
            double winRate = tradingMetrics[3];
            double maxDrawdown = tradingMetrics[4];
            // Sauvegarde des métriques
            hyperparamsRepository.saveTuningMetrics(symbol, config, score, rmse, direction,
                profitTotal, profitFactor, winRate, maxDrawdown, numTrades);
            long endConfig = System.currentTimeMillis();
            logger.info("[TUNING] [{}] Fin config {} | MSE={}, RMSE={}, direction={}, durée={} ms", symbol, grid.size(), score, rmse, direction, (endConfig - startConfig));
            // Mise à jour du suivi d'avancement
            TuningProgress p = tuningProgressMap.get(symbol);
            if (p != null) {
                p.testedConfigs.incrementAndGet();
                p.lastUpdate = System.currentTimeMillis();
            }

            logger.info("[TUNING] [{}] Fin config {}/{} | MSE={}, RMSE={}, direction={}, durée={} ms", symbol, i+1, grid.size(), score, rmse, direction, (endConfig - startConfig));
            if (score < bestScore) {
                bestScore = score;
                bestConfig = config;
                bestModel = model;
                bestScalers = trainResult.scalers;
            }
            double businessScore = computeBusinessScore(profitFactor, winRate, maxDrawdown);
            if (businessScore > bestBusinessScore) {
                bestBusinessScore = businessScore;
                bestConfig = config;
                bestModel = model;
                bestScalers = trainResult.scalers;
            }
            logger.info("[TUNING] [{}] Progression : {}/{} configs terminées", symbol, i+1, grid.size());
        }

        long endSymbol = System.currentTimeMillis();
        if (failedConfigs == grid.size()) {
            progress.status = "failed";
            progress.endTime = endSymbol;
            progress.lastUpdate = endSymbol;
            logger.error("[TUNING][EARLY STOP GLOBAL] Toutes les configs ont échoué pour le symbole {}. Tuning stoppé.", symbol);
            tuningExceptionReport.add(new TuningExceptionReportEntry(symbol, null, "Toutes les configs ont échoué (early stopping global)", "", System.currentTimeMillis()));
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
            logger.info("[TUNING] Fin tuning pour {} | Meilleure config : windowSize={}, neurons={}, dropout={}, lr={}, l1={}, l2={}, CV MSE={}, durée totale={} ms", symbol, bestConfig.getWindowSize(), bestConfig.getLstmNeurons(), bestConfig.getDropoutRate(), bestConfig.getLearningRate(), bestConfig.getL1(), bestConfig.getL2(), bestScore, (endSymbol - startSymbol));
        } else {
            logger.warn("[TUNING] Aucun modèle valide trouvé pour {}", symbol);
        }
        // Nettoyage mémoire ND4J/DL4J pour libérer le GPU
        try {
            org.nd4j.linalg.factory.Nd4j.getMemoryManager().invokeGc();
            org.nd4j.linalg.api.memory.MemoryWorkspaceManager wsManager = org.nd4j.linalg.factory.Nd4j.getWorkspaceManager();
            wsManager.destroyAllWorkspacesForCurrentThread();
            logger.info("Nettoyage mémoire ND4J/DL4J effectué après tuning.");
        } catch (Exception e) {
            logger.warn("Erreur lors du nettoyage ND4J/DL4J : {}", e.getMessage());
        }
        return bestConfig;
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
        String[] scopes = {"window", "global"};
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
        String[] scopes = {"window", "global"};
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
            // Ajout du mode de validation croisée
            config.setCvMode(cvMode);
            grid.add(config);
        }
        return grid;
    }

    /**
     * Lance le tuning automatique pour une liste de symboles.
     * @param symbols liste des symboles à tuner
     * @param jdbcTemplate accès base
     * @param seriesProvider fonction pour obtenir BarSeries par symbole
     */
    public void tuneAllSymbolsMultiThread(List<String> symbols, List<LstmConfig> grid, JdbcTemplate jdbcTemplate, java.util.function.Function<String, BarSeries> seriesProvider) {
        int numThreads = Math.min(symbols.size(), Runtime.getRuntime().availableProcessors());
        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(numThreads);
        java.util.concurrent.CompletionService<Void> completionService = new java.util.concurrent.ExecutorCompletionService<>(executor);
        int submitted = 0;
        long startAll = System.currentTimeMillis();
        logger.info("[TUNING] Début tuning multi-symboles ({} symboles)", symbols.size());
        for (String symbol : symbols) {
            completionService.submit(() -> {
                long startSymbol = System.currentTimeMillis();
                try {
                    BarSeries series = seriesProvider.apply(symbol);
                    tuneSymbol(symbol, grid, series, jdbcTemplate);
                } catch (Exception e) {
                    logger.error("Erreur dans le tuning du symbole {} : {}", symbol, e.getMessage());
                }
                long endSymbol = System.currentTimeMillis();
                logger.info("[TUNING] Fin tuning symbole {} | durée={} ms", symbol, (endSymbol - startSymbol));
                return null;
            });
            submitted++;
        }
        for (int i = 0; i < submitted; i++) {
            try {
                completionService.take();
            } catch (Exception e) {
                logger.error("Erreur lors de la récupération d'une tâche de tuning : {}", e.getMessage());
            }
        }
        executor.shutdown();
        try {
            if (!executor.awaitTermination(10, java.util.concurrent.TimeUnit.MINUTES)) {
                executor.shutdownNow();
                logger.warn("Arrêt forcé du pool de threads tuning");
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            logger.error("Interruption lors de l'arrêt du pool de threads tuning : {}", e.getMessage());
        }
        long endAll = System.currentTimeMillis();
        logger.info("[TUNING] Fin tuning multi-symboles | durée totale={} ms", (endAll - startAll));
    }

    public void tuneAllSymbols(List<String> symbols, List<LstmConfig> grid, JdbcTemplate jdbcTemplate, java.util.function.Function<String, BarSeries> seriesProvider) {
        long startAll = System.currentTimeMillis();
        logger.info("[TUNING] Début tuning multi-symboles ({} symboles)", symbols.size());
        for (int i = 0; i < symbols.size(); i++) {
            String symbol = symbols.get(i);
            long startSymbol = System.currentTimeMillis();
            logger.info("[TUNING] Début tuning symbole {}/{} : {}", i+1, symbols.size(), symbol);
            BarSeries series = seriesProvider.apply(symbol);
            tuneSymbol(symbol, grid, series, jdbcTemplate);
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

    /**
     * Calcule un score métier pour le swing trade en combinant profit factor, win rate et drawdown.
     * Plus le score est élevé, mieux c'est.
     */
    private double computeBusinessScore(double profitFactor, double winRate, double maxDrawdown) {
        // Exemple de formule : maximise profitFactor et winRate, pénalise le drawdown
        return profitFactor * winRate / (1.0 + maxDrawdown);
    }

    // Classe interne pour stocker le résultat du tuning
    private static class TuningResult {
        LstmConfig config;
        MultiLayerNetwork model;
        double score;
        LstmTradePredictor.ScalerSet scalers;
        double profitFactor;
        double winRate;
        double maxDrawdown;
        double businessScore;
        TuningResult(LstmConfig config, MultiLayerNetwork model, double score, LstmTradePredictor.ScalerSet scalers,
                     double profitFactor, double winRate, double maxDrawdown, double businessScore) {
            this.config = config;
            this.model = model;
            this.score = score;
            this.scalers = scalers;
            this.profitFactor = profitFactor;
            this.winRate = winRate;
            this.maxDrawdown = maxDrawdown;
            this.businessScore = businessScore;
        }
    }
}
