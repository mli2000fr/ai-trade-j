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

    public LstmTuningService(LstmTradePredictor lstmTradePredictor, LstmHyperparamsRepository hyperparamsRepository) {
        this.lstmTradePredictor = lstmTradePredictor;
        this.hyperparamsRepository = hyperparamsRepository;
    }

    public ConcurrentHashMap<String, TuningProgress> getTuningProgressMap() {
        return tuningProgressMap;
    }

    /**
     * Tune automatiquement les hyperparamètres pour un symbole donné.
     * @param symbol le symbole à tuner
     * @param grid la liste des configurations à tester
     * @param series les données historiques du symbole
     * @param jdbcTemplate accès base
     * @return la meilleure configuration trouvée
     */
    public LstmConfig tuneSymbolMultiThread(String symbol, List<LstmConfig> grid, BarSeries series, JdbcTemplate jdbcTemplate, boolean useTimeSeriesCV) {
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
        int numThreads = Math.min(grid.size(), Runtime.getRuntime().availableProcessors());
        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(numThreads);
        List<java.util.concurrent.Future<TuningResult>> futures = new java.util.ArrayList<>();
        for (int i = 0; i < grid.size(); i++) {
            final int configIndex = i + 1;
            LstmConfig config = grid.get(i);
            futures.add(executor.submit(() -> {
                long startConfig = System.currentTimeMillis();
                logger.info("[TUNING] [{}] {} | Début config {}/{}", symbol, Thread.currentThread().getName(), configIndex, grid.size());
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
                        config
                );
                model = lstmTradePredictor.trainLstm(series, config, model);
                double score = useTimeSeriesCV
                    ? lstmTradePredictor.crossValidateLstmTimeSeriesSplit(series, config)
                    : lstmTradePredictor.crossValidateLstm(series, config);
                double rmse = Math.sqrt(score);
                double predicted = lstmTradePredictor.predictNextClose(symbol, series, config, model);
                double[] closes = lstmTradePredictor.extractCloseValues(series);
                double lastClose = closes[closes.length - 1];
                double delta = predicted - lastClose;
                double th = lstmTradePredictor.computeSwingTradeThreshold(series);
                String direction;
                if (delta > th) {
                    direction = "up";
                } else if (delta < -th) {
                    direction = "down";
                } else {
                    direction = "stable";
                }
                hyperparamsRepository.saveTuningMetrics(symbol, config, score, rmse, direction);
                long endConfig = System.currentTimeMillis();
                logger.info("[TUNING] [{}] Fin config {}/{} | MSE={}, RMSE={}, direction={}, durée={} ms", symbol, configIndex, grid.size(), score, rmse, direction, (endConfig - startConfig));
                // Mise à jour du suivi d'avancement
                TuningProgress p = tuningProgressMap.get(symbol);
                if (p != null) {
                    p.testedConfigs.incrementAndGet();
                    p.lastUpdate = System.currentTimeMillis();
                }
                return new TuningResult(config, model, score);
            }));
        }
        executor.shutdown();
        for (int i = 0; i < futures.size(); i++) {
            try {
                TuningResult result = futures.get(i).get();
                logger.info("[TUNING] [{}] Progression : {}/{} configs terminées", symbol, i+1, grid.size());
                if (result.score < bestScore) {
                    bestScore = result.score;
                    bestConfig = result.config;
                    bestModel = result.model;
                }
            } catch (Exception e) {
                progress.status = "erreur";
                progress.lastUpdate = System.currentTimeMillis();
                logger.error("Erreur lors de la récupération du résultat de tuning : {}", e.getMessage());
            }
        }
        long endSymbol = System.currentTimeMillis();
        progress.status = "termine";
        progress.endTime = endSymbol;
        progress.lastUpdate = endSymbol;
        if (bestConfig != null && bestModel != null) {
            hyperparamsRepository.saveHyperparams(symbol, bestConfig);
            try {
                lstmTradePredictor.saveModelToDb(symbol, bestModel, jdbcTemplate, bestConfig);
            } catch (Exception e) {
                logger.error("Erreur lors de la sauvegarde du meilleur modèle : {}", e.getMessage());
            }
            logger.info("[TUNING] Fin tuning pour {} | Meilleure config : windowSize={}, neurons={}, dropout={}, lr={}, l1={}, l2={}, Test MSE={}, durée totale={} ms", symbol, bestConfig.getWindowSize(), bestConfig.getLstmNeurons(), bestConfig.getDropoutRate(), bestConfig.getLearningRate(), bestConfig.getL1(), bestConfig.getL2(), bestScore, (endSymbol - startSymbol));
        } else {
            logger.warn("[TUNING] Aucun modèle valide trouvé pour {}", symbol);
        }
        return bestConfig;
    }

    public LstmConfig tuneSymbol(String symbol, List<LstmConfig> grid, BarSeries series, JdbcTemplate jdbcTemplate, boolean useTimeSeriesCV) {
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
        for (int i = 0; i < grid.size(); i++) {
            long startConfig = System.currentTimeMillis();
            LstmConfig config = grid.get(i);
            logger.info("[TUNING] [{}] Début config {}/{}", symbol, i+1, grid.size());
            double score = useTimeSeriesCV
                ? lstmTradePredictor.crossValidateLstmTimeSeriesSplit(series, config)
                : lstmTradePredictor.crossValidateLstm(series, config);
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
                    config
            );
            model = lstmTradePredictor.trainLstm(series, config, model);
            double rmse = Math.sqrt(score);
            double predicted = lstmTradePredictor.predictNextClose(symbol, series, config, model);
            double[] closes = lstmTradePredictor.extractCloseValues(series);
            double lastClose = closes[closes.length - 1];
            double delta = predicted - lastClose;
            double th = lstmTradePredictor.computeSwingTradeThreshold(series);
            String direction;
            if (delta > th) {
                direction = "up";
            } else if (delta < -th) {
                direction = "down";
            } else {
                direction = "stable";
            }
            hyperparamsRepository.saveTuningMetrics(symbol, config, score, rmse, direction);
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
            }
            logger.info("[TUNING] [{}] Progression : {}/{} configs terminées", symbol, i+1, grid.size());
        }

        long endSymbol = System.currentTimeMillis();
        progress.status = "termine";
        progress.endTime = endSymbol;
        progress.lastUpdate = endSymbol;
        if (bestConfig != null && bestModel != null) {
            hyperparamsRepository.saveHyperparams(symbol, bestConfig);
            try {
                lstmTradePredictor.saveModelToDb(symbol, bestModel, jdbcTemplate, bestConfig);
            } catch (Exception e) {
                logger.error("Erreur lors de la sauvegarde du meilleur modèle : {}", e.getMessage());
            }
            logger.info("[TUNING] Fin tuning pour {} | Meilleure config : windowSize={}, neurons={}, dropout={}, lr={}, l1={}, l2={}, CV MSE={}, durée totale={} ms", symbol, bestConfig.getWindowSize(), bestConfig.getLstmNeurons(), bestConfig.getDropoutRate(), bestConfig.getLearningRate(), bestConfig.getL1(), bestConfig.getL2(), bestScore, (endSymbol - startSymbol));
        } else {
            logger.warn("[TUNING] Aucun modèle valide trouvé pour {}", symbol);
        }
        return bestConfig;
    }




    /**
     * Génère automatiquement une grille de configurations adaptée au swing trade.
     * @return liste de LstmConfig à tester
     */
    public List<LstmConfig> generateSwingTradeGrid() {
        List<LstmConfig> grid = new java.util.ArrayList<>();
        int[] windowSizes = {10, 20, 30, 40, 60};
        int[] lstmNeurons = {64, 100, 128, 256, 512};
        double[] dropoutRates = {0.2, 0.3, 0.4};
        double[] learningRates = {0.0005, 0.001, 0.002};
        double[] l1s = {0.0, 0.0001};
        double[] l2s = {0.0001, 0.001, 0.01};
        int numEpochs = 300;
        int patience = 20;
        double minDelta = 0.0002;
        int kFolds = 5;
        String optimizer = "adam";
        String[] scopes = {"window", "global"};
        String[] swingTypes = {"range", "breakout", "mean_reversion"};
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
        int[] windowSizes = {10, 20, 30, 40, 60};
        int[] lstmNeurons = {64, 100, 128, 256, 512};
        double[] dropoutRates = {0.2, 0.3, 0.4};
        double[] learningRates = {0.0005, 0.001, 0.002};
        double[] l1s = {0.0, 0.0001};
        double[] l2s = {0.0001, 0.001, 0.01};
        int numEpochs = 300;
        int patience = 20;
        double minDelta = 0.0002;
        int kFolds = 5;
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
                    tuneSymbol(symbol, grid, series, jdbcTemplate, true);
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
            tuneSymbolMultiThread(symbol, grid, series, jdbcTemplate, true);
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

    // Classe interne pour stocker le résultat du tuning
    private static class TuningResult {
        LstmConfig config;
        MultiLayerNetwork model;
        double score;
        TuningResult(LstmConfig config, MultiLayerNetwork model, double score) {
            this.config = config;
            this.model = model;
            this.score = score;
        }
    }
}
