package com.app.backend.trade.lstm;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.ta4j.core.BarSeries;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.List;

@Service
public class LstmTuningService {
    private static final Logger logger = LoggerFactory.getLogger(LstmTuningService.class);
    private final LstmTradePredictor lstmTradePredictor;
    private final LstmHyperparamsRepository hyperparamsRepository;

    public LstmTuningService(LstmTradePredictor lstmTradePredictor, LstmHyperparamsRepository hyperparamsRepository) {
        this.lstmTradePredictor = lstmTradePredictor;
        this.hyperparamsRepository = hyperparamsRepository;
    }

    /**
     * Tune automatiquement les hyperparamètres pour un symbole donné.
     * @param symbol le symbole à tuner
     * @param grid la liste des configurations à tester
     * @param series les données historiques du symbole
     * @param jdbcTemplate accès base
     * @return la meilleure configuration trouvée
     */
    public LstmConfig tuneSymbol(String symbol, List<LstmConfig> grid, BarSeries series, JdbcTemplate jdbcTemplate) {
        double bestScore = Double.MAX_VALUE;
        LstmConfig conf = hyperparamsRepository.loadHyperparams(symbol);
        if(conf != null){
            logger.info("Hyperparamètres existants trouvés pour {}. Ignorer le tuning.", symbol);
            return conf;
        }
        LstmConfig bestConfig = null;
        MultiLayerNetwork bestModel = null;
        for (LstmConfig config : grid) {
            MultiLayerNetwork model = lstmTradePredictor.initModel(
                config.getWindowSize(),
                1,
                config.getLstmNeurons(),
                config.getDropoutRate(),
                config.getLearningRate(),
                config.getOptimizer(),
                config.getL1(),
                config.getL2()
            );
            model = lstmTradePredictor.trainLstm(series, config, model);
            // Évaluation sur le jeu de test
            double score = evaluateModel(model, series, config);
            logger.info("Tuning {} | Config: windowSize={}, neurons={}, dropout={}, lr={}, l1={}, l2={} | Test MSE={}", symbol, config.getWindowSize(), config.getLstmNeurons(), config.getDropoutRate(), config.getLearningRate(), config.getL1(), config.getL2(), score);
            if (score < bestScore) {
                bestScore = score;
                bestConfig = config;
                bestModel = model;
            }
        }
        // Sauvegarde la meilleure config et le modèle
        if (bestConfig != null && bestModel != null) {
            hyperparamsRepository.saveHyperparams(symbol, bestConfig);
            try {
                lstmTradePredictor.saveModelToDb(symbol, bestModel, jdbcTemplate, bestConfig);
            } catch (Exception e) {
                logger.error("Erreur lors de la sauvegarde du meilleur modèle : {}", e.getMessage());
            }
            logger.info("Meilleure config pour {} : windowSize={}, neurons={}, dropout={}, lr={}, l1={}, l2={}, Test MSE={}", symbol, bestConfig.getWindowSize(), bestConfig.getLstmNeurons(), bestConfig.getDropoutRate(), bestConfig.getLearningRate(), bestConfig.getL1(), bestConfig.getL2(), bestScore);
        }
        return bestConfig;
    }

    // Évalue le modèle sur le jeu de test (MSE)
    private double evaluateModel(MultiLayerNetwork model, BarSeries series, LstmConfig config) {
        double[] closes = lstmTradePredictor.extractCloseValues(series);
        double[] normalized = lstmTradePredictor.normalize(closes);
        int numSeq = normalized.length - config.getWindowSize();
        if (numSeq <= 0) {
            logger.warn("Pas assez de données pour évaluer le modèle (windowSize={}, closes={})", config.getWindowSize(), closes.length);
            return Double.POSITIVE_INFINITY;
        }
        double[][][] sequences = lstmTradePredictor.createSequences(normalized, config.getWindowSize());
        double[][][] labelSeq = new double[numSeq][1][1];
        for (int i = 0; i < numSeq; i++) {
            labelSeq[i][0][0] = normalized[i + config.getWindowSize()];
        }
        int splitIdx = (int)(numSeq * 0.8);
        if (splitIdx == numSeq) splitIdx = numSeq - 1;
        double[][][] testSeq = java.util.Arrays.copyOfRange(sequences, splitIdx, numSeq);
        double[][][] testLabel = java.util.Arrays.copyOfRange(labelSeq, splitIdx, numSeq);
        if (testSeq.length == 0 || testLabel.length == 0) {
            logger.warn("Jeu de test vide pour l'évaluation du modèle");
            return Double.POSITIVE_INFINITY;
        }
        org.nd4j.linalg.api.ndarray.INDArray testInput = lstmTradePredictor.toINDArray(testSeq);
        org.nd4j.linalg.api.ndarray.INDArray testOutput = org.nd4j.linalg.factory.Nd4j.create(testLabel);
        if (lstmTradePredictor.containsNaN(testOutput)) {
            logger.warn("TestOutput contient des NaN pour l'évaluation du modèle");
            return Double.POSITIVE_INFINITY;
        }
        org.nd4j.linalg.api.ndarray.INDArray predictions = model.output(testInput);
        if (lstmTradePredictor.containsNaN(predictions)) {
            logger.warn("Prédictions contiennent des NaN pour l'évaluation du modèle");
            return Double.POSITIVE_INFINITY;
        }
        // Vérification et alignement des shapes
        logger.info("Shape predictions: {}", java.util.Arrays.toString(predictions.shape()));
        logger.info("Shape testOutput: {}", java.util.Arrays.toString(testOutput.shape()));
        if (!java.util.Arrays.equals(predictions.shape(), testOutput.shape())) {
            try {
                predictions = predictions.reshape(testOutput.shape());
            } catch (Exception e) {
                logger.error("Erreur lors du reshape des prédictions : {}", e.getMessage());
                return Double.POSITIVE_INFINITY;
            }
        }
        double mse = Double.POSITIVE_INFINITY;
        try {
            // Calcul alternatif du MSE si squaredDistance échoue
            org.nd4j.linalg.api.ndarray.INDArray diff = predictions.sub(testOutput);
            org.nd4j.linalg.api.ndarray.INDArray squared = diff.mul(diff);
            mse = squared.meanNumber().doubleValue();
        } catch (Exception e) {
            logger.error("Erreur lors du calcul du MSE d'évaluation : {}", e.getMessage());
        }
        if (Double.isInfinite(mse) || Double.isNaN(mse)) {
            logger.warn("MSE infini ou NaN lors de l'évaluation du modèle");
            return Double.POSITIVE_INFINITY;
        }
        return mse;
    }

    /**
     * Génère automatiquement une grille de configurations adaptée au swing trade.
     * @return liste de LstmConfig à tester
     */
    public List<LstmConfig> generateSwingTradeGrid() {
        List<LstmConfig> grid = new java.util.ArrayList<>();
        int[] windowSizes = {20, 30, 40};
        int[] lstmNeurons = {64, 100, 128};
        double[] dropoutRates = {0.2, 0.3, 0.4};
        double[] learningRates = {0.0005, 0.001};
        double[] l1s = {0.0, 0.0001};
        double[] l2s = {0.0001, 0.001, 0.01};
        int numEpochs = 150;
        int patience = 15;
        double minDelta = 0.0002;
        int kFolds = 5;
        String optimizer = "adam";
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
                                grid.add(config);
                            }
                        }
                    }
                }
            }
        }
        return grid;
    }

    /**
     * Lance le tuning automatique pour une liste de symboles.
     * @param symbols liste des symboles à tuner
     * @param jdbcTemplate accès base
     * @param seriesProvider fonction pour obtenir BarSeries par symbole
     */
    public void tuneAllSymbols_bis(List<String> symbols, JdbcTemplate jdbcTemplate, java.util.function.Function<String, BarSeries> seriesProvider) {
        List<LstmConfig> grid = generateSwingTradeGrid();
        int numThreads = Math.min(symbols.size(), Runtime.getRuntime().availableProcessors());
        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(numThreads);
        java.util.concurrent.CompletionService<Void> completionService = new java.util.concurrent.ExecutorCompletionService<>(executor);
        int submitted = 0;
        for (String symbol : symbols) {
            completionService.submit(() -> {
                try {
                    BarSeries series = seriesProvider.apply(symbol);
                    tuneSymbol(symbol, grid, series, jdbcTemplate);
                } catch (Exception e) {
                    logger.error("Erreur dans le tuning du symbole {} : {}", symbol, e.getMessage());
                }
                return null;
            });
            submitted++;
        }
        // Consommer les résultats au fur et à mesure qu'ils terminent
        for (int i = 0; i < submitted; i++) {
            try {
                completionService.take(); // attends qu'une tâche se termine
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
    }

    /**
     * Lance le tuning automatique pour une liste de symboles.
     * @param symbols liste des symboles à tuner
     * @param jdbcTemplate accès base
     * @param seriesProvider fonction pour obtenir BarSeries par symbole
     */
    public void tuneAllSymbols(List<String> symbols, JdbcTemplate jdbcTemplate, java.util.function.Function<String, BarSeries> seriesProvider) {
        List<LstmConfig> grid = generateSwingTradeGrid();
        for (String symbol : symbols) {
            BarSeries series = seriesProvider.apply(symbol);
            tuneSymbol(symbol, grid, series, jdbcTemplate);
        }
    }
}
