package com.app.backend.trade.lstm;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.sql.SQLException;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.dao.EmptyResultDataAccessException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.ta4j.core.BarSeries;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Service
public class LstmTradePredictor {
    private MultiLayerNetwork model;
    private static final Logger logger = LoggerFactory.getLogger(LstmTradePredictor.class);

    public LstmTradePredictor() {
        // Le modèle sera initialisé via la méthode initModel
    }

    /**
     * Initialise le modèle LSTM avec Dropout.
     * @param inputSize taille de l'entrée
     * @param outputSize taille de la sortie
     * @param lstmNeurons nombre de neurones dans la couche LSTM
     * @param dropoutRate taux de Dropout (ex : 0.2)
     */
    public void initModel(int inputSize, int outputSize, int lstmNeurons, double dropoutRate) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .list()
            .layer(new LSTM.Builder()
                .nIn(inputSize)
                .nOut(lstmNeurons)
                .activation(Activation.TANH)
                .build())
            .layer(new DropoutLayer.Builder()
                .dropOut(dropoutRate)
                .build())
            .layer(new RnnOutputLayer.Builder()
                .nIn(lstmNeurons)
                .nOut(outputSize)
                .activation(Activation.IDENTITY)
                .lossFunction(LossFunctions.LossFunction.MSE)
                .build())
            .build();
        model = new MultiLayerNetwork(conf);
        model.init();
    }

    // Ancienne version conservée pour compatibilité
    /**
     * @deprecated Utiliser initModel avec lstmNeurons et dropoutRate
     */
    @Deprecated
    public void initModel(int inputSize, int outputSize, int lstmNeurons) {
        initModel(inputSize, outputSize, lstmNeurons, 0.2);
    }
    /**
     * @deprecated Utiliser initModel avec lstmNeurons et dropoutRate
     */
    @Deprecated
    public void initModel(int inputSize, int outputSize) {
        initModel(inputSize, outputSize, 50, 0.2);
    }

    // Extraction des valeurs de clôture
    public double[] extractCloseValues(BarSeries series) {
        double[] closes = new double[series.getBarCount()];
        for (int i = 0; i < series.getBarCount(); i++) {
            closes[i] = series.getBar(i).getClosePrice().doubleValue();
        }
        return closes;
    }

    // Normalisation MinMax
    public double[] normalize(double[] values) {
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        for (double v : values) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        double[] normalized = new double[values.length];
        if (min == max) {
            // Toutes les valeurs sont identiques, éviter division par zéro
            for (int i = 0; i < values.length; i++) {
                normalized[i] = 0.5; // ou 0.0, ou 1.0 selon convention
            }
        } else {
            for (int i = 0; i < values.length; i++) {
                normalized[i] = (values[i] - min) / (max - min);
            }
        }
        return normalized;
    }

    // Création des séquences pour LSTM
    public double[][][] createSequences(double[] values, int windowSize) {
        int numSeq = values.length - windowSize;
        double[][][] sequences = new double[numSeq][windowSize][1];
        for (int i = 0; i < numSeq; i++) {
            for (int j = 0; j < windowSize; j++) {
                sequences[i][j][0] = values[i + j];
            }
        }
        return sequences;
    }

    // Conversion en INDArray
    public org.nd4j.linalg.api.ndarray.INDArray toINDArray(double[][][] sequences) {
        return org.nd4j.linalg.factory.Nd4j.create(sequences);
    }

    // Préparation complète des données pour LSTM
    public org.nd4j.linalg.api.ndarray.INDArray prepareLstmInput(BarSeries series, int windowSize) {
        double[] closes = extractCloseValues(series);
        double[] normalized = normalize(closes);
        double[][][] sequences = createSequences(normalized, windowSize);
        return toINDArray(sequences);
    }

    /**
     * Entraîne le modèle LSTM avec séparation train/test (80/20) et early stopping.
     * Arrête l'entraînement si le score MSE sur le jeu de test ne s'améliore plus selon patience/minDelta.
     * @param series Série de bougies
     * @param windowSize Taille de la fenêtre
     * @param numEpochs Nombre maximal d'epochs
     * @param patience Nombre d'epochs sans amélioration avant arrêt
     * @param minDelta Amélioration minimale pour considérer le score comme meilleur
     */
    public void trainLstm(BarSeries series, int windowSize, int numEpochs, int patience, double minDelta) {
        double[] closes = extractCloseValues(series);
        double[] normalized = normalize(closes);
        double[][][] sequences = createSequences(normalized, windowSize);
        // Labels : valeur suivante après chaque séquence
        double[][][] labelSeq = new double[sequences.length][1][1];
        for (int i = 0; i < labelSeq.length; i++) {
            labelSeq[i][0][0] = normalized[i + windowSize];
        }
        // Séparation train/test (80/20)
        int splitIdx = (int)(sequences.length * 0.8);
        double[][][] trainSeq = java.util.Arrays.copyOfRange(sequences, 0, splitIdx);
        double[][][] testSeq = java.util.Arrays.copyOfRange(sequences, splitIdx, sequences.length);
        double[][][] trainLabel = java.util.Arrays.copyOfRange(labelSeq, 0, splitIdx);
        double[][][] testLabel = java.util.Arrays.copyOfRange(labelSeq, splitIdx, labelSeq.length);
        org.nd4j.linalg.api.ndarray.INDArray trainInput = toINDArray(trainSeq);
        org.nd4j.linalg.api.ndarray.INDArray trainOutput = org.nd4j.linalg.factory.Nd4j.create(trainLabel);
        org.nd4j.linalg.api.ndarray.INDArray testInput = toINDArray(testSeq);
        org.nd4j.linalg.api.ndarray.INDArray testOutput = org.nd4j.linalg.factory.Nd4j.create(testLabel);
        org.nd4j.linalg.dataset.api.iterator.DataSetIterator trainIterator = new ListDataSetIterator<>(
            java.util.Collections.singletonList(new org.nd4j.linalg.dataset.DataSet(trainInput, trainOutput))
        );
        // Early stopping
        double bestScore = Double.MAX_VALUE;
        int epochsWithoutImprovement = 0;
        int actualEpochs = 0;
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainIterator);
            actualEpochs++;
            org.nd4j.linalg.api.ndarray.INDArray predictions = model.output(testInput);
            double mse = predictions.squaredDistance(testOutput) / testOutput.length();
            logger.info("Epoch {} terminé, Test MSE : {}", i + 1, mse);
            if (bestScore - mse > minDelta) {
                bestScore = mse;
                epochsWithoutImprovement = 0;
            } else {
                epochsWithoutImprovement++;
                if (epochsWithoutImprovement >= patience) {
                    logger.info("Early stopping déclenché à l'epoch {}. Meilleur Test MSE : {}", i + 1, bestScore);
                    break;
                }
            }
        }
        logger.info("Entraînement terminé après {} epochs. Meilleur Test MSE : {}", actualEpochs, bestScore);
    }

    /**
     * Effectue une validation croisée k-fold sur le modèle LSTM.
     * Logge le score MSE moyen et l'écart-type sur les k folds.
     * @param series Série de bougies
     * @param windowSize Taille de la fenêtre
     * @param numEpochs Nombre maximal d'epochs
     * @param kFolds Nombre de folds pour la cross-validation
     * @param lstmNeurons Nombre de neurones LSTM
     * @param dropoutRate Taux de Dropout
     * @param patience Nombre d'epochs sans amélioration avant early stopping
     * @param minDelta Amélioration minimale pour considérer le score comme meilleur
     */
    public void crossValidateLstm(BarSeries series, int windowSize, int numEpochs, int kFolds, int lstmNeurons, double dropoutRate, int patience, double minDelta) {
        double[] closes = extractCloseValues(series);
        double[] normalized = normalize(closes);
        double[][][] sequences = createSequences(normalized, windowSize);
        double[][][] labelSeq = new double[sequences.length][1][1];
        for (int i = 0; i < labelSeq.length; i++) {
            labelSeq[i][0][0] = normalized[i + windowSize];
        }
        int foldSize = sequences.length / kFolds;
        double[] foldScores = new double[kFolds];
        for (int fold = 0; fold < kFolds; fold++) {
            // Définir les indices de test
            int testStart = fold * foldSize;
            int testEnd = (fold == kFolds - 1) ? sequences.length : testStart + foldSize;
            // Split test
            double[][][] testSeq = java.util.Arrays.copyOfRange(sequences, testStart, testEnd);
            double[][][] testLabel = java.util.Arrays.copyOfRange(labelSeq, testStart, testEnd);
            // Split train
            double[][][] trainSeq = new double[sequences.length - (testEnd - testStart)][windowSize][1];
            double[][][] trainLabel = new double[sequences.length - (testEnd - testStart)][1][1];
            int idx = 0;
            for (int i = 0; i < sequences.length; i++) {
                if (i < testStart || i >= testEnd) {
                    trainSeq[idx] = sequences[i];
                    trainLabel[idx] = labelSeq[i];
                    idx++;
                }
            }
            org.nd4j.linalg.api.ndarray.INDArray trainInput = toINDArray(trainSeq);
            org.nd4j.linalg.api.ndarray.INDArray trainOutput = org.nd4j.linalg.factory.Nd4j.create(trainLabel);
            org.nd4j.linalg.api.ndarray.INDArray testInput = toINDArray(testSeq);
            org.nd4j.linalg.api.ndarray.INDArray testOutput = org.nd4j.linalg.factory.Nd4j.create(testLabel);
            org.nd4j.linalg.dataset.api.iterator.DataSetIterator trainIterator = new ListDataSetIterator<>(
                java.util.Collections.singletonList(new org.nd4j.linalg.dataset.DataSet(trainInput, trainOutput))
            );
            // Initialiser un nouveau modèle pour ce fold
            initModel(windowSize, 1, lstmNeurons, dropoutRate);
            // Early stopping
            double bestScore = Double.MAX_VALUE;
            int epochsWithoutImprovement = 0;
            for (int epoch = 0; epoch < numEpochs; epoch++) {
                model.fit(trainIterator);
                org.nd4j.linalg.api.ndarray.INDArray predictions = model.output(testInput);
                double mse = predictions.squaredDistance(testOutput) / testOutput.length();
                if (bestScore - mse > minDelta) {
                    bestScore = mse;
                    epochsWithoutImprovement = 0;
                } else {
                    epochsWithoutImprovement++;
                    if (epochsWithoutImprovement >= patience) {
                        logger.info("Early stopping fold {} à l'epoch {}. Meilleur Test MSE : {}", fold + 1, epoch + 1, bestScore);
                        break;
                    }
                }
            }
            foldScores[fold] = bestScore;
            logger.info("Fold {} terminé. Meilleur Test MSE : {}", fold + 1, bestScore);
        }
        // Calculer la moyenne et l'écart-type
        double sum = 0.0;
        for (double score : foldScores) sum += score;
        double mean = sum / kFolds;
        double variance = 0.0;
        for (double score : foldScores) variance += Math.pow(score - mean, 2);
        double std = Math.sqrt(variance / kFolds);
        logger.info("Validation croisée terminée. MSE moyen : {}, Ecart-type : {}", mean, std);
    }

    // Prédiction de la prochaine valeur de clôture
    public double predictNextClose(BarSeries series, int windowSize) {
        double[] closes = extractCloseValues(series);
        double[] normalized = normalize(closes);
        // Prendre la dernière séquence
        double[][][] lastSequence = new double[1][windowSize][1];
        for (int j = 0; j < windowSize; j++) {
            lastSequence[0][j][0] = normalized[normalized.length - windowSize + j];
        }
        org.nd4j.linalg.api.ndarray.INDArray input = toINDArray(lastSequence);
        org.nd4j.linalg.api.ndarray.INDArray output = model.output(input);
        // Dénormaliser la prédiction
        double min = java.util.Arrays.stream(closes).min().getAsDouble();
        double max = java.util.Arrays.stream(closes).max().getAsDouble();
        double predictedNorm = output.getDouble(0);
        double predicted = predictedNorm * (max - min) + min;
        return predicted;
    }

    // Sauvegarde du modèle
    public void saveModel(String path) throws IOException {
        if (model != null) {
            File file = new File(path);
            File parent = file.getParentFile();
            if (parent != null && !parent.exists()) {
                parent.mkdirs();
            }
            try {
                ModelSerializer.writeModel(model, file, true);
                logger.info("Modèle sauvegardé dans le fichier : {}", path);
            } catch (IOException e) {
                logger.error("Erreur lors de la sauvegarde du modèle : {}", e.getMessage());
                throw e;
            }
        }
    }

    // Chargement du modèle
    public void loadModel(String path) throws IOException {
        File f = new File(path);
        if (f.exists()) {
            try {
                model = ModelSerializer.restoreMultiLayerNetwork(f);
                logger.info("Modèle chargé depuis le fichier : {}", path);
            } catch (IOException e) {
                logger.error("Erreur lors du chargement du modèle : {}", e.getMessage());
                throw e;
            }
        }
    }

    // Sauvegarde du modèle dans MySQL
    public void saveModelToDb(String symbol, JdbcTemplate jdbcTemplate) throws IOException {
        if (model != null) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ModelSerializer.writeModel(model, baos, true);
            byte[] modelBytes = baos.toByteArray();
            String sql = "REPLACE INTO lstm_models (symbol, model_blob, updated_date) VALUES (?, ?, CURRENT_TIMESTAMP)";
            try {
                jdbcTemplate.update(sql, symbol, modelBytes);
                logger.info("Modèle sauvegardé en base pour le symbole : {}", symbol);
            } catch (Exception e) {
                logger.error("Erreur lors de la sauvegarde du modèle en base : {}", e.getMessage());
                throw e;
            }
        }
    }

    // Chargement du modèle depuis MySQL
    public void loadModelFromDb(String symbol, JdbcTemplate jdbcTemplate) throws IOException, SQLException {
        String sql = "SELECT model_blob FROM lstm_models WHERE symbol = ?";
        try {
            byte[] modelBytes = jdbcTemplate.queryForObject(sql, new Object[]{symbol}, byte[].class);
            if (modelBytes != null) {
                ByteArrayInputStream bais = new ByteArrayInputStream(modelBytes);
                model = ModelSerializer.restoreMultiLayerNetwork(bais);
                logger.info("Modèle chargé depuis la base pour le symbole : {}", symbol);
            }
        } catch (EmptyResultDataAccessException e) {
            logger.error("Modèle non trouvé en base pour le symbole : {}", symbol);
            throw new IOException("Modèle non trouvé en base");
        } catch (Exception e) {
            logger.error("Erreur lors du chargement du modèle en base : {}", e.getMessage());
            throw e;
        }
    }
}
