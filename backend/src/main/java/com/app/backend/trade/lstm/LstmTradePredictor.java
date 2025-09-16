/**
 * Service de prédiction LSTM pour le trading.
 * <p>
 * Permet d'entraîner, valider, sauvegarder, charger et utiliser un modèle LSTM
 * pour prédire la prochaine clôture, le delta ou la classe (hausse/baisse/stable)
 * d'une série de données financières.
 * Les hyperparamètres sont configurés dynamiquement via {@link LstmConfig}.
 * </p>
 */
package com.app.backend.trade.lstm;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.sql.SQLException;
import java.time.format.DateTimeFormatter;

import com.app.backend.trade.model.PreditLsdm;
import com.app.backend.trade.model.SignalType;
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
import com.app.backend.trade.exception.InsufficientDataException;
import com.app.backend.trade.exception.ModelNotFoundException;

@Service
public class LstmTradePredictor {
    private static final Logger logger = LoggerFactory.getLogger(LstmTradePredictor.class);

    private final LstmHyperparamsRepository hyperparamsRepository;

    public LstmTradePredictor(LstmHyperparamsRepository hyperparamsRepository) {
        this.hyperparamsRepository = hyperparamsRepository;
    }

    /**
     * Constructeur principal.
     * Initialise le modèle LSTM avec les paramètres du fichier de configuration.
     * @param config configuration des hyperparamètres LSTM
     */
    public void initWithConfig(LstmConfig config) {
        // Initialisation automatique du modèle avec les paramètres du fichier de config
        initModel(
            config.getWindowSize(),
            1,
            config.getLstmNeurons(),
            config.getDropoutRate(),
            config.getLearningRate(),
            config.getOptimizer(),
            config.getL1(),
            config.getL2()
        );
    }

    /**
     * Initialise le modèle LSTM avec les hyperparamètres fournis.
     * @param inputSize taille de l'entrée (doit être égal au windowSize utilisé)
     * @param outputSize taille de la sortie
     * @param lstmNeurons nombre de neurones dans la couche LSTM
     * @param dropoutRate taux de Dropout (ex : 0.2)
     * @param learningRate taux d'apprentissage
     * @param optimizer nom de l'optimiseur ("adam", "rmsprop", "sgd")
     */
    public MultiLayerNetwork initModel(int inputSize, int outputSize, int lstmNeurons, double dropoutRate, double learningRate, String optimizer, double l1, double l2) {
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.updater(
            "adam".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.Adam(learningRate)
            : "rmsprop".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.RmsProp(learningRate)
            : new org.nd4j.linalg.learning.config.Sgd(learningRate)
        );
        builder.l1(l1);
        builder.l2(l2);
        MultiLayerConfiguration conf = builder
            .list()
            .layer(new LSTM.Builder()
                .nIn(inputSize)
                .nOut(lstmNeurons)
                .activation(Activation.TANH)
                .build())
            .layer(new DropoutLayer.Builder()
                .dropOut(dropoutRate)
                .build())
            .layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                .nIn(lstmNeurons)
                .nOut(Math.max(16, lstmNeurons / 4))
                .activation(Activation.RELU)
                .build())
            .layer(new RnnOutputLayer.Builder()
                .nIn(Math.max(16, lstmNeurons / 4))
                .nOut(outputSize)
                .activation(Activation.IDENTITY)
                .lossFunction(LossFunctions.LossFunction.MSE)
                .build())
            .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    /**
     * Vérifie et réinitialise le modèle si le nombre de features demandé est différent du modèle courant.
     */
    private MultiLayerNetwork ensureModelWindowSize(MultiLayerNetwork model, int numFeatures, LstmConfig config) {
        if (model == null) {
            logger.info("Réinitialisation du modèle LSTM : numFeatures demandé = {}", numFeatures);
            return initModel(
                numFeatures, // Correction ici : nombre de features
                1,
                config.getLstmNeurons(),
                config.getDropoutRate(),
                config.getLearningRate(),
                config.getOptimizer(),
                config.getL1(),
                config.getL2()
            );
        } else {
            logger.info("Modèle LSTM déjà initialisé");
            return model;
        }
    }


    /**
     * Extrait les valeurs de clôture d'une série de bougies.
     * @param series série de bougies TA4J
     * @return tableau des valeurs de clôture
     */
    // Extraction des valeurs de clôture
    public double[] extractCloseValues(BarSeries series) {
        double[] closes = new double[series.getBarCount()];
        for (int i = 0; i < series.getBarCount(); i++) {
            closes[i] = series.getBar(i).getClosePrice().doubleValue();
        }
        return closes;
    }

    /**
     * Normalise un tableau de valeurs selon la méthode MinMax.
     * @param values tableau de valeurs à normaliser
     * @return tableau normalisé entre 0 et 1
     */
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

    /**
     * Normalise un tableau de valeurs selon la méthode MinMax, avec choix du scope.
     * @param values tableau de valeurs à normaliser
     * @param min valeur minimale de référence
     * @param max valeur maximale de référence
     * @return tableau normalisé entre 0 et 1
     */
    public double[] normalize(double[] values, double min, double max) {
        double[] normalized = new double[values.length];
        if (min == max) {
            for (int i = 0; i < values.length; i++) {
                normalized[i] = 0.5;
            }
        } else {
            for (int i = 0; i < values.length; i++) {
                normalized[i] = (values[i] - min) / (max - min);
            }
        }
        return normalized;
    }

    /**
     * Crée les séquences d'entrée pour le LSTM à partir des valeurs normalisées.
     * @param values tableau de valeurs normalisées
     * @param windowSize taille de la fenêtre
     * @return séquences 3D pour le LSTM
     */
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

    /**
     * Convertit les séquences en INDArray pour ND4J.
     * @param sequences séquences 3D
     * @return INDArray ND4J
     */
    // Conversion en INDArray
    public org.nd4j.linalg.api.ndarray.INDArray toINDArray(double[][][] sequences) {
        return org.nd4j.linalg.factory.Nd4j.create(sequences);
    }

    /**
     * Prépare l'entrée LSTM complète à partir d'une série de bougies.
     * @param series série de bougies
     * @param windowSize taille de la fenêtre
     * @return INDArray prêt pour le modèle LSTM
     */
    // Préparation complète des données pour LSTM
    public org.nd4j.linalg.api.ndarray.INDArray prepareLstmInput(BarSeries series, int windowSize, LstmConfig config) {
        double[] closes = extractCloseValues(series);
        double min = java.util.Arrays.stream(closes).min().orElse(0.0);
        double max = java.util.Arrays.stream(closes).max().orElse(0.0);
        double[] normalized = normalizeByConfig(closes, min, max, config);
        double[][][] sequences = createSequences(normalized, windowSize);
        return toINDArray(sequences);
    }

    /**
     * Prépare l'entrée LSTM complète à partir d'une série de bougies et d'une liste de features.
     * @param series série de bougies
     * @param windowSize taille de la fenêtre
     * @param features liste des features à inclure
     * @return INDArray prêt pour le modèle LSTM
     */
    public org.nd4j.linalg.api.ndarray.INDArray prepareLstmInputMulti(BarSeries series, int windowSize, java.util.List<String> features) {
        double[][] matrix = extractFeatureMatrix(series, features);
        double[][] normMatrix = normalizeMatrix(matrix);
        double[][][] sequences = createSequencesMulti(normMatrix, windowSize);
        return toINDArray(sequences);
    }

    /**
     * Entraîne le modèle LSTM avec early stopping et séparation train/test.
     * @param series série de bougies
     * @param windowSize taille de la fenêtre
     * @param numEpochs nombre maximal d'epochs
     * @param patience nombre d'epochs sans amélioration avant arrêt
     * @param minDelta amélioration minimale pour considérer le score comme meilleur
     */
    /**
     * Entraîne le modèle LSTM avec séparation train/test (80/20) et early stopping.
     * Arrête l'entraînement si le score MSE sur le jeu de test ne s'améliore plus selon patience/minDelta.
     * @param series Série de bougies
     */
    public MultiLayerNetwork trainLstm(BarSeries series, LstmConfig config, MultiLayerNetwork model) {
        java.util.List<String> features = config.getFeatures();
        int numFeatures = features.size();
        model = ensureModelWindowSize(model, numFeatures, config);
        double[][] matrix = extractFeatureMatrix(series, features);
        double[][] normMatrix = normalizeMatrix(matrix);
        int numSeq = normMatrix.length - config.getWindowSize();
        if (numSeq <= 0) {
            logger.error("Pas assez de données pour entraîner le modèle (windowSize={}, barCount={})", config.getWindowSize(), normMatrix.length);
            throw new IllegalArgumentException("Pas assez de données pour entraîner le modèle");
        }
        double[][][] sequences = createSequencesMulti(normMatrix, config.getWindowSize());
        // Correction : transposer les séquences pour DL4J [batch, numFeatures, windowSize]
        double[][][] sequencesTransposed = transposeSequencesMulti(sequences);
        // Correction : extraire la dernière valeur de chaque séquence pour input [batch, numFeatures, 1]
        double[][][] lastStepSeq = new double[numSeq][numFeatures][1];
        for (int i = 0; i < numSeq; i++) {
            for (int f = 0; f < numFeatures; f++) {
                lastStepSeq[i][f][0] = sequencesTransposed[i][f][config.getWindowSize() - 1];
            }
        }
        // Utiliser lastStepSeq pour l'input (et non sequencesTransposed)
        double[] closes = extractCloseValues(series);
        double[] normCloses = normalize(closes);
        // Correction: labels sous forme [numSeq, 1, 1] pour DL4J
        double[][][] labelSeq = new double[numSeq][1][1];
        for (int i = 0; i < numSeq; i++) {
            labelSeq[i][0][0] = normCloses[i + config.getWindowSize()];
        }
        int splitIdx = (int)(numSeq * 0.8);
        if (splitIdx == numSeq) splitIdx = numSeq - 1;
        double[][][] trainSeq = java.util.Arrays.copyOfRange(lastStepSeq, 0, splitIdx);
        double[][][] testSeq = java.util.Arrays.copyOfRange(lastStepSeq, splitIdx, numSeq);
        double[][][] trainLabel = java.util.Arrays.copyOfRange(labelSeq, 0, splitIdx);
        double[][][] testLabel = java.util.Arrays.copyOfRange(labelSeq, splitIdx, numSeq);
        if (trainSeq.length == 0 || trainLabel.length == 0) {
            logger.error("Jeu d'entraînement vide, impossible d'entraîner le modèle");
            throw new IllegalArgumentException("Jeu d'entraînement vide");
        }
        if (testSeq.length == 0 || testLabel.length == 0) {
            logger.error("Jeu de test vide, impossible d'évaluer le modèle");
            throw new IllegalArgumentException("Jeu de test vide");
        }
        org.nd4j.linalg.api.ndarray.INDArray trainInput = toINDArray(trainSeq); // [minibatch, numFeatures, 1]
        org.nd4j.linalg.api.ndarray.INDArray trainOutput = org.nd4j.linalg.factory.Nd4j.create(trainLabel); // [minibatch, 1, 1]
        org.nd4j.linalg.api.ndarray.INDArray testInput = toINDArray(testSeq);
        org.nd4j.linalg.api.ndarray.INDArray testOutput = org.nd4j.linalg.factory.Nd4j.create(testLabel); // [minibatch, 1, 1]
        // Ajout log debug shapeslogger.info("Shape trainInput: {} | trainOutput: {} | testInput: {} | testOutput: {}", java.util.Arrays.toString(trainInput.shape()), java.util.Arrays.toString(trainOutput.shape()), java.util.Arrays.toString(testInput.shape()), java.util.Arrays.toString(testOutput.shape()));
        if (containsNaN(trainOutput)) {
            logger.error("TrainOutput contient des NaN, impossible d'entraîner le modèle");
            throw new IllegalArgumentException("TrainOutput contient des NaN");
        }
        if (containsNaN(testOutput)) {
            logger.error("TestOutput contient des NaN, impossible d'évaluer le modèle");
            throw new IllegalArgumentException("TestOutput contient des NaN");
        }
        org.nd4j.linalg.dataset.api.iterator.DataSetIterator trainIterator = new ListDataSetIterator<>(
            java.util.Collections.singletonList(new org.nd4j.linalg.dataset.DataSet(trainInput, trainOutput))
        );
        double bestScore = Double.MAX_VALUE;
        int epochsWithoutImprovement = 0;
        int actualEpochs = 0;
        for (int i = 0; i < config.getNumEpochs(); i++) {
            model.fit(trainIterator);
            actualEpochs++;
            org.nd4j.linalg.api.ndarray.INDArray predictions = model.output(testInput);
            if (containsNaN(predictions)) {
                logger.error("Prédictions contiennent des NaN à l'epoch {}", i + 1);
                break;
            }
            double mse = Double.POSITIVE_INFINITY;
            try {
                mse = predictions.squaredDistance(testOutput) / testOutput.length();
            } catch (Exception e) {
                logger.error("Erreur lors du calcul du MSE à l'epoch {} : {}", i + 1, e.getMessage());
            }
            if (Double.isInfinite(mse) || Double.isNaN(mse)) {
                logger.error("MSE infini ou NaN à l'epoch {}", i + 1);
                break;
            }
            logger.info("Epoch {} terminé, Test MSE : {}", i + 1, mse);
            if (bestScore - mse > config.getMinDelta()) {
                bestScore = mse;
                epochsWithoutImprovement = 0;
            } else {
                epochsWithoutImprovement++;
                if (epochsWithoutImprovement >= config.getPatience()) {
                    logger.info("Early stopping déclenché à l'epoch {}. Meilleur Test MSE : {}", i + 1, bestScore);
                    break;
                }
            }
        }
        logger.info("Entraînement terminé après {} epochs. Meilleur Test MSE : {}", actualEpochs, bestScore);
        return model;
    }

    /**
     * Effectue une validation croisée k-fold sur le modèle LSTM.
     * Logge le score MSE, RMSE et MAE moyen et l'écart-type sur les k folds.
     * @param series série de bougies
     */
    public void crossValidateLstm(BarSeries series, LstmConfig config) {
        int windowSize = config.getWindowSize();
        double[] closes = extractCloseValues(series);
        double min, max;
        if ("global".equalsIgnoreCase(config.getNormalizationScope())) {
            min = java.util.Arrays.stream(closes).min().orElse(0.0);
            max = java.util.Arrays.stream(closes).max().orElse(0.0);
        } else {
            // min/max sur la dernière fenêtre
            min = Double.MAX_VALUE;
            max = -Double.MAX_VALUE;
            for (int i = closes.length - windowSize; i < closes.length; i++) {
                if (i >= 0) {
                    if (closes[i] < min) min = closes[i];
                    if (closes[i] > max) max = closes[i];
                }
            }
        }
        double[] normalized = normalizeByConfig(closes, min, max, config);
        double[][][] sequences = createSequences(normalized, windowSize);
        // Correction : labels rank 3 [batch, 1, 1]
        double[][][] labelSeq = new double[sequences.length][1][1];
        for (int i = 0; i < labelSeq.length; i++) {
            labelSeq[i][0][0] = normalized[i + windowSize];
        }
        int foldSize = sequences.length / config.getKFolds();
        double[] foldMSE = new double[config.getKFolds()];
        double[] foldRMSE = new double[config.getKFolds()];
        double[] foldMAE = new double[config.getKFolds()];
        for (int fold = 0; fold < config.getKFolds(); fold++) {
            int testStart = fold * foldSize;
            int testEnd = (fold == config.getKFolds() - 1) ? sequences.length : testStart + foldSize;
            double[][][] testSeq = java.util.Arrays.copyOfRange(sequences, testStart, testEnd);
            double[][][] testLabel = java.util.Arrays.copyOfRange(labelSeq, testStart, testEnd);
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
            org.nd4j.linalg.api.ndarray.INDArray trainOutput = org.nd4j.linalg.factory.Nd4j.create(trainLabel); // [batch, 1, 1]
            org.nd4j.linalg.api.ndarray.INDArray testInput = toINDArray(testSeq);
            org.nd4j.linalg.api.ndarray.INDArray testOutput = org.nd4j.linalg.factory.Nd4j.create(testLabel); // [batch, 1, 1]
            org.nd4j.linalg.dataset.api.iterator.DataSetIterator trainIterator = new ListDataSetIterator<>(
                java.util.Collections.singletonList(new org.nd4j.linalg.dataset.DataSet(trainInput, trainOutput))
            );
            MultiLayerNetwork model = initModel(windowSize, 1, config.getLstmNeurons(), config.getDropoutRate(), config.getLearningRate(), config.getOptimizer(), config.getL1(), config.getL2());
            double bestMSE = Double.MAX_VALUE;
            double bestRMSE = Double.MAX_VALUE;
            double bestMAE = Double.MAX_VALUE;
            int epochsWithoutImprovement = 0;
            for (int epoch = 0; epoch < config.getNumEpochs(); epoch++) {
                model.fit(trainIterator);
                org.nd4j.linalg.api.ndarray.INDArray predictions = model.output(testInput);
                double mse = predictions.squaredDistance(testOutput) / testOutput.length();
                double rmse = Math.sqrt(mse);
                double mae = 0.0;
                for (int i = 0; i < testOutput.length(); i++) {
                    mae += Math.abs(predictions.getDouble(i) - testOutput.getDouble(i));
                }
                mae /= testOutput.length();
                if (bestMSE - mse > config.getMinDelta()) {
                    bestMSE = mse;
                    bestRMSE = rmse;
                    bestMAE = mae;
                    epochsWithoutImprovement = 0;
                } else {
                    epochsWithoutImprovement++;
                    if (epochsWithoutImprovement >= config.getPatience()) {
                        logger.info("Early stopping fold {} à l'epoch {}. Meilleur Test MSE : {}, RMSE : {}, MAE : {}", fold + 1, epoch + 1, bestMSE, bestRMSE, bestMAE);
                        break;
                    }
                }
            }
            foldMSE[fold] = bestMSE;
            foldRMSE[fold] = bestRMSE;
            foldMAE[fold] = bestMAE;
            logger.info("Fold {} terminé. Meilleur Test MSE : {}, RMSE : {}, MAE : {}", fold + 1, bestMSE, bestRMSE, bestMAE);
        }
        // Calculer la moyenne et l'écart-type pour chaque métrique
        double meanMSE = 0.0, meanRMSE = 0.0, meanMAE = 0.0;
        for (int i = 0; i < config.getKFolds(); i++) {
            meanMSE += foldMSE[i];
            meanRMSE += foldRMSE[i];
            meanMAE += foldMAE[i];
        }
        meanMSE /= config.getKFolds();
        meanRMSE /= config.getKFolds();
        meanMAE /= config.getKFolds();
        double stdMSE = 0.0, stdRMSE = 0.0, stdMAE = 0.0;
        for (int i = 0; i < config.getKFolds(); i++) {
            stdMSE += Math.pow(foldMSE[i] - meanMSE, 2);
            stdRMSE += Math.pow(foldRMSE[i] - meanRMSE, 2);
            stdMAE += Math.pow(foldMAE[i] - meanMAE, 2);
        }
        stdMSE = Math.sqrt(stdMSE / config.getKFolds());
        stdRMSE = Math.sqrt(stdRMSE / config.getKFolds());
        stdMAE = Math.sqrt(stdMAE / config.getKFolds());
        logger.info("Validation croisée terminée. MSE moyen : {}, Ecart-type : {} | RMSE moyen : {}, Ecart-type : {} | MAE moyen : {}, Ecart-type : {}", meanMSE, stdMSE, meanRMSE, stdRMSE, meanMAE, stdMAE);
    }

    /**
     * Prédit la prochaine valeur de clôture à partir de la série et du windowSize.
     * @param series série de bougies
     * @return valeur de clôture prédite
     * @throws ModelNotFoundException si le modèle n'est pas initialisé
     * @throws InsufficientDataException si la série est trop courte
     */
    // Prédiction de la prochaine valeur de clôture
    public double predictNextClose(String symbol, BarSeries series, LstmConfig config, MultiLayerNetwork model) {
        java.util.List<String> features = config.getFeatures();
        int numFeatures = features.size();
        model = ensureModelWindowSize(model, numFeatures, config);
        double[][] matrix = extractFeatureMatrix(series, features);
        double[][] normMatrix = normalizeMatrix(matrix);
        int barCount = normMatrix.length;
        if (barCount < config.getWindowSize()) {
            throw new InsufficientDataException("Données insuffisantes pour la prédiction (windowSize=" + config.getWindowSize() + ", barCount=" + barCount + ").");
        }
        double[][] lastWindow = new double[config.getWindowSize()][numFeatures];
        for (int i = 0; i < config.getWindowSize(); i++) {
            for (int f = 0; f < numFeatures; f++) {
                lastWindow[i][f] = normMatrix[barCount - config.getWindowSize() + i][f];
            }
        }
        double[][][] lastSequence = new double[1][config.getWindowSize()][numFeatures];
        for (int j = 0; j < config.getWindowSize(); j++) {
            for (int f = 0; f < numFeatures; f++) {
                lastSequence[0][j][f] = lastWindow[j][f];
            }
        }
        // Correction : transposer la séquence pour obtenir [1, numFeatures, windowSize]
        double[][][] transposed = transposeSequencesMulti(lastSequence);
        // Correction : extraire la dernière étape pour chaque feature [1, numFeatures, 1]
        double[][][] inputSeq = new double[1][numFeatures][1];
        for (int f = 0; f < numFeatures; f++) {
            inputSeq[0][f][0] = transposed[0][f][config.getWindowSize() - 1];
        }
        org.nd4j.linalg.api.ndarray.INDArray input = toINDArray(inputSeq);
        org.nd4j.linalg.api.ndarray.INDArray output = model.output(input);
        double predictedNorm = output.getDouble(0);
        // Dénormalisation sur la clôture uniquement
        double[] closes = extractCloseValues(series);
        double[] lastCloseWindow = new double[config.getWindowSize()];
        System.arraycopy(closes, closes.length - config.getWindowSize(), lastCloseWindow, 0, config.getWindowSize());
        double min = "global".equalsIgnoreCase(config.getNormalizationScope()) ? java.util.Arrays.stream(closes).min().orElse(0.0) : java.util.Arrays.stream(lastCloseWindow).min().orElse(0.0);
        double max = "global".equalsIgnoreCase(config.getNormalizationScope()) ? java.util.Arrays.stream(closes).max().orElse(0.0) : java.util.Arrays.stream(lastCloseWindow).max().orElse(0.0);
        double predicted;
        if ("zscore".equalsIgnoreCase(config.getNormalizationMethod())) {
            double mean = java.util.Arrays.stream(lastCloseWindow).average().orElse(0.0);
            double std = Math.sqrt(java.util.Arrays.stream(lastCloseWindow).map(v -> (v - mean) * (v - mean)).average().orElse(0.0));
            predicted = predictedNorm * std + mean;
        } else {
            predicted = predictedNorm * (max - min) + min;
        }
        String position = analyzePredictionPosition(lastCloseWindow, predicted);
        if (predicted < min || predicted > max) {
            if (series.getBarCount() > 0) {
                logger.warn("[LSTM WARNING] La prédiction ({}) sort de la plage locale [{}, {}] pour le symbole {}.", predicted, min, max, symbol);
            } else {
                logger.warn("[LSTM WARNING] La série est vide, impossible d'obtenir le symbole.");
            }
        }
        logger.info("[PREDICT-NORM] windowSize={}, lastWindow={}, min={}, max={}, predictedNorm={}, predicted={}, normalizationScope={}, position={}",
            config.getWindowSize(), java.util.Arrays.toString(lastCloseWindow), min, max, predictedNorm, predicted, config.getNormalizationScope(), position);
        return predicted;
    }

    /**
     * Analyse la position de la prédiction par rapport à la tendance de la fenêtre.
     * @param window tableau des clôtures de la fenêtre
     * @param predicted valeur prédite
     * @return description de la position (cassure, rebond, range, etc.)
     */
    public String analyzePredictionPosition(double[] window, double predicted) {
        double lastClose = window[window.length - 1];
        double firstClose = window[0];
        double meanWindow = java.util.Arrays.stream(window).average().orElse(0.0);
        double min = java.util.Arrays.stream(window).min().orElse(0.0);
        double max = java.util.Arrays.stream(window).max().orElse(0.0);
        double trend = lastClose - firstClose;
        double range = max - min;
        // Surachat/survente simple
        boolean isOverbought = predicted > max * 1.01;
        boolean isOversold = predicted < min * 0.99;
        if (trend > 0 && predicted > max) {
            return "Cassure haussière";
        } else if (trend < 0 && predicted < min) {
            return "Cassure baissière";
        } else if (trend > 0 && predicted < meanWindow) {
            return "Rebond baissier";
        } else if (trend < 0 && predicted > meanWindow) {
            return "Rebond haussier";
        } else if (range < meanWindow * 0.05) {
            return "Range / faible volatilité";
        } else if (isOverbought) {
            return "Surachat (overbought)";
        } else if (isOversold) {
            return "Survente (oversold)";
        } else if (Math.abs(predicted - meanWindow) < range * 0.1) {
            return "Retour à la moyenne";
        } else {
            return "Continuation ou indéterminé";
        }
    }


    /**
     * Calcule automatiquement un threshold adapté pour le swing trade.
     * Utilise la volatilité moyenne ou un pourcentage du prix moyen.
     * @param series Série de bougies
     * @return threshold adapté
     */
    public double computeSwingTradeThreshold(BarSeries series) {
        double[] closes = extractCloseValues(series);
        if (closes.length < 2) return 0.0;
        double avgPrice = java.util.Arrays.stream(closes).average().orElse(0.0);
        double volatility = 0.0;
        for (int i = 1; i < closes.length; i++) {
            volatility += Math.abs(closes[i] - closes[i-1]);
        }
        volatility /= (closes.length - 1);
        // On prend le max entre 1% du prix moyen et la volatilité moyenne
        double threshold = Math.max(avgPrice * 0.01, volatility);
        return threshold;
    }

    /**
     * Prédit la classe (hausse/baisse/stable) pour la prochaine clôture, avec threshold automatique pour swing trade.
     * @param series Série de bougies
     * @return "up" si hausse, "down" si baisse, "stable" sinon
     */
    public PreditLsdm getPredit(String symbol, BarSeries series, LstmConfig config, MultiLayerNetwork model) {
        model = ensureModelWindowSize(model, config.getWindowSize(), config);
        double th = computeSwingTradeThreshold(series);
        double predicted = predictNextClose(symbol, series, config, model);
        predicted = Math.round(predicted * 1000.0) / 1000.0;
        double[] closes = extractCloseValues(series);
        double lastClose = closes[closes.length - 1];
        double[] lastWindow = new double[config.getWindowSize()];
        System.arraycopy(closes, closes.length - config.getWindowSize(), lastWindow, 0, config.getWindowSize());
        String position = analyzePredictionPosition(lastWindow, predicted);
        double delta =  predicted - lastClose;
        SignalType signal;
        if (delta > th) {
            signal = SignalType.BUY;
        } else if (delta < -th) {
            signal = SignalType.SELL;
        } else {
            signal = SignalType.HOLD;
        }
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd/MM");
        String formattedDate = series.getLastBar().getEndTime().format(formatter);
        logger.info("------------PREDICT {} | lastClose={}, predictedClose={}, delta={}, threshold={}, signal={}, position={}",
            config.getWindowSize(), lastClose, predicted, delta, th, signal, position);
        return PreditLsdm.builder()
                .lastClose(lastClose)
                .predictedClose(predicted)
                .signal(signal)
                .lastDate(formattedDate)
                .position(position)
                .build();
    }


    /**
     * Sauvegarde le modèle LSTM en base MySQL.
     * @param symbol symbole du modèle
     * @param jdbcTemplate template JDBC Spring
     * @throws IOException en cas d'erreur d'accès à la base
     */
    // Sauvegarde du modèle dans MySQL
    public void saveModelToDb(String symbol, MultiLayerNetwork model, JdbcTemplate jdbcTemplate, LstmConfig config) throws IOException {
        if (model != null) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ModelSerializer.writeModel(model, baos, true);
            byte[] modelBytes = baos.toByteArray();
            // Sauvegarde des hyperparamètres dans la table dédiée
            hyperparamsRepository.saveHyperparams(symbol, config);
            // Création des hyperparamètres pour la sérialisation JSON
            LstmHyperparameters params = new LstmHyperparameters(
                config.getWindowSize(),
                config.getLstmNeurons(),
                config.getDropoutRate(),
                config.getLearningRate(),
                config.getNumEpochs(),
                config.getPatience(),
                config.getMinDelta(),
                config.getOptimizer(),
                config.getL1(),
                config.getL2(),
                config.getNormalizationScope(),
                config.getNormalizationMethod()
            );
            // Sérialisation JSON
            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            String hyperparamsJson = mapper.writeValueAsString(params);
            String sql = "REPLACE INTO lstm_models (symbol, model_blob, hyperparams_json, normalization_scope, updated_date) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)";
            try {
                jdbcTemplate.update(sql, symbol, modelBytes, hyperparamsJson, config.getNormalizationScope());
                logger.info("Modèle et hyperparamètres sauvegardés en base pour le symbole : {} (scope={})", symbol, config.getNormalizationScope());
            } catch (Exception e) {
                logger.error("Erreur lors de la sauvegarde du modèle en base : {}", e.getMessage());
                throw e;
            }
        }
    }

    /**
     * Charge le modèle LSTM depuis la base MySQL.
     * @param symbol symbole du modèle
     * @param jdbcTemplate template JDBC Spring
     * @throws IOException en cas d'erreur d'accès à la base
     * @throws SQLException jamais levée (pour compatibilité)
     */
    // Chargement du modèle depuis MySQL
    public MultiLayerNetwork loadModelFromDb(String symbol, JdbcTemplate jdbcTemplate) throws IOException {
        LstmConfig config = hyperparamsRepository.loadHyperparams(symbol);
        if (config == null) {
            throw new IOException("Aucun hyperparamètre trouvé pour le symbole " + symbol);
        }
        String sql = "SELECT model_blob, normalization_scope, hyperparams_json FROM lstm_models WHERE symbol = ?";
        MultiLayerNetwork model = null;
        try {
            java.util.Map<String, Object> result = jdbcTemplate.queryForMap(sql, symbol);
            byte[] modelBytes = (byte[]) result.get("model_blob");
            String normalizationScope = (String) result.get("normalization_scope");
            String hyperparamsJson = result.containsKey("hyperparams_json") ? (String) result.get("hyperparams_json") : null;
            boolean normalizationSet = false;
            if (normalizationScope != null && !normalizationScope.isEmpty()) {
                config.setNormalizationScope(normalizationScope);
                normalizationSet = true;
            }
            // Fallback : si normalization_scope absent, essayer de lire du JSON
            if (!normalizationSet && hyperparamsJson != null && !hyperparamsJson.isEmpty()) {
                try {
                    com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                    java.util.Map<?,?> jsonMap = mapper.readValue(hyperparamsJson, java.util.Map.class);
                    Object normScopeObj = jsonMap.get("normalizationScope");
                    if (normScopeObj != null && normScopeObj instanceof String && !((String)normScopeObj).isEmpty()) {
                        config.setNormalizationScope((String)normScopeObj);
                        normalizationSet = true;
                    }
                } catch (Exception e) {
                    logger.warn("Impossible de parser hyperparams_json pour normalizationScope : {}", e.getMessage());
                }
            }
            if (modelBytes != null) {
                ByteArrayInputStream bais = new ByteArrayInputStream(modelBytes);
                model = ModelSerializer.restoreMultiLayerNetwork(bais);
                logger.info("Modèle chargé depuis la base pour le symbole : {} (scope={})", symbol, config.getNormalizationScope());
            }
        } catch (EmptyResultDataAccessException e) {
            logger.error("Modèle non trouvé en base pour le symbole : {}", symbol);
            throw new IOException("Modèle non trouvé en base");
        }
        return model;
    }

    /**
     * Entraîne le modèle avec les paramètres du fichier de configuration.
     * Permet d'activer ou désactiver la validation croisée k-fold via le paramètre crossValidation.
     * @param series série de bougies
     * @param crossValidation true pour activer la validation croisée, false pour entraînement classique
     */
    public void trainWithConfig(BarSeries series, LstmConfig config, boolean crossValidation) {
        if (crossValidation) {
            logger.info("Entraînement LSTM avec validation croisée ({} folds)", config.getKFolds());
            crossValidateLstm(series, config);
        } else {
            logger.info("Entraînement LSTM classique (train/test split 80/20)");
            trainLstm(series, config, null);
        }
    }

    /**
     * Vérifie si un INDArray contient au moins un NaN.
     */
    public boolean containsNaN(org.nd4j.linalg.api.ndarray.INDArray array) {
        for (int i = 0; i < array.length(); i++) {
            if (Double.isNaN(array.getDouble(i))) return true;
        }
        return false;
    }

    /**
     * Normalise un tableau de valeurs selon la méthode choisie dans la config (MinMax, z-score, etc.).
     * @param values tableau de valeurs à normaliser
     * @param min valeur minimale (pour MinMax)
     * @param max valeur maximale (pour MinMax)
     * @param config configuration LSTM
     * @return tableau normalisé
     */
    public double[] normalizeByConfig(double[] values, double min, double max, LstmConfig config) {
        String method = config.getNormalizationMethod();
        if ("zscore".equalsIgnoreCase(method)) {
            double mean = java.util.Arrays.stream(values).average().orElse(0.0);
            double std = Math.sqrt(java.util.Arrays.stream(values).map(v -> (v - mean) * (v - mean)).average().orElse(0.0));
            double[] normalized = new double[values.length];
            if (std == 0.0) {
                for (int i = 0; i < values.length; i++) normalized[i] = 0.0;
            } else {
                for (int i = 0; i < values.length; i++) normalized[i] = (values[i] - mean) / std;
            }
            return normalized;
        } else {
            // MinMax par défaut
            double[] normalized = new double[values.length];
            if (min == max) {
                for (int i = 0; i < values.length; i++) normalized[i] = 0.5;
            } else {
                for (int i = 0; i < values.length; i++) normalized[i] = (values[i] - min) / (max - min);
            }
            return normalized;
        }
    }

    /**
     * Extrait une matrice de features (close, volume, indicateurs techniques) d'une série de bougies.
     * @param series série de bougies TA4J
     * @param features liste des features à inclure (ex: "close", "volume", "rsi", "macd", "sma", "ema")
     * @return matrice [barCount][numFeatures]
     */
    public double[][] extractFeatureMatrix(BarSeries series, java.util.List<String> features) {
        int barCount = series.getBarCount();
        int numFeatures = features.size();
        double[][] matrix = new double[barCount][numFeatures];
        for (int i = 0; i < barCount; i++) {
            int f = 0;
            for (String feat : features) {
                switch (feat.toLowerCase()) {
                    case "close":
                        matrix[i][f] = series.getBar(i).getClosePrice().doubleValue();
                        break;
                    case "volume":
                        matrix[i][f] = series.getBar(i).getVolume().doubleValue();
                        break;
                    case "open":
                        matrix[i][f] = series.getBar(i).getOpenPrice().doubleValue();
                        break;
                    case "high":
                        matrix[i][f] = series.getBar(i).getHighPrice().doubleValue();
                        break;
                    case "low":
                        matrix[i][f] = series.getBar(i).getLowPrice().doubleValue();
                        break;
                    case "sma":
                        org.ta4j.core.indicators.SMAIndicator sma = new org.ta4j.core.indicators.SMAIndicator(new org.ta4j.core.indicators.helpers.ClosePriceIndicator(series), 14);
                        matrix[i][f] = i >= 13 ? sma.getValue(i).doubleValue() : 0.0;
                        break;
                    case "ema":
                        org.ta4j.core.indicators.EMAIndicator ema = new org.ta4j.core.indicators.EMAIndicator(new org.ta4j.core.indicators.helpers.ClosePriceIndicator(series), 14);
                        matrix[i][f] = i >= 13 ? ema.getValue(i).doubleValue() : 0.0;
                        break;
                    case "rsi":
                        org.ta4j.core.indicators.RSIIndicator rsi = new org.ta4j.core.indicators.RSIIndicator(new org.ta4j.core.indicators.helpers.ClosePriceIndicator(series), 14);
                        matrix[i][f] = i >= 13 ? rsi.getValue(i).doubleValue() : 0.0;
                        break;
                    case "macd":
                        org.ta4j.core.indicators.MACDIndicator macd = new org.ta4j.core.indicators.MACDIndicator(new org.ta4j.core.indicators.helpers.ClosePriceIndicator(series), 12, 26);
                        matrix[i][f] = i >= 25 ? macd.getValue(i).doubleValue() : 0.0;
                        break;
                    default:
                        matrix[i][f] = 0.0;
                }
                f++;
            }
        }
        return matrix;
    }

    /**
     * Normalise une matrice de features (MinMax par colonne).
     * @param matrix matrice [barCount][numFeatures]
     * @return matrice normalisée
     */
    public double[][] normalizeMatrix(double[][] matrix) {
        int barCount = matrix.length;
        int numFeatures = matrix[0].length;
        double[][] norm = new double[barCount][numFeatures];
        for (int f = 0; f < numFeatures; f++) {
            double min = Double.MAX_VALUE;
            double max = -Double.MAX_VALUE;
            for (int i = 0; i < barCount; i++) {
                if (matrix[i][f] < min) min = matrix[i][f];
                if (matrix[i][f] > max) max = matrix[i][f];
            }
            for (int i = 0; i < barCount; i++) {
                norm[i][f] = (min == max) ? 0.5 : (matrix[i][f] - min) / (max - min);
            }
        }
        return norm;
    }

    /**
     * Crée les séquences d'entrée pour le LSTM à partir d'une matrice de features normalisées.
     * @param matrix matrice [barCount][numFeatures]
     * @param windowSize taille de la fenêtre
     * @return séquences 3D [numSeq][windowSize][numFeatures]
     */
    public double[][][] createSequencesMulti(double[][] matrix, int windowSize) {
        int numSeq = matrix.length - windowSize;
        int numFeatures = matrix[0].length;
        double[][][] sequences = new double[numSeq][windowSize][numFeatures];
        for (int i = 0; i < numSeq; i++) {
            for (int j = 0; j < windowSize; j++) {
                for (int f = 0; f < numFeatures; f++) {
                    sequences[i][j][f] = matrix[i + j][f];
                }
            }
        }
        return sequences;
    }

    /**
     * Transpose les séquences multi-features pour DL4J : [numSeq][windowSize][numFeatures] -> [numSeq][numFeatures][windowSize]
     */
    public double[][][] transposeSequencesMulti(double[][][] sequences) {
        int numSeq = sequences.length;
        int windowSize = sequences[0].length;
        int numFeatures = sequences[0][0].length;
        double[][][] transposed = new double[numSeq][numFeatures][windowSize];
        for (int i = 0; i < numSeq; i++) {
            for (int j = 0; j < windowSize; j++) {
                for (int f = 0; f < numFeatures; f++) {
                    transposed[i][f][j] = sequences[i][j][f];
                }
            }
        }
        return transposed;
    }
}
