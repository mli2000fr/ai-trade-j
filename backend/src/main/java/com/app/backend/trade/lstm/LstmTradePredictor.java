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
import org.ta4j.core.indicators.ROCIndicator;
import org.ta4j.core.indicators.statistics.StandardDeviationIndicator;
import org.ta4j.core.num.Num;

@Service
public class LstmTradePredictor {
    private static final Logger logger = LoggerFactory.getLogger(LstmTradePredictor.class);

    private final LstmHyperparamsRepository hyperparamsRepository;

    public LstmTradePredictor(LstmHyperparamsRepository hyperparamsRepository) {
        this.hyperparamsRepository = hyperparamsRepository;
    }

    /**
     * Initialise le modèle LSTM avec les hyperparamètres fournis.
     * @param inputSize nombre de features (nIn)
     * @param outputSize taille de la sortie (nOut)
     */
    public MultiLayerNetwork initModel(int inputSize, int outputSize, int lstmNeurons, double dropoutRate, double learningRate, String optimizer, double l1, double l2, LstmConfig config) {
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.updater(
            "adam".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.Adam(learningRate)
            : "rmsprop".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.RmsProp(learningRate)
            : new org.nd4j.linalg.learning.config.Sgd(learningRate)
        );
        builder.l1(l1);
        builder.l2(l2);

        org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder listBuilder = builder.list();
        int nLayers = config != null ? config.getNumLstmLayers() : 1;
        boolean bidir = config != null && config.isBidirectional();
        boolean attention = config != null && config.isAttention();

        // Couches LSTM empilées
        for (int i = 0; i < nLayers; i++) {
            LSTM.Builder lstmBuilder = new LSTM.Builder()
                .nIn(i == 0 ? inputSize : lstmNeurons)
                .nOut(lstmNeurons)
                .activation(Activation.TANH);
            if (bidir) {
                listBuilder.layer(new org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional(lstmBuilder.build()));
            } else {
                listBuilder.layer(lstmBuilder.build());
            }
            if (dropoutRate > 0.0 && (i < nLayers - 1)) {
                listBuilder.layer(new DropoutLayer.Builder().dropOut(dropoutRate).build());
            }
        }
        if (attention) {
            listBuilder.layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                .nIn(lstmNeurons)
                .nOut(lstmNeurons)
                .activation(Activation.SOFTMAX)
                .build());
        }
        listBuilder.layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
            .nIn(lstmNeurons)
            .nOut(Math.max(16, lstmNeurons / 4))
            .activation(Activation.RELU)
            .build());
        listBuilder.layer(new RnnOutputLayer.Builder()
            .nIn(Math.max(16, lstmNeurons / 4))
            .nOut(outputSize)
            .activation(Activation.IDENTITY)
            .lossFunction(LossFunctions.LossFunction.MSE)
            .build());
        MultiLayerConfiguration conf = listBuilder.build();
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
                numFeatures, // nIn = nombre de features
                1,
                config.getLstmNeurons(),
                config.getDropoutRate(),
                config.getLearningRate(),
                config.getOptimizer(),
                config.getL1(),
                config.getL2(),
                config
            );
        } else {
            logger.info("Modèle LSTM déjà initialisé");
            return model;
        }
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

    // Séquences univariées [numSeq][windowSize][1]
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

    /**
     * Entraîne le modèle LSTM avec séquences complètes [batch, numFeatures, windowSize].
     * Labels: next-step pour chaque time step de la fenêtre (séquence à séquence).
     * Retourne le modèle et le set de scalers appris sur le train.
     */
    public static class TrainResult {
        public MultiLayerNetwork model;
        public ScalerSet scalers;
        public TrainResult(MultiLayerNetwork model, ScalerSet scalers) {
            this.model = model;
            this.scalers = scalers;
        }
    }
    public TrainResult trainLstmWithScalers(BarSeries series, LstmConfig config, MultiLayerNetwork model) {
        java.util.List<String> features = config.getFeatures();
        int numFeatures = features.size();
        model = ensureModelWindowSize(model, numFeatures, config);

        double[][] matrix = extractFeatureMatrix(series, features);
        int windowSize = config.getWindowSize();
        int numSeq = matrix.length - windowSize;
        if (numSeq <= 0) {
            logger.error("Pas assez de données pour entraîner le modèle (windowSize={}, barCount={})", windowSize, matrix.length);
            throw new IllegalArgumentException("Pas assez de données pour entraîner le modèle");
        }
        // Split train/test
        int splitIdx = (int)(numSeq * 0.8);
        if (splitIdx == numSeq) splitIdx = numSeq - 1;
        // Apprentissage des scalers sur le train uniquement
        ScalerSet scalers = new ScalerSet();
        for (int f = 0; f < numFeatures; f++) {
            double[] col = new double[splitIdx + windowSize];
            for (int i = 0; i < splitIdx + windowSize; i++) col[i] = matrix[i][f];
            String normType = getFeatureNormalizationType(features.get(f));
            FeatureScaler.Type type = normType.equals("zscore") ? FeatureScaler.Type.ZSCORE : FeatureScaler.Type.MINMAX;
            FeatureScaler scaler = new FeatureScaler(type);
            scaler.fit(col);
            scalers.featureScalers.put(features.get(f), scaler);
        }
        // Label scaler (close)
        double[] closes = extractCloseValues(series);
        double[] labelTrain = new double[splitIdx + windowSize + 1];
        for (int i = 0; i < splitIdx + windowSize + 1; i++) labelTrain[i] = closes[i];
        FeatureScaler.Type labelType = config.getNormalizationMethod().equalsIgnoreCase("zscore") ? FeatureScaler.Type.ZSCORE : FeatureScaler.Type.MINMAX;
        FeatureScaler labelScaler = new FeatureScaler(labelType);
        labelScaler.fit(labelTrain);
        scalers.labelScaler = labelScaler;
        // Normalisation des features
        double[][] normMatrix = new double[matrix.length][numFeatures];
        for (int f = 0; f < numFeatures; f++) {
            double[] col = new double[matrix.length];
            for (int i = 0; i < matrix.length; i++) col[i] = matrix[i][f];
            normMatrix = applyScalerToMatrix(normMatrix, col, scalers.featureScalers.get(features.get(f)), f);
        }
        // Séquences
        double[][][] sequences = createSequencesMulti(normMatrix, windowSize);
        double[][][] sequencesTransposed = transposeSequencesMulti(sequences);
        // Labels normalisés
        double[] normCloses = scalers.labelScaler.transform(closes);
        double[][][] labelSeq = new double[numSeq][1][windowSize];
        for (int i = 0; i < numSeq; i++) {
            for (int t = 0; t < windowSize; t++) {
                labelSeq[i][0][t] = normCloses[i + t + 1];
            }
        }
        double[][][] trainSeq = java.util.Arrays.copyOfRange(sequencesTransposed, 0, splitIdx);
        double[][][] testSeq = java.util.Arrays.copyOfRange(sequencesTransposed, splitIdx, numSeq);
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
        org.nd4j.linalg.api.ndarray.INDArray trainInput = toINDArray(trainSeq);
        org.nd4j.linalg.api.ndarray.INDArray trainOutput = org.nd4j.linalg.factory.Nd4j.create(trainLabel);
        org.nd4j.linalg.api.ndarray.INDArray testInput = toINDArray(testSeq);
        org.nd4j.linalg.api.ndarray.INDArray testOutput = org.nd4j.linalg.factory.Nd4j.create(testLabel);
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
                mse = org.nd4j.linalg.ops.transforms.Transforms.pow(predictions.sub(testOutput), 2).meanNumber().doubleValue();
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
        return new TrainResult(model, scalers);
    }

    // Applique un scaler à une colonne du matrix
    private double[][] applyScalerToMatrix(double[][] normMatrix, double[] col, FeatureScaler scaler, int f) {
        double[] normCol = scaler.transform(col);
        for (int i = 0; i < normCol.length; i++) normMatrix[i][f] = normCol[i];
        return normMatrix;
    }

    /**
     * Validation croisée k-fold avec séquences complètes [batch, numFeatures, windowSize].
     */
    public double crossValidateLstm(BarSeries series, LstmConfig config) {
        int windowSize = config.getWindowSize();
        java.util.List<String> features = config.getFeatures();
        int numFeatures = features.size();

        // Entrées multi-features normalisées
        double[][] matrix = extractFeatureMatrix(series, features);
        double[][] normMatrix = normalizeMatrix(matrix, features);
        int numSeq = normMatrix.length - windowSize;
        if (numSeq <= 0) {
            logger.error("[CV] Pas assez de données (windowSize={}, barCount={})", windowSize, normMatrix.length);
            return Double.POSITIVE_INFINITY;
        }
        double[][][] sequences = createSequencesMulti(normMatrix, windowSize); // [numSeq, window, numFeatures]
        double[][][] sequencesTransposed = transposeSequencesMulti(sequences); // [numSeq, numFeatures, window]

        // Labels sur close normalisé (minmax global par défaut via normalize())
        double[] closes = extractCloseValues(series);
        double[] normCloses = normalize(closes);
        double[][][] labelSeq = new double[numSeq][1][windowSize];
        for (int i = 0; i < numSeq; i++) {
            for (int t = 0; t < windowSize; t++) {
                labelSeq[i][0][t] = normCloses[i + t + 1];
            }
        }

        int kFolds = config.getKFolds();
        int foldSize = sequencesTransposed.length / kFolds;
        double[] foldMSE = new double[kFolds];
        int validFolds = 0;
        for (int fold = 0; fold < kFolds; fold++) {
            int testStart = fold * foldSize;
            int testEnd = (fold == kFolds - 1) ? sequencesTransposed.length : testStart + foldSize;
            double[][][] testSeq = java.util.Arrays.copyOfRange(sequencesTransposed, testStart, testEnd);
            double[][][] testLabel = java.util.Arrays.copyOfRange(labelSeq, testStart, testEnd);
            double[][][] trainSeq = new double[sequencesTransposed.length - (testEnd - testStart)][numFeatures][windowSize];
            double[][][] trainLabel = new double[sequencesTransposed.length - (testEnd - testStart)][1][windowSize];
            int idx = 0;
            for (int i = 0; i < sequencesTransposed.length; i++) {
                if (i < testStart || i >= testEnd) {
                    trainSeq[idx] = sequencesTransposed[i];
                    trainLabel[idx] = labelSeq[i];
                    idx++;
                }
            }
            if (trainSeq.length == 0 || testSeq.length == 0) {
                logger.warn("[CV] Fold {} ignoré: train ou test vide", fold);
                foldMSE[fold] = Double.POSITIVE_INFINITY;
                continue;
            }
            org.nd4j.linalg.api.ndarray.INDArray trainInput = toINDArray(trainSeq);
            org.nd4j.linalg.api.ndarray.INDArray trainOutput = org.nd4j.linalg.factory.Nd4j.create(trainLabel);
            org.nd4j.linalg.api.ndarray.INDArray testInput = toINDArray(testSeq);
            org.nd4j.linalg.api.ndarray.INDArray testOutput = org.nd4j.linalg.factory.Nd4j.create(testLabel);
            if (containsNaN(trainOutput) || containsNaN(testOutput)) {
                logger.warn("[CV] Fold {} ignoré: NaN dans les labels", fold);
                foldMSE[fold] = Double.POSITIVE_INFINITY;
                continue;
            }

            MultiLayerNetwork model = initModel(
                numFeatures, // nIn = numFeatures (séquences complètes)
                1,
                config.getLstmNeurons(),
                config.getDropoutRate(),
                config.getLearningRate(),
                config.getOptimizer(),
                config.getL1(),
                config.getL2(),
                config
            );

            double bestMSE = Double.MAX_VALUE;
            int epochsWithoutImprovement = 0;
            for (int epoch = 0; epoch < config.getNumEpochs(); epoch++) {
                model.fit(new ListDataSetIterator<>(java.util.Collections.singletonList(new org.nd4j.linalg.dataset.DataSet(trainInput, trainOutput))));
                org.nd4j.linalg.api.ndarray.INDArray predictions = model.output(testInput);
                if (containsNaN(predictions)) {
                    logger.error("[CV] Fold {} epoch {}: NaN dans les prédictions", fold, epoch);
                    bestMSE = Double.POSITIVE_INFINITY;
                    break;
                }
                double mse;
                try {
                    mse = org.nd4j.linalg.ops.transforms.Transforms.pow(predictions.sub(testOutput), 2).meanNumber().doubleValue();
                } catch (Exception e) {
                    logger.error("[CV] Fold {} epoch {}: erreur MSE {}", fold, epoch, e.getMessage());
                    bestMSE = Double.POSITIVE_INFINITY;
                    break;
                }
                if (Double.isNaN(mse) || Double.isInfinite(mse)) {
                    logger.warn("[CV] Fold {} epoch {}: MSE NaN/Inf", fold, epoch);
                    bestMSE = Double.POSITIVE_INFINITY;
                    break;
                }
                if (bestMSE - mse > config.getMinDelta()) {
                    bestMSE = mse;
                    epochsWithoutImprovement = 0;
                } else {
                    epochsWithoutImprovement++;
                    if (epochsWithoutImprovement >= config.getPatience()) break;
                }
            }
            if (!Double.isNaN(bestMSE) && !Double.isInfinite(bestMSE)) {
                foldMSE[fold] = bestMSE;
                validFolds++;
            } else {
                foldMSE[fold] = Double.POSITIVE_INFINITY;
            }
        }
        if (validFolds == 0) return Double.POSITIVE_INFINITY;
        double mean = 0.0; int cnt = 0;
        for (double v : foldMSE) if (!Double.isNaN(v) && !Double.isInfinite(v)) { mean += v; cnt++; }
        return cnt == 0 ? Double.POSITIVE_INFINITY : mean / cnt;
    }

    /**
     * Validation croisée temporelle (Time Series Split)
     */
    public double crossValidateLstmTimeSeriesSplit(BarSeries series, LstmConfig config) {
        int windowSize = config.getWindowSize();
        int kFolds = config.getKFolds();
        double[] closes = extractCloseValues(series);
        int totalLength = closes.length;
        int foldSize = (totalLength - windowSize) / kFolds;
        if (foldSize < 1) {
            logger.error("Pas assez de données pour la validation croisée temporelle (windowSize={}, closes={})", windowSize, totalLength);
            return Double.POSITIVE_INFINITY;
        }
        double[] foldMSE = new double[kFolds];
        for (int fold = 0; fold < kFolds; fold++) {
            int testStart = windowSize + fold * foldSize;
            int testEnd = (fold == kFolds - 1) ? totalLength : testStart + foldSize;
            if (testEnd > totalLength) testEnd = totalLength;
            if (testStart - windowSize < windowSize) {
                foldMSE[fold] = Double.POSITIVE_INFINITY;
                continue;
            }
            BarSeries trainSeries = series.getSubSeries(0, testStart);
            BarSeries testSeries = series.getSubSeries(testStart - windowSize, testEnd);
            MultiLayerNetwork model = initModel(
                config.getFeatures().size(), // nIn = numFeatures
                1,
                config.getLstmNeurons(),
                config.getDropoutRate(),
                config.getLearningRate(),
                config.getOptimizer(),
                config.getL1(),
                config.getL2(),
                config
            );
            try {
                LstmTradePredictor.TrainResult trainResult = trainLstmWithScalers(trainSeries, config, model);
                model = trainResult.model;
                double mse = evaluateModel(model, testSeries, config);
                foldMSE[fold] = mse;
                logger.info("[TimeSeriesCV] Fold {} : MSE={}", fold, mse);
            } catch (Exception e) {
                logger.error("[TimeSeriesCV] Fold {} : erreur {}", fold, e.getMessage());
                foldMSE[fold] = Double.POSITIVE_INFINITY;
            }
        }
        double meanMSE = 0.0;
        int count = 0;
        for (int i = 0; i < kFolds; i++) {
            if (!Double.isNaN(foldMSE[i]) && !Double.isInfinite(foldMSE[i])) {
                meanMSE += foldMSE[i];
                count++;
            }
        }
        if (count == 0) {
            logger.error("Validation croisée temporelle impossible : aucun fold valide");
            return Double.POSITIVE_INFINITY;
        }
        meanMSE /= count;
        logger.info("Validation croisée temporelle terminée. MSE moyen ({} folds valides) : {}", count, meanMSE);
        return meanMSE;
    }

    // Évalue le modèle sur le jeu de test (MSE) avec séquences complètes
    private double evaluateModel(MultiLayerNetwork model, BarSeries series, LstmConfig config) {
        int numFeatures = config.getFeatures() != null ? config.getFeatures().size() : 1;
        int windowSize = config.getWindowSize();
        int numSeq;
        double[][][] testSeq;
        double[][][] testLabel;

        if (numFeatures > 1) {
            double[][] matrix = extractFeatureMatrix(series, config.getFeatures());
            double[][] normMatrix = normalizeMatrix(matrix, config.getFeatures());
            numSeq = normMatrix.length - windowSize;
            if (numSeq <= 0) {
                logger.warn("Pas assez de données pour évaluer le modèle (windowSize={}, barCount={})", windowSize, normMatrix.length);
                return Double.POSITIVE_INFINITY;
            }
            double[][][] sequences = createSequencesMulti(normMatrix, windowSize);
            double[][][] sequencesTransposed = transposeSequencesMulti(sequences); // [numSeq, numFeatures, window]

            double[] closes = extractCloseValues(series);
            double[] normCloses = normalize(closes);
            double[][][] labelSeq = new double[numSeq][1][windowSize];
            for (int i = 0; i < numSeq; i++) {
                for (int t = 0; t < windowSize; t++) {
                    labelSeq[i][0][t] = normCloses[i + t + 1];
                }
            }
            int splitIdx = (int)(numSeq * 0.8);
            if (splitIdx == numSeq) splitIdx = numSeq - 1;
            testSeq = java.util.Arrays.copyOfRange(sequencesTransposed, splitIdx, numSeq);
            testLabel = java.util.Arrays.copyOfRange(labelSeq, splitIdx, numSeq);
        } else {
            double[] closes = extractCloseValues(series);
            double min, max;
            if ("global".equalsIgnoreCase(config.getNormalizationScope())) {
                min = java.util.Arrays.stream(closes).min().orElse(0.0);
                max = java.util.Arrays.stream(closes).max().orElse(0.0);
            } else {
                min = Double.MAX_VALUE;
                max = -Double.MAX_VALUE;
                for (int i = closes.length - windowSize; i < closes.length; i++) {
                    if (i >= 0) {
                        if (closes[i] < min) min = closes[i];
                        if (closes[i] > max) max = closes[i];
                    }
                }
            }
            double[] normalized = normalize(closes, min, max);
            numSeq = normalized.length - windowSize;
            if (numSeq <= 0) {
                logger.warn("Pas assez de données pour évaluer le modèle (windowSize={}, closes={})", windowSize, closes.length);
                return Double.POSITIVE_INFINITY;
            }
            double[][][] sequences = createSequences(normalized, windowSize); // [numSeq, windowSize, 1]
            double[][][] sequencesTransposed = transposeSequencesMulti(sequences); // [numSeq, 1, windowSize]

            double[][][] labelSeq = new double[numSeq][1][windowSize];
            for (int i = 0; i < numSeq; i++) {
                for (int t = 0; t < windowSize; t++) {
                    labelSeq[i][0][t] = normalized[i + t + 1];
                }
            }
            int splitIdx = (int)(numSeq * 0.8);
            if (splitIdx == numSeq) splitIdx = numSeq - 1;
            testSeq = java.util.Arrays.copyOfRange(sequencesTransposed, splitIdx, numSeq);
            testLabel = java.util.Arrays.copyOfRange(labelSeq, splitIdx, numSeq);
        }
        if (testSeq.length == 0 || testLabel.length == 0) {
            logger.warn("Jeu de test vide pour l'évaluation du modèle");
            return Double.POSITIVE_INFINITY;
        }
        org.nd4j.linalg.api.ndarray.INDArray testInput = toINDArray(testSeq);
        org.nd4j.linalg.api.ndarray.INDArray testOutput = org.nd4j.linalg.factory.Nd4j.create(testLabel); // [batch, 1, windowSize]
        if (containsNaN(testOutput)) {
            logger.warn("TestOutput contient des NaN pour l'évaluation du modèle");
            return Double.POSITIVE_INFINITY;
        }
        org.nd4j.linalg.api.ndarray.INDArray predictions = model.output(testInput); // [batch, 1, windowSize]
        if (containsNaN(predictions)) {
            logger.warn("Prédictions contiennent des NaN pour l'évaluation du modèle");
            return Double.POSITIVE_INFINITY;
        }
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

    // Prédiction de la prochaine valeur de clôture
    public double predictNextClose(String symbol, BarSeries series, LstmConfig config, MultiLayerNetwork model) {
        java.util.List<String> features = config.getFeatures();
        int numFeatures = features.size();
        model = ensureModelWindowSize(model, numFeatures, config);

        double[][] matrix = extractFeatureMatrix(series, features);
        double[][] normMatrix = normalizeMatrix(matrix, features);
        int barCount = normMatrix.length;
        if (barCount < config.getWindowSize()) {
            throw new InsufficientDataException("Données insuffisantes pour la prédiction (windowSize=" + config.getWindowSize() + ", barCount=" + barCount + ").");
        }
        int windowSize = config.getWindowSize();
        double[][] lastWindow = new double[windowSize][numFeatures];
        for (int i = 0; i < windowSize; i++) {
            for (int f = 0; f < numFeatures; f++) {
                lastWindow[i][f] = normMatrix[barCount - windowSize + i][f];
            }
        }
        for (int i = 0; i < windowSize; i++) {
            for (int f = 0; f < numFeatures; f++) {
                if (Double.isNaN(lastWindow[i][f]) || lastWindow[i][f] == 0.0) {
                    logger.warn("[LSTM WARNING] Feature anormale dans la fenêtre : i={}, f={}, value={}", i, f, lastWindow[i][f]);
                }
            }
        }
        double[][][] lastSequence = new double[1][windowSize][numFeatures];
        for (int j = 0; j < windowSize; j++) {
            for (int f = 0; f < numFeatures; f++) {
                lastSequence[0][j][f] = lastWindow[j][f];
            }
        }
        double[][][] inputSeq = transposeSequencesMulti(lastSequence); // [1, numFeatures, windowSize]
        org.nd4j.linalg.api.ndarray.INDArray input = toINDArray(inputSeq);
        org.nd4j.linalg.api.ndarray.INDArray output = model.output(input); // [1,1,windowSize]
        double predictedNorm = output.getDouble(0, 0, windowSize - 1); // dernier pas de temps

        double[] closes = extractCloseValues(series);
        double[] lastCloseWindow = new double[windowSize];
        System.arraycopy(closes, closes.length - windowSize, lastCloseWindow, 0, windowSize);
        String normScope = config.getNormalizationScope();
        String normMethod = config.getNormalizationMethod();
        double min, max;
        if ("global".equalsIgnoreCase(normScope)) {
            min = java.util.Arrays.stream(closes).min().orElse(0.0);
            max = java.util.Arrays.stream(closes).max().orElse(0.0);
        } else {
            min = java.util.Arrays.stream(lastCloseWindow).min().orElse(0.0);
            max = java.util.Arrays.stream(lastCloseWindow).max().orElse(0.0);
        }
        double lastClose = closes[closes.length - 1];
        if (Math.abs(lastClose - min) / lastClose > 0.2 || Math.abs(lastClose - max) / lastClose > 0.2) {
            logger.warn("[LSTM WARNING] min/max trop éloignés du dernier close : min={}, max={}, lastClose={}", min, max, lastClose);
        }
        double predicted;
        if ("zscore".equalsIgnoreCase(normMethod)) {
            double mean = java.util.Arrays.stream(lastCloseWindow).average().orElse(0.0);
            double std = Math.sqrt(java.util.Arrays.stream(lastCloseWindow).map(v -> (v - mean) * (v - mean)).average().orElse(0.0));
            predicted = predictedNorm * std + mean;
        } else {
            predicted = predictedNorm * (max - min) + min;
        }
        logger.info("[LSTM DEBUG] predictedNorm={}, min={}, max={}, predicted={}, normScope={}, normMethod={}", predictedNorm, min, max, predicted, normScope, normMethod);
        // Si la prédiction est hors de la plage min/max, log d'alerte
        if (predicted < min || predicted > max) {
            logger.warn("[LSTM WARNING] Prédiction hors plage min/max : predicted={}, min={}, max={}", predicted, min, max);
        }
        // Limitation de la prédiction à ±10% autour du dernier close (sécurité, mais log si utilisé)
        double lowerBound = lastClose * 0.9;
        double upperBound = lastClose * 1.1;
        double predictedLimited = Math.max(lowerBound, Math.min(predicted, upperBound));
        if (predicted != predictedLimited) {
            logger.warn("[LSTM WARNING] Prédiction limitée : predicted={} -> limited=[{}, {}]", predicted, lowerBound, upperBound);
        }
        String position = analyzePredictionPosition(lastCloseWindow, predictedLimited);
        logger.info("[PREDICT-NORM] windowSize={}, lastWindow={}, min={}, max={}, predictedNorm={}, predicted={}, predictedLimited={}, normalizationScope={}, position={}",
            windowSize, java.util.Arrays.toString(lastCloseWindow), min, max, predictedNorm, predicted, predictedLimited, config.getNormalizationScope(), position);
        return predictedLimited;
    }

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

    public double computeSwingTradeThreshold(BarSeries series) {
        double[] closes = extractCloseValues(series);
        if (closes.length < 2) return 0.0;
        double avgPrice = java.util.Arrays.stream(closes).average().orElse(0.0);
        double volatility = 0.0;
        for (int i = 1; i < closes.length; i++) {
            volatility += Math.abs(closes[i] - closes[i-1]);
        }
        volatility /= (closes.length - 1);
        double threshold = Math.max(avgPrice * 0.01, volatility);
        return threshold;
    }

    public PreditLsdm getPredit(String symbol, BarSeries series, LstmConfig config, MultiLayerNetwork model) {
        // Toujours nIn = numFeatures
        model = ensureModelWindowSize(model, config.getFeatures().size(), config);
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
            signal = SignalType.UP;
        } else if (delta < -th) {
            signal = SignalType.DOWN;
        } else {
            signal = SignalType.STABLE;
        }
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd/MM");
        String formattedDate = series.getLastBar().getEndTime().format(formatter);
        logger.info("------------PREDICT {} | lastClose={}, predictedClose={}, delta={}, threshold={}, signal={}, position={}",
            config.getWindowSize(), lastClose, predicted, delta, th, signal, position);
        logPredictionAnomalies(symbol, lastClose, predicted, th);
        return PreditLsdm.builder()
                .lastClose(lastClose)
                .predictedClose(predicted)
                .signal(signal)
                .lastDate(formattedDate)
                .position(position)
                .build();
    }

    public PreditLsdm getPreditAtIndex(String symbol, BarSeries series, LstmConfig config, MultiLayerNetwork model, int index) {
        if (index < config.getWindowSize()) {
            return PreditLsdm.builder()
                    .lastClose(series.getBar(index).getClosePrice().doubleValue())
                    .predictedClose(series.getBar(index).getClosePrice().doubleValue())
                    .signal(SignalType.NONE)
                    .position("")
                    .lastDate(series.getBar(index).getEndTime().toString())
                    .build();
        }
        BarSeries subSeries = series.getSubSeries(index - config.getWindowSize(), index + 1);
        PreditLsdm pred = getPredit(symbol, subSeries, config, model);
        double th = computeSwingTradeThreshold(subSeries);
        logPredictionAnomalies(symbol, pred.getLastClose(), pred.getPredictedClose(), th);
        return pred;
    }

    /**
     * Sauvegarde le modèle LSTM en base MySQL.
     * @param symbol symbole du modèle
     * @param jdbcTemplate template JDBC Spring
     * @throws IOException en cas d'erreur d'accès à la base
     */
    // Sauvegarde du modèle dans MySQL
    public void saveModelToDb(String symbol, MultiLayerNetwork model, JdbcTemplate jdbcTemplate, LstmConfig config, ScalerSet scalers) throws IOException {
        if (model != null) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ModelSerializer.writeModel(model, baos, true);
            byte[] modelBytes = baos.toByteArray();
            hyperparamsRepository.saveHyperparams(symbol, config);
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
            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            String hyperparamsJson = mapper.writeValueAsString(params);
            String scalersJson = mapper.writeValueAsString(scalers);
            String sql = "REPLACE INTO lstm_models (symbol, model_blob, hyperparams_json, normalization_scope, scalers_json, updated_date) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)";
            try {
                jdbcTemplate.update(sql, symbol, modelBytes, hyperparamsJson, config.getNormalizationScope(), scalersJson);
                logger.info("Modèle, hyperparamètres et scalers sauvegardés en base pour le symbole : {} (scope={})", symbol, config.getNormalizationScope());
            } catch (Exception e) {
                logger.error("Erreur lors de la sauvegarde du modèle/scalers en base : {}", e.getMessage());
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

    public static class LoadedModel {
        public MultiLayerNetwork model;
        public LstmTradePredictor.ScalerSet scalers;
        public LoadedModel(MultiLayerNetwork model, LstmTradePredictor.ScalerSet scalers) {
            this.model = model;
            this.scalers = scalers;
        }
    }

    public LoadedModel loadModelAndScalersFromDb(String symbol, JdbcTemplate jdbcTemplate) throws IOException {
        LstmConfig config = hyperparamsRepository.loadHyperparams(symbol);
        if (config == null) {
            throw new IOException("Aucun hyperparamètre trouvé pour le symbole " + symbol);
        }
        String sql = "SELECT model_blob, normalization_scope, hyperparams_json, scalers_json FROM lstm_models WHERE symbol = ?";
        MultiLayerNetwork model = null;
        LstmTradePredictor.ScalerSet scalers = null;
        try {
            java.util.Map<String, Object> result = jdbcTemplate.queryForMap(sql, symbol);
            byte[] modelBytes = (byte[]) result.get("model_blob");
            String normalizationScope = (String) result.get("normalization_scope");
            String hyperparamsJson = result.containsKey("hyperparams_json") ? (String) result.get("hyperparams_json") : null;
            String scalersJson = result.containsKey("scalers_json") ? (String) result.get("scalers_json") : null;
            boolean normalizationSet = false;
            if (normalizationScope != null && !normalizationScope.isEmpty()) {
                config.setNormalizationScope(normalizationScope);
                normalizationSet = true;
            }
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
            if (scalersJson != null && !scalersJson.isEmpty()) {
                try {
                    com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                    scalers = mapper.readValue(scalersJson, LstmTradePredictor.ScalerSet.class);
                } catch (Exception e) {
                    logger.warn("Impossible de parser scalers_json : {}", e.getMessage());
                }
            }
        } catch (EmptyResultDataAccessException e) {
            logger.error("Modèle non trouvé en base pour le symbole : {}", symbol);
            throw new IOException("Modèle non trouvé en base");
        }
        return new LoadedModel(model, scalers);
    }

    public boolean containsNaN(org.nd4j.linalg.api.ndarray.INDArray array) {
        for (int i = 0; i < array.length(); i++) {
            if (Double.isNaN(array.getDouble(i))) return true;
        }
        return false;
    }

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
            double[] normalized = new double[values.length];
            if (min == max) {
                for (int i = 0; i < values.length; i++) normalized[i] = 0.5;
            } else {
                for (int i = 0; i < values.length; i++) normalized[i] = (values[i] - min) / (max - min);
            }
            return normalized;
        }
    }

    public String getFeatureNormalizationType(String feature) {
        String f = feature.toLowerCase();
        if (f.equals("rsi") || f.equals("stochastic") || f.equals("cci") || f.equals("momentum")) {
            return "zscore";
        } else if (f.equals("close") || f.equals("open") || f.equals("high") || f.equals("low") || f.equals("sma") || f.equals("ema") || f.equals("macd") || f.equals("atr") || f.startsWith("bollinger")) {
            return "minmax";
        } else if (f.equals("volume")) {
            return "minmax";
        } else if (f.equals("day_of_week") || f.equals("month") || f.equals("session")) {
            return "minmax";
        }
        return "minmax";
    }

    public double[] interpolateNaN(double[] col) {
        int n = col.length;
        double[] res = new double[n];
        System.arraycopy(col, 0, res, 0, n);
        double lastValid = Double.NaN;
        for (int i = 0; i < n; i++) {
            if (Double.isNaN(res[i])) {
                int j = i + 1;
                while (j < n && Double.isNaN(res[j])) j++;
                if (j < n) res[i] = res[j];
                else if (!Double.isNaN(lastValid)) res[i] = lastValid;
            } else {
                lastValid = res[i];
            }
        }
        for (int i = 0; i < n; i++) {
            if (Double.isNaN(res[i])) res[i] = 0.0;
        }
        return res;
    }

    public double[][] extractFeatureMatrix(BarSeries series, java.util.List<String> features) {
        int barCount = series.getBarCount();
        int numFeatures = features.size();
        double[][] matrix = new double[barCount][numFeatures];
        // Pré-calcul des indicateurs techniques pour éviter recalculs inutiles
        org.ta4j.core.indicators.helpers.ClosePriceIndicator closeIndicator = new org.ta4j.core.indicators.helpers.ClosePriceIndicator(series);
        org.ta4j.core.indicators.helpers.VolumeIndicator volumeIndicator = new org.ta4j.core.indicators.helpers.VolumeIndicator(series);
        org.ta4j.core.indicators.SMAIndicator sma = new org.ta4j.core.indicators.SMAIndicator(closeIndicator, 14);
        org.ta4j.core.indicators.EMAIndicator ema = new org.ta4j.core.indicators.EMAIndicator(closeIndicator, 14);
        org.ta4j.core.indicators.RSIIndicator rsi = new org.ta4j.core.indicators.RSIIndicator(closeIndicator, 14);
        org.ta4j.core.indicators.MACDIndicator macd = new org.ta4j.core.indicators.MACDIndicator(closeIndicator, 12, 26);
        org.ta4j.core.indicators.ATRIndicator atr = new org.ta4j.core.indicators.ATRIndicator(series, 14);
        org.ta4j.core.indicators.bollinger.BollingerBandsMiddleIndicator bbm = new org.ta4j.core.indicators.bollinger.BollingerBandsMiddleIndicator(sma);
        Num k = series.numOf(2.0);
        org.ta4j.core.indicators.bollinger.BollingerBandsUpperIndicator bbu = new org.ta4j.core.indicators.bollinger.BollingerBandsUpperIndicator(bbm, new StandardDeviationIndicator(closeIndicator, 14), k);
        org.ta4j.core.indicators.bollinger.BollingerBandsLowerIndicator bbl = new org.ta4j.core.indicators.bollinger.BollingerBandsLowerIndicator(bbm, new StandardDeviationIndicator(closeIndicator, 14), k);
        org.ta4j.core.indicators.StochasticOscillatorKIndicator stochastic = new org.ta4j.core.indicators.StochasticOscillatorKIndicator(series, 14);
        org.ta4j.core.indicators.CCIIndicator cci = new org.ta4j.core.indicators.CCIIndicator(series, 14);
        ROCIndicator momentum = new ROCIndicator(closeIndicator, 10);
        for (int i = 0; i < barCount; i++) {
            int f = 0;
            for (String feat : features) {
                switch (feat.toLowerCase()) {
                    case "close":
                        matrix[i][f] = closeIndicator.getValue(i).doubleValue();
                        break;
                    case "volume":
                        matrix[i][f] = volumeIndicator.getValue(i).doubleValue();
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
                        matrix[i][f] = i >= 13 ? sma.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "ema":
                        matrix[i][f] = i >= 13 ? ema.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "rsi":
                        matrix[i][f] = i >= 13 ? rsi.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "macd":
                        matrix[i][f] = i >= 25 ? macd.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "atr":
                        matrix[i][f] = i >= 13 ? atr.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "bollinger_high":
                        matrix[i][f] = i >= 13 ? bbu.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "bollinger_low":
                        matrix[i][f] = i >= 13 ? bbl.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "stochastic":
                        matrix[i][f] = i >= 13 ? stochastic.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "cci":
                        matrix[i][f] = i >= 13 ? cci.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "momentum":
                        matrix[i][f] = i >= 9 ? momentum.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "day_of_week":
                        matrix[i][f] = series.getBar(i).getEndTime().getDayOfWeek().getValue();
                        break;
                    case "month":
                        matrix[i][f] = series.getBar(i).getEndTime().getMonthValue();
                        break;
                    case "session":
                        int hour = series.getBar(i).getEndTime().getHour();
                        matrix[i][f] = hour < 8 ? 0 : (hour < 12 ? 1 : (hour < 17 ? 2 : 3));
                        break;
                    default:
                        matrix[i][f] = Double.NaN;
                }
                f++;
            }
        }
        for (int f = 0; f < numFeatures; f++) {
            double[] col = new double[barCount];
            for (int i = 0; i < barCount; i++) col[i] = matrix[i][f];
            col = interpolateNaN(col);
            for (int i = 0; i < barCount; i++) matrix[i][f] = col[i];
        }
        return matrix;
    }

    public double[] filterOutliers(double[] col) {
        int n = col.length;
        double[] sorted = java.util.Arrays.copyOf(col, n);
        java.util.Arrays.sort(sorted);
        double q1 = sorted[n / 4];
        double q3 = sorted[(3 * n) / 4];
        double iqr = q3 - q1;
        double lower = q1 - 1.5 * iqr;
        double upper = q3 + 1.5 * iqr;
        double[] filtered = new double[n];
        for (int i = 0; i < n; i++) {
            filtered[i] = Math.max(lower, Math.min(upper, col[i]));
        }
        return filtered;
    }

    private java.util.Map<String, double[]> driftStats = new java.util.HashMap<>();
    public void monitorDrift(double[] col, String featureName) {
        double mean = java.util.Arrays.stream(col).average().orElse(0.0);
        double std = Math.sqrt(java.util.Arrays.stream(col).map(v -> (v - mean) * (v - mean)).average().orElse(0.0));
        double[] lastStats = driftStats.get(featureName);
        if (lastStats != null) {
            double lastMean = lastStats[0];
            double lastStd = lastStats[1];
            if (Math.abs(mean - lastMean) / (Math.abs(lastMean) + 1e-8) > 0.2 || Math.abs(std - lastStd) / (Math.abs(lastStd) + 1e-8) > 0.2) {
                logger.warn("[DRIFT] Feature '{}' : drift détecté (mean {:.3f} -> {:.3f}, std {:.3f} -> {:.3f})", featureName, lastMean, mean, lastStd, std);
            }
        }
        driftStats.put(featureName, new double[]{mean, std});
    }

    public double[][] normalizeMatrix(double[][] matrix, java.util.List<String> features) {
        int barCount = matrix.length;
        int numFeatures = matrix[0].length;
        double[][] norm = new double[barCount][numFeatures];
        for (int f = 0; f < numFeatures; f++) {
            double[] col = new double[barCount];
            for (int i = 0; i < barCount; i++) col[i] = matrix[i][f];
            col = filterOutliers(col);
            monitorDrift(col, features.get(f));
            String normType = getFeatureNormalizationType(features.get(f));
            if (normType.equals("zscore")) {
                double mean = java.util.Arrays.stream(col).average().orElse(0.0);
                double std = Math.sqrt(java.util.Arrays.stream(col).map(v -> (v - mean) * (v - mean)).average().orElse(0.0));
                for (int i = 0; i < barCount; i++) norm[i][f] = std == 0.0 ? 0.0 : (col[i] - mean) / std;
            } else {
                double min = java.util.Arrays.stream(col).min().orElse(0.0);
                double max = java.util.Arrays.stream(col).max().orElse(0.0);
                for (int i = 0; i < barCount; i++) norm[i][f] = (min == max) ? 0.5 : (col[i] - min) / (max - min);
            }
        }
        return norm;
    }

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

    // Classe utilitaire pour la normalisation cohérente
    public static class FeatureScaler implements java.io.Serializable {
        public enum Type { MINMAX, ZSCORE }
        public Type type;
        public double min, max, mean, std;
        public FeatureScaler(Type type) { this.type = type; }
        public void fit(double[] values) {
            if (type == Type.MINMAX) {
                min = java.util.Arrays.stream(values).min().orElse(0.0);
                max = java.util.Arrays.stream(values).max().orElse(0.0);
            } else {
                mean = java.util.Arrays.stream(values).average().orElse(0.0);
                std = Math.sqrt(java.util.Arrays.stream(values).map(v -> (v - mean) * (v - mean)).average().orElse(0.0));
            }
        }
        public double[] transform(double[] values) {
            double[] res = new double[values.length];
            if (type == Type.MINMAX) {
                if (min == max) {
                    for (int i = 0; i < values.length; i++) res[i] = 0.5;
                } else {
                    for (int i = 0; i < values.length; i++) res[i] = (values[i] - min) / (max - min);
                }
            } else {
                if (std == 0.0) {
                    for (int i = 0; i < values.length; i++) res[i] = 0.0;
                } else {
                    for (int i = 0; i < values.length; i++) res[i] = (values[i] - mean) / std;
                }
            }
            return res;
        }
        public double inverse(double value) {
            if (type == Type.MINMAX) return value * (max - min) + min;
            else return value * std + mean;
        }
    }

    // Structure pour stocker les scalers de toutes les features + label
    public static class ScalerSet implements java.io.Serializable {
        public java.util.Map<String, FeatureScaler> featureScalers = new java.util.HashMap<>();
        public FeatureScaler labelScaler;
    }

    public void logPredictionAnomalies(String symbol, double lastClose, double predictedClose, double threshold) {
        double delta = Math.abs(predictedClose - lastClose);
        if (delta > 2 * threshold || delta / lastClose > 0.1) {
            logger.warn("[LSTM ANOMALIE] Prédiction très éloignée du close réel : symbol={}, lastClose={}, predictedClose={}, delta={}, threshold={}", symbol, lastClose, predictedClose, delta, threshold);
        }
    }
}
