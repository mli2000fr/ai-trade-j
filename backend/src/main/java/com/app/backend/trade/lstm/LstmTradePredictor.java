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
     * Vérifie et réinitialise le modèle si le windowSize demandé est différent du modèle courant.
     */
    private MultiLayerNetwork ensureModelWindowSize(MultiLayerNetwork model, int windowSize, LstmConfig config) {
        // On ne vérifie plus currentWindowSize, on vérifie uniquement le modèle
        if (model == null) {
            logger.info("Réinitialisation du modèle LSTM : windowSize demandé = {}", windowSize);
            return initModel(
                windowSize,
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
    public org.nd4j.linalg.api.ndarray.INDArray prepareLstmInput(BarSeries series, int windowSize) {
        double[] closes = extractCloseValues(series);
        double[] normalized = normalize(closes);
        double[][][] sequences = createSequences(normalized, windowSize);
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
        model = ensureModelWindowSize(model, config.getWindowSize(), config);
        double[] closes = extractCloseValues(series);
        double minTrain, maxTrain;
        if ("global".equalsIgnoreCase(config.getNormalizationScope())) {
            minTrain = java.util.Arrays.stream(closes).min().orElse(0.0);
            maxTrain = java.util.Arrays.stream(closes).max().orElse(0.0);
        } else {
            minTrain = Double.MAX_VALUE;
            maxTrain = -Double.MAX_VALUE;
            for (int i = closes.length - config.getWindowSize(); i < closes.length; i++) {
                if (i >= 0) {
                    if (closes[i] < minTrain) minTrain = closes[i];
                    if (closes[i] > maxTrain) maxTrain = closes[i];
                }
            }
        }
        double[] normalized = normalize(closes, minTrain, maxTrain);
        int numSeq = normalized.length - config.getWindowSize();
        if (numSeq <= 0) {
            logger.error("Pas assez de données pour entraîner le modèle (windowSize={}, closes={})", config.getWindowSize(), closes.length);
            throw new IllegalArgumentException("Pas assez de données pour entraîner le modèle");
        }
        double[][][] sequences = createSequences(normalized, config.getWindowSize());
        double[][][] labelSeq = new double[numSeq][1][1];
        for (int i = 0; i < numSeq; i++) {
            labelSeq[i][0][0] = normalized[i + config.getWindowSize()];
        }
        int splitIdx = (int)(numSeq * 0.8);
        if (splitIdx == numSeq) splitIdx = numSeq - 1;
        double[][][] trainSeq = java.util.Arrays.copyOfRange(sequences, 0, splitIdx);
        double[][][] testSeq = java.util.Arrays.copyOfRange(sequences, splitIdx, numSeq);
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
        double[] normalized = normalize(closes, min, max);
        double[][][] sequences = createSequences(normalized, windowSize);
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
            org.nd4j.linalg.api.ndarray.INDArray trainOutput = org.nd4j.linalg.factory.Nd4j.create(trainLabel);
            org.nd4j.linalg.api.ndarray.INDArray testInput = toINDArray(testSeq);
            org.nd4j.linalg.api.ndarray.INDArray testOutput = org.nd4j.linalg.factory.Nd4j.create(testLabel);
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
    public double predictNextClose(BarSeries series, LstmConfig config, MultiLayerNetwork model) {
        model = ensureModelWindowSize(model, config.getWindowSize(), config);
        if (model == null) {
            throw new ModelNotFoundException("Le modèle LSTM n'est pas initialisé.");
        }
        double[] closes = extractCloseValues(series);
        if (closes.length < config.getWindowSize()) {
            throw new InsufficientDataException("Données insuffisantes pour la prédiction (windowSize=" + config.getWindowSize() + ", closes=" + closes.length + ").");
        }
        double[] lastWindow = new double[config.getWindowSize()];
        System.arraycopy(closes, closes.length - config.getWindowSize(), lastWindow, 0, config.getWindowSize());
        double min, max;
        if ("global".equalsIgnoreCase(config.getNormalizationScope())) {
            min = java.util.Arrays.stream(closes).min().orElse(0.0);
            max = java.util.Arrays.stream(closes).max().orElse(0.0);
        } else {
            min = java.util.Arrays.stream(lastWindow).min().orElse(0.0);
            max = java.util.Arrays.stream(lastWindow).max().orElse(0.0);
        }
        double[] normalized = normalize(lastWindow, min, max);
        double[][][] lastSequence = new double[1][config.getWindowSize()][1];
        for (int j = 0; j < config.getWindowSize(); j++) {
            lastSequence[0][j][0] = normalized[j];
        }
        org.nd4j.linalg.api.ndarray.INDArray input = toINDArray(lastSequence);
        org.nd4j.linalg.api.ndarray.INDArray output = model.output(input);
        double predictedNorm = output.getDouble(0);
        double predicted = predictedNorm * (max - min) + min;
        logger.info("[PREDICT-NORM] windowSize={}, lastWindow={}, min={}, max={}, predictedNorm={}, predicted={}, normalizationScope={}",
            config.getWindowSize(), java.util.Arrays.toString(lastWindow), min, max, predictedNorm, predicted, config.getNormalizationScope());
        return predicted;
    }

    /**
     * Prédit le delta (variation) entre la dernière clôture et la prédiction.
     * @param series série de bougies
     * @param windowSize taille de la fenêtre
     * @return variation prédite
     */
    // Prédiction du delta (variation) de la prochaine clôture
    public double predictNextDelta(BarSeries series, int windowSize, LstmConfig config, MultiLayerNetwork model) {
        model = ensureModelWindowSize(model, windowSize, config);
        double predicted = predictNextClose(series, config, model);
        double[] closes = extractCloseValues(series);
        double lastClose = closes[closes.length - 1];
        return predicted - lastClose;
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
    public PreditLsdm getPredit(BarSeries series, LstmConfig config, MultiLayerNetwork model) {
        model = ensureModelWindowSize(model, config.getWindowSize(), config);
        double th = computeSwingTradeThreshold(series);
        double predicted = predictNextClose(series, config, model);
        predicted = Math.round(predicted * 1000.0) / 1000.0;
        double[] closes = extractCloseValues(series);
        double lastClose = closes[closes.length - 1];
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
        // Ajout de logs détaillés pour analyse
        logger.info("------------PREDICT {} | lastClose={}, predictedClose={}, delta={}, threshold={}, signal={}",
            config.getWindowSize(), lastClose, predicted, delta, th, signal);
        return PreditLsdm.builder()
                .lastClose(lastClose)
                .predictedClose(predicted)
                .signal(signal)
                .lastDate(formattedDate)
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
                config.getNormalizationScope()
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
}
