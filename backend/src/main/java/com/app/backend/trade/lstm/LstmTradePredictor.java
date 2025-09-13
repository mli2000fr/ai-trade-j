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
import com.app.backend.trade.exception.InsufficientDataException;
import com.app.backend.trade.exception.ModelNotFoundException;
import com.app.backend.trade.exception.ModelPersistenceException;

@Service
public class LstmTradePredictor {
    private MultiLayerNetwork model;
    private static final Logger logger = LoggerFactory.getLogger(LstmTradePredictor.class);
    private LstmConfig config;
    private int currentWindowSize = -1;

    /**
     * Constructeur principal.
     * Initialise le modèle LSTM avec les paramètres du fichier de configuration.
     * @param config configuration des hyperparamètres LSTM
     */
    public LstmTradePredictor(LstmConfig config) {
        this.config = config;
        // Initialisation automatique du modèle avec les paramètres du fichier de config
        initModel(
            config.getWindowSize(),
            1,
            config.getLstmNeurons(),
            config.getDropoutRate(),
            config.getLearningRate(),
            config.getOptimizer()
        );
    }

    /**
     * Initialise le modèle LSTM avec les hyperparamètres fournis et mémorise le windowSize courant.
     * @param inputSize taille de l'entrée (doit être égal au windowSize utilisé)
     * @param outputSize taille de la sortie
     * @param lstmNeurons nombre de neurones dans la couche LSTM
     * @param dropoutRate taux de Dropout (ex : 0.2)
     * @param learningRate taux d'apprentissage
     * @param optimizer nom de l'optimiseur ("adam", "rmsprop", "sgd")
     */
    public void initModel(int inputSize, int outputSize, int lstmNeurons, double dropoutRate, double learningRate, String optimizer) {
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.updater(
            "adam".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.Adam(learningRate)
            : "rmsprop".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.RmsProp(learningRate)
            : new org.nd4j.linalg.learning.config.Sgd(learningRate)
        );
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
            .layer(new RnnOutputLayer.Builder()
                .nIn(lstmNeurons)
                .nOut(outputSize)
                .activation(Activation.IDENTITY)
                .lossFunction(LossFunctions.LossFunction.MSE)
                .build())
            .build();
        model = new MultiLayerNetwork(conf);
        model.init();
        currentWindowSize = inputSize;
    }

    /**
     * Vérifie et réinitialise le modèle si le windowSize demandé est différent du modèle courant.
     */
    private void ensureModelWindowSize(int windowSize) {
        if (model == null || currentWindowSize != windowSize) {
            logger.info("Réinitialisation du modèle LSTM : windowSize courant = {} | windowSize demandé = {}", currentWindowSize, windowSize);
            initModel(
                windowSize,
                1,
                config.getLstmNeurons(),
                config.getDropoutRate(),
                config.getLearningRate(),
                config.getOptimizer()
            );
        } else {
            logger.info("Modèle LSTM déjà initialisé avec windowSize = {}", currentWindowSize);
        }
    }

    // Ancienne version conservée pour compatibilité
    /**
     * @deprecated Utiliser initModel avec tous les paramètres
     * Initialise le modèle avec des paramètres par défaut.
     */
    @Deprecated
    public void initModel(int inputSize, int outputSize, int lstmNeurons, double dropoutRate) {
        initModel(inputSize, outputSize, lstmNeurons, dropoutRate, 0.001, "adam");
    }
    /**
     * @deprecated Utiliser initModel avec tous les paramètres
     * Initialise le modèle avec des paramètres par défaut.
     */
    @Deprecated
    public void initModel(int inputSize, int outputSize, int lstmNeurons) {
        initModel(inputSize, outputSize, lstmNeurons, 0.2, 0.001, "adam");
    }
    /**
     * @deprecated Utiliser initModel avec tous les paramètres
     * Initialise le modèle avec des paramètres par défaut.
     */
    @Deprecated
    public void initModel(int inputSize, int outputSize) {
        initModel(inputSize, outputSize, 50, 0.2, 0.001, "adam");
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
     * @param windowSize Taille de la fenêtre
     * @param numEpochs Nombre maximal d'epochs
     * @param patience Nombre d'epochs sans amélioration avant arrêt
     * @param minDelta Amélioration minimale pour considérer le score comme meilleur
     */
    public void trainLstm(BarSeries series, int windowSize, int numEpochs, int patience, double minDelta) {
        ensureModelWindowSize(windowSize);
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
     * Logge le score MSE, RMSE et MAE moyen et l'écart-type sur les k folds.
     * @param series série de bougies
     * @param windowSize taille de la fenêtre
     * @param numEpochs nombre maximal d'epochs
     * @param kFolds nombre de folds pour la cross-validation
     * @param lstmNeurons nombre de neurones LSTM
     * @param dropoutRate taux de Dropout
     * @param patience nombre d'epochs sans amélioration avant early stopping
     * @param minDelta amélioration minimale pour considérer le score comme meilleur
     * @param learningRate taux d'apprentissage
     * @param optimizer nom de l'optimiseur
     */
    public void crossValidateLstm(BarSeries series, int windowSize, int numEpochs, int kFolds, int lstmNeurons, double dropoutRate, int patience, double minDelta, double learningRate, String optimizer) {
        ensureModelWindowSize(windowSize);
        double[] closes = extractCloseValues(series);
        double[] normalized = normalize(closes);
        double[][][] sequences = createSequences(normalized, windowSize);
        double[][][] labelSeq = new double[sequences.length][1][1];
        for (int i = 0; i < labelSeq.length; i++) {
            labelSeq[i][0][0] = normalized[i + windowSize];
        }
        int foldSize = sequences.length / kFolds;
        double[] foldMSE = new double[kFolds];
        double[] foldRMSE = new double[kFolds];
        double[] foldMAE = new double[kFolds];
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
            // Initialiser un nouveau modèle pour ce fold avec les bons paramètres
            initModel(windowSize, 1, lstmNeurons, dropoutRate, learningRate, optimizer);
            // Early stopping
            double bestMSE = Double.MAX_VALUE;
            double bestRMSE = Double.MAX_VALUE;
            double bestMAE = Double.MAX_VALUE;
            int epochsWithoutImprovement = 0;
            for (int epoch = 0; epoch < numEpochs; epoch++) {
                model.fit(trainIterator);
                org.nd4j.linalg.api.ndarray.INDArray predictions = model.output(testInput);
                double mse = predictions.squaredDistance(testOutput) / testOutput.length();
                double rmse = Math.sqrt(mse);
                double mae = 0.0;
                for (int i = 0; i < testOutput.length(); i++) {
                    mae += Math.abs(predictions.getDouble(i) - testOutput.getDouble(i));
                }
                mae /= testOutput.length();
                if (bestMSE - mse > minDelta) {
                    bestMSE = mse;
                    bestRMSE = rmse;
                    bestMAE = mae;
                    epochsWithoutImprovement = 0;
                } else {
                    epochsWithoutImprovement++;
                    if (epochsWithoutImprovement >= patience) {
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
        for (int i = 0; i < kFolds; i++) {
            meanMSE += foldMSE[i];
            meanRMSE += foldRMSE[i];
            meanMAE += foldMAE[i];
        }
        meanMSE /= kFolds;
        meanRMSE /= kFolds;
        meanMAE /= kFolds;
        double stdMSE = 0.0, stdRMSE = 0.0, stdMAE = 0.0;
        for (int i = 0; i < kFolds; i++) {
            stdMSE += Math.pow(foldMSE[i] - meanMSE, 2);
            stdRMSE += Math.pow(foldRMSE[i] - meanRMSE, 2);
            stdMAE += Math.pow(foldMAE[i] - meanMAE, 2);
        }
        stdMSE = Math.sqrt(stdMSE / kFolds);
        stdRMSE = Math.sqrt(stdRMSE / kFolds);
        stdMAE = Math.sqrt(stdMAE / kFolds);
        logger.info("Validation croisée terminée. MSE moyen : {}, Ecart-type : {} | RMSE moyen : {}, Ecart-type : {} | MAE moyen : {}, Ecart-type : {}", meanMSE, stdMSE, meanRMSE, stdRMSE, meanMAE, stdMAE);
    }

    /**
     * Prédit la prochaine valeur de clôture à partir de la série et du windowSize.
     * @param series série de bougies
     * @param windowSize taille de la fenêtre
     * @return valeur de clôture prédite
     * @throws ModelNotFoundException si le modèle n'est pas initialisé
     * @throws InsufficientDataException si la série est trop courte
     */
    // Prédiction de la prochaine valeur de clôture
    public double predictNextClose(BarSeries series, int windowSize) {
        ensureModelWindowSize(windowSize);
        if (model == null) {
            throw new ModelNotFoundException("Le modèle LSTM n'est pas initialisé.");
        }
        double[] closes = extractCloseValues(series);
        if (closes.length < windowSize) {
            throw new InsufficientDataException("Données insuffisantes pour la prédiction (windowSize=" + windowSize + ", closes=" + closes.length + ").");
        }
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

    /**
     * Prédit le delta (variation) entre la dernière clôture et la prédiction.
     * @param series série de bougies
     * @param windowSize taille de la fenêtre
     * @return variation prédite
     */
    // Prédiction du delta (variation) de la prochaine clôture
    public double predictNextDelta(BarSeries series, int windowSize) {
        ensureModelWindowSize(windowSize);
        double predicted = predictNextClose(series, windowSize);
        double[] closes = extractCloseValues(series);
        double lastClose = closes[closes.length - 1];
        return predicted - lastClose;
    }

    /**
     * Prédit la classe (hausse/baisse/stable) pour la prochaine clôture.
     * @param series série de bougies
     * @param windowSize taille de la fenêtre
     * @param threshold seuil pour considérer une variation comme stable
     * @return "up" si hausse, "down" si baisse, "stable" sinon
     */
    /**
     * Prédiction de la classe (hausse/baisse/stable) pour la prochaine clôture.
     * @param series Série de bougies
     * @param windowSize Taille de la fenêtre
     * @param threshold Seuil pour considérer une variation comme stable
     * @return "up" si hausse, "down" si baisse, "stable" sinon
     */
    public String predictNextClass(BarSeries series, int windowSize, double threshold) {
        ensureModelWindowSize(windowSize);
        double delta = predictNextDelta(series, windowSize);
        if (delta > threshold) {
            return "up";
        } else if (delta < -threshold) {
            return "down";
        } else {
            return "stable";
        }
    }

    /**
     * Sauvegarde le modèle LSTM dans un fichier et exporte les hyperparamètres dans un fichier JSON à côté.
     * @param path chemin du fichier du modèle
     * @throws ModelPersistenceException en cas d'erreur de sauvegarde
     * @throws ModelNotFoundException si le modèle n'est pas initialisé
     */
    // Sauvegarde du modèle
    public void saveModel(String path) throws ModelPersistenceException {
        if (model != null) {
            File file = new File(path);
            File parent = file.getParentFile();
            if (parent != null && !parent.exists()) {
                parent.mkdirs();
            }
            try {
                ModelSerializer.writeModel(model, file, true);
                logger.info("Modèle sauvegardé dans le fichier : {}", path);
                // Sauvegarde des hyperparamètres
                LstmHyperparameters params = new LstmHyperparameters(
                    currentWindowSize,
                    config.getLstmNeurons(),
                    config.getDropoutRate(),
                    config.getLearningRate(),
                    config.getNumEpochs(),
                    config.getPatience(),
                    config.getMinDelta(),
                    config.getOptimizer()
                );
                String paramsPath = path + ".params.json";
                params.saveToFile(paramsPath);
                logger.info("Hyperparamètres sauvegardés dans le fichier : {}", paramsPath);
            } catch (IOException e) {
                logger.error("Erreur lors de la sauvegarde du modèle : {}", e.getMessage());
                throw new ModelPersistenceException("Erreur lors de la sauvegarde du modèle", e);
            }
        } else {
            throw new ModelNotFoundException("Impossible de sauvegarder : modèle non initialisé.");
        }
    }

    /**
     * Charge le modèle LSTM depuis un fichier.
     * @param path chemin du fichier
     * @throws ModelPersistenceException en cas d'erreur de chargement
     * @throws ModelNotFoundException si le fichier n'existe pas
     */
    // Chargement du modèle
    public void loadModel(String path) throws ModelPersistenceException {
        File f = new File(path);
        if (f.exists()) {
            try {
                model = ModelSerializer.restoreMultiLayerNetwork(f);
                logger.info("Modèle chargé depuis le fichier : {}", path);
            } catch (IOException e) {
                logger.error("Erreur lors du chargement du modèle : {}", e.getMessage());
                throw new ModelPersistenceException("Erreur lors du chargement du modèle", e);
            }
        } else {
            throw new ModelNotFoundException("Modèle non trouvé dans le fichier : " + path);
        }
    }

    /**
     * Sauvegarde le modèle LSTM en base MySQL.
     * @param symbol symbole du modèle
     * @param jdbcTemplate template JDBC Spring
     * @throws IOException en cas d'erreur d'accès à la base
     */
    // Sauvegarde du modèle dans MySQL
    public void saveModelToDb(String symbol, JdbcTemplate jdbcTemplate) throws IOException {
        if (model != null) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ModelSerializer.writeModel(model, baos, true);
            byte[] modelBytes = baos.toByteArray();
            // Création des hyperparamètres
            LstmHyperparameters params = new LstmHyperparameters(
                currentWindowSize,
                config.getLstmNeurons(),
                config.getDropoutRate(),
                config.getLearningRate(),
                config.getNumEpochs(),
                config.getPatience(),
                config.getMinDelta(),
                config.getOptimizer()
            );
            // Sérialisation JSON
            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            String hyperparamsJson = mapper.writeValueAsString(params);
            String sql = "REPLACE INTO lstm_models (symbol, model_blob, hyperparams_json, updated_date) VALUES (?, ?, ?, CURRENT_TIMESTAMP)";
            try {
                jdbcTemplate.update(sql, symbol, modelBytes, hyperparamsJson);
                logger.info("Modèle et hyperparamètres sauvegardés en base pour le symbole : {}", symbol);
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
    public void loadModelFromDb(String symbol, JdbcTemplate jdbcTemplate) throws IOException, SQLException {
        String sql = "SELECT model_blob, hyperparams_json FROM lstm_models WHERE symbol = ?";
        try {
            java.util.Map<String, Object> result = jdbcTemplate.queryForMap(sql, symbol);
            byte[] modelBytes = (byte[]) result.get("model_blob");
            String hyperparamsJson = (String) result.get("hyperparams_json");
            if (modelBytes != null) {
                ByteArrayInputStream bais = new ByteArrayInputStream(modelBytes);
                model = ModelSerializer.restoreMultiLayerNetwork(bais);
                logger.info("Modèle chargé depuis la base pour le symbole : {}", symbol);
            }
            if (hyperparamsJson != null) {
                try {
                    com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                    LstmHyperparameters params = mapper.readValue(hyperparamsJson, LstmHyperparameters.class);
                    // Injection des hyperparamètres dans l'objet (à adapter selon usage)
                    currentWindowSize = params.windowSize;
                    config.setLstmNeurons(params.lstmNeurons);
                    config.setDropoutRate(params.dropoutRate);
                    config.setLearningRate(params.learningRate);
                    config.setNumEpochs(params.numEpochs);
                    config.setPatience(params.patience);
                    config.setMinDelta(params.minDelta);
                    config.setOptimizer(params.optimizer);
                    logger.info("Hyperparamètres chargés depuis la base pour le symbole : {}", symbol);
                } catch (Exception e) {
                    logger.error("Erreur de désérialisation des hyperparamètres : {}", e.getMessage());
                }
            }
        } catch (EmptyResultDataAccessException e) {
            logger.error("Modèle non trouvé en base pour le symbole : {}", symbol);
            throw new IOException("Modèle non trouvé en base");
        }
    }

    /**
     * Entraîne le modèle avec les paramètres du fichier de configuration.
     * @param series série de bougies
     */
    // Méthode utilitaire pour entraîner le modèle avec les paramètres du fichier de config
    public void trainWithConfig(BarSeries series) {
        trainLstm(
            series,
            config.getWindowSize(),
            config.getNumEpochs(),
            config.getPatience(),
            config.getMinDelta()
        );
    }
}
