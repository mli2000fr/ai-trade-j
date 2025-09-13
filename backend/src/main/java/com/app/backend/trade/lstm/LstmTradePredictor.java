package com.app.backend.trade.lstm;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.sql.ResultSet;
import java.sql.SQLException;

import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
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

@Service
public class LstmTradePredictor {
    private MultiLayerNetwork model;

    public LstmTradePredictor() {
        // Le modèle sera initialisé via la méthode initModel
    }

    public void initModel(int inputSize, int outputSize) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .list()
            .layer(new LSTM.Builder()
                .nIn(inputSize)
                .nOut(50)
                .activation(Activation.TANH)
                .build())
            .layer(new RnnOutputLayer.Builder()
                .nIn(50)
                .nOut(outputSize)
                .activation(Activation.IDENTITY) // Correction ici
                .lossFunction(LossFunctions.LossFunction.MSE) // Correction ici
                .build())
            .build();
        model = new MultiLayerNetwork(conf);
        model.init();
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
        double[] norm = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            norm[i] = (values[i] - min) / (max - min);
        }
        return norm;
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

    // Entraînement du modèle LSTM
    public void trainLstm(BarSeries series, int windowSize, int numEpochs) {
        double[] closes = extractCloseValues(series);
        double[] normalized = normalize(closes);
        double[][][] sequences = createSequences(normalized, windowSize);
        org.nd4j.linalg.api.ndarray.INDArray input = toINDArray(sequences);
        // Les labels sont la valeur suivante après chaque séquence, sous forme 3D
        double[][][] labelSeq = new double[sequences.length][1][1];
        for (int i = 0; i < labelSeq.length; i++) {
            labelSeq[i][0][0] = normalized[i + windowSize];
        }
        org.nd4j.linalg.api.ndarray.INDArray labelArray = org.nd4j.linalg.factory.Nd4j.create(labelSeq);
        org.nd4j.linalg.dataset.api.iterator.DataSetIterator iterator = new ListDataSetIterator<>(
            java.util.Collections.singletonList(new org.nd4j.linalg.dataset.DataSet(input, labelArray))
        );
        for (int i = 0; i < numEpochs; i++) {
            model.fit(iterator);
        }
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
            ModelSerializer.writeModel(model, file, true);
        }
    }

    // Chargement du modèle
    public void loadModel(String path) throws IOException {
        File f = new File(path);
        if (f.exists()) {
            model = ModelSerializer.restoreMultiLayerNetwork(f);
        }
    }

    // Sauvegarde du modèle dans MySQL
    public void saveModelToDb(String symbol, JdbcTemplate jdbcTemplate) throws IOException {
        if (model != null) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ModelSerializer.writeModel(model, baos, true);
            byte[] modelBytes = baos.toByteArray();
            String sql = "REPLACE INTO lstm_models (symbol, model_blob, updated_date) VALUES (?, ?, CURRENT_TIMESTAMP)";
            jdbcTemplate.update(sql, symbol, modelBytes);
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
            }
        } catch (EmptyResultDataAccessException e) {
            throw new IOException("Modèle non trouvé en base");
        }
    }
}
