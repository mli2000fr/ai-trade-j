package com.app.backend.trade.portfolio.learning;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;

import java.io.ByteArrayInputStream;
import java.io.IOException;

/**
 * Modèle MLP simple pour produire un score risque-ajusté par symbole.
 * Entrée: vecteur features par symbole (dimension N).
 * Sortie: score scalaire (non normalisé). Les scores seront transformés en poids.
 */
@Getter
@Setter
public class PortfolioAllocationModel {

    private MultiLayerNetwork network;
    private int inputSize;
    private PortfolioLearningConfig config;
    // Stats de normalisation pour inférence cohérente
    private double[] featureMeans;
    private double[] featureStds;
    // Loss personnalisées
    private LossPortfolioAllocation customLoss;        // utilitaire d'évaluation (labels/pred)
    private PortfolioCustomLoss trainLoss;             // ILossFunction pour le réseau

    public PortfolioAllocationModel(int inputSize, PortfolioLearningConfig config) {
        this.inputSize = inputSize;
        this.config = config;
        this.customLoss = new LossPortfolioAllocation(config.getLambdaTurnover(), config.getLambdaDrawdown());
        this.trainLoss = new PortfolioCustomLoss(config.getLambdaTurnover(), config.getLambdaDrawdown());
        this.network = buildNetwork();
    }

    private MultiLayerNetwork buildNetwork() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(config.getLearningRate()))
                .l2(config.getL2())
                .list()
                .layer(new DenseLayer.Builder().nIn(inputSize).nOut(config.getHidden1())
                        .activation(Activation.RELU)
                        .dropOut(config.getDropout())
                        .build())
                .layer(new DenseLayer.Builder().nIn(config.getHidden1()).nOut(config.getHidden2())
                        .activation(Activation.RELU)
                        .dropOut(config.getDropout())
                        .build())
                .layer(new OutputLayer.Builder()
                        .lossFunction(trainLoss)
                        .activation(Activation.IDENTITY)
                        .nIn(config.getHidden2()).nOut(1).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(50));
        return net;
    }

    // Méthode utilitaire: calcule la perte custom à partir de labels et prédictions
    public double computeCustomLoss(org.nd4j.linalg.api.ndarray.INDArray labels,
                                    org.nd4j.linalg.api.ndarray.INDArray predictions) {
        return customLoss.computeScore(labels, predictions);
    }

    public double[] predictBatch(double[][] features) {
        double[] scores = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            double[] f = features[i];
            if (featureMeans != null && featureStds != null && featureMeans.length == f.length && featureStds.length == f.length) {
                f = f.clone();
                for (int j = 0; j < f.length; j++) {
                    double std = featureStds[j] == 0 ? 1.0 : featureStds[j];
                    f[j] = (f[j] - featureMeans[j]) / std;
                }
            }
            org.nd4j.linalg.api.ndarray.INDArray in = org.nd4j.linalg.factory.Nd4j.create(f);
            double v = network.output(in, false).getDouble(0);
            scores[i] = v;
        }
        return scores;
    }

    public static PortfolioAllocationModel loadFromBytes(byte[] bytes, PortfolioLearningConfig config, int inputSize, double[] means, double[] stds) throws IOException {
        if (bytes == null || bytes.length == 0) throw new IOException("Bytes modèle vides");
        try (ByteArrayInputStream bais = new ByteArrayInputStream(bytes)) {
            org.deeplearning4j.nn.multilayer.MultiLayerNetwork net = org.deeplearning4j.util.ModelSerializer.restoreMultiLayerNetwork(bais);
            PortfolioAllocationModel m = new PortfolioAllocationModel(inputSize, config);
            m.setNetwork(net);
            m.setFeatureMeans(means);
            m.setFeatureStds(stds);
            return m;
        }
    }
}
