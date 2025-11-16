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
import org.nd4j.linalg.lossfunctions.LossFunctions;

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

    public PortfolioAllocationModel(int inputSize, PortfolioLearningConfig config) {
        this.inputSize = inputSize;
        this.config = config;
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
                        .build())
                .layer(new DenseLayer.Builder().nIn(config.getHidden1()).nOut(config.getHidden2())
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(config.getHidden2()).nOut(1).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(50));
        return net;
    }

    public double[] predictBatch(double[][] features) {
        double[] scores = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            org.nd4j.linalg.api.ndarray.INDArray in = org.nd4j.linalg.factory.Nd4j.create(features[i]);
            double v = network.output(in, false).getDouble(0);
            scores[i] = v;
        }
        return scores;
    }
}

