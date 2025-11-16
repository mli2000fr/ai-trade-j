package com.app.backend.trade.portfolio.rl;

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

import java.io.ByteArrayInputStream;
import java.io.IOException;

/**
 * Policy réseau simple (actor) pour la couche RL offline.
 * Entrée: features par symbole (mêmes vecteurs que buildInferenceFeatures).
 * Sortie: un logit d'action (score) par symbole.
 * Les logits sont transformés en poids via softmax + contraintes risk plus loin.
 */
@Getter
@Setter
public class PortfolioRlPolicyModel {
    private MultiLayerNetwork network;
    private int inputSize;
    private double[] featureMeans;
    private double[] featureStds;
    private PortfolioRlConfig config;

    public PortfolioRlPolicyModel(int inputSize, PortfolioRlConfig config) {
        this.inputSize = inputSize;
        this.config = config;
        this.network = buildNetwork(inputSize, config);
    }

    private MultiLayerNetwork buildNetwork(int in, PortfolioRlConfig cfg) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(777)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(cfg.getLearningRate()))
                .l2(cfg.getL2())
                .list()
                .layer(new DenseLayer.Builder().nIn(in).nOut(cfg.getHidden1()).activation(Activation.RELU).dropOut(cfg.getDropout()).build())
                .layer(new DenseLayer.Builder().nIn(cfg.getHidden1()).nOut(cfg.getHidden2()).activation(Activation.RELU).dropOut(cfg.getDropout()).build())
                .layer(new OutputLayer.Builder().activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).nIn(cfg.getHidden2()).nOut(1).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(50));
        return net;
    }

    public double[] predictBatch(double[][] features) {
        double[] logits = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            double[] f = features[i];
            if (featureMeans != null && featureStds != null && featureMeans.length == f.length) {
                f = f.clone();
                for (int j = 0; j < f.length; j++) {
                    double std = featureStds[j] == 0 ? 1.0 : featureStds[j];
                    f[j] = (f[j] - featureMeans[j]) / std;
                }
            }
            org.nd4j.linalg.api.ndarray.INDArray in = org.nd4j.linalg.factory.Nd4j.create(f);
            logits[i] = network.output(in, false).getDouble(0);
        }
        return logits;
    }

    public static PortfolioRlPolicyModel loadFromBytes(byte[] bytes, PortfolioRlConfig cfg, int inputSize, double[] means, double[] stds) throws IOException {
        if (bytes == null || bytes.length == 0) throw new IOException("Bytes RL policy vides");
        try (ByteArrayInputStream bais = new ByteArrayInputStream(bytes)) {
            MultiLayerNetwork net = org.deeplearning4j.util.ModelSerializer.restoreMultiLayerNetwork(bais);
            PortfolioRlPolicyModel m = new PortfolioRlPolicyModel(inputSize, cfg);
            m.setNetwork(net);
            m.setFeatureMeans(means);
            m.setFeatureStds(stds);
            return m;
        }
    }
}

