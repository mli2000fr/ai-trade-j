package com.app.backend.trade.portfolio.advanced;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.ByteArrayInputStream;
import java.io.IOException;

/**
 * Modèle multi-tête avancé:
 * - Tête sélection (sigmoid) => probabilité inclusion du symbole.
 * - Tête poids brut (relu) => intensité non normalisée.
 * - Tête direction (tanh) => orientation (-1..1) pour long/short.
 * Chaque symbole est inféré indépendamment; la normalisation croisée est effectuée côté service.
 */
@Getter
@Setter
public class PortfolioMultiHeadModel {

    private final int inputSize;
    private final ComputationGraph graph;
    private double[] featureMeans;
    private double[] featureStds;

    public PortfolioMultiHeadModel(int inputSize, int h1, int h2, double lr, double l2, double dropout) {
        this.inputSize = inputSize;
        ComputationGraphConfiguration.GraphBuilder gb = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(lr))
                .l2(l2)
                .graphBuilder()
                .addInputs("in")
                .setInputTypes(InputType.feedForward(inputSize))
                .addLayer("dense1", new DenseLayer.Builder().nIn(inputSize).nOut(h1).activation(Activation.RELU).dropOut(dropout).build(), "in")
                .addLayer("dense2", new DenseLayer.Builder().nIn(h1).nOut(h2).activation(Activation.RELU).dropOut(dropout).build(), "dense1")
                .addLayer("head_select", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.XENT).activation(Activation.SIGMOID).nIn(h2).nOut(1).build(), "dense2")
                .addLayer("head_weight", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).activation(Activation.RELU).nIn(h2).nOut(1).build(), "dense2")
                .addLayer("head_side", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).activation(Activation.TANH).nIn(h2).nOut(1).build(), "dense2")
                .setOutputs("head_select", "head_weight", "head_side");
        ComputationGraphConfiguration conf = gb.build();
        this.graph = new ComputationGraph(conf);
        this.graph.init();
        this.graph.setListeners(new ScoreIterationListener(50));
    }

    public double[][] predict(double[] features) {
        double[] f = features;
        if (featureMeans != null && featureStds != null && featureMeans.length == f.length) {
            f = f.clone();
            for (int i = 0; i < f.length; i++) {
                double std = featureStds[i] == 0 ? 1.0 : featureStds[i];
                f[i] = (f[i] - featureMeans[i]) / std;
            }
        }
        org.nd4j.linalg.api.ndarray.INDArray in = org.nd4j.linalg.factory.Nd4j.create(f);
        org.nd4j.linalg.api.ndarray.INDArray[] out = graph.output(in);
        return new double[][]{
                {out[0].getDouble(0)}, // select prob
                {out[1].getDouble(0)}, // raw weight
                {out[2].getDouble(0)}  // side (-1..1)
        };
    }

    public double[][] predictBatch(double[][] features) {
        double[][] result = new double[features.length][3];
        for (int i = 0; i < features.length; i++) {
            double[][] r = predict(features[i]);
            result[i][0] = r[0][0];
            result[i][1] = r[1][0];
            result[i][2] = r[2][0];
        }
        return result;
    }

    public static PortfolioMultiHeadModel loadFromBytes(byte[] bytes, int inputSize, int h1, int h2, double lr, double l2, double dropout, double[] means, double[] stds) throws IOException {
        if (bytes == null || bytes.length == 0) throw new IOException("Bytes modèle vides multi-head");
        try (ByteArrayInputStream bais = new ByteArrayInputStream(bytes)) {
            ComputationGraph restored = org.deeplearning4j.util.ModelSerializer.restoreComputationGraph(bais);
            PortfolioMultiHeadModel m = new PortfolioMultiHeadModel(inputSize, h1, h2, lr, l2, dropout);
            // Remplacer graph par restauré
            m.getGraph().setParams(restored.params());
            m.setFeatureMeans(means);
            m.setFeatureStds(stds);
            return m;
        }
    }
}
