package com.app.backend.trade.lstm;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class LstmConfig {
    private int windowSize;
    private int lstmNeurons;
    private double dropoutRate;
    private double learningRate;
    private int numEpochs;
    private int patience;
    private double minDelta;
    private String optimizer;

    public LstmConfig() {
        Properties props = new Properties();
        try (InputStream input = getClass().getClassLoader().getResourceAsStream("lstm-config.properties")) {
            if (input != null) {
                props.load(input);
                windowSize = Integer.parseInt(props.getProperty("windowSize", "20"));
                lstmNeurons = Integer.parseInt(props.getProperty("lstmNeurons", "50"));
                dropoutRate = Double.parseDouble(props.getProperty("dropoutRate", "0.2"));
                learningRate = Double.parseDouble(props.getProperty("learningRate", "0.001"));
                numEpochs = Integer.parseInt(props.getProperty("numEpochs", "100"));
                patience = Integer.parseInt(props.getProperty("patience", "10"));
                minDelta = Double.parseDouble(props.getProperty("minDelta", "0.0001"));
                optimizer = props.getProperty("optimizer", "adam");
            }
        } catch (IOException e) {
            throw new RuntimeException("Impossible de charger lstm-config.properties", e);
        }
    }

    public int getWindowSize() { return windowSize; }
    public int getLstmNeurons() { return lstmNeurons; }
    public double getDropoutRate() { return dropoutRate; }
    public double getLearningRate() { return learningRate; }
    public int getNumEpochs() { return numEpochs; }
    public int getPatience() { return patience; }
    public double getMinDelta() { return minDelta; }
    public String getOptimizer() { return optimizer; }
}

