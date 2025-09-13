package com.app.backend.trade.lstm;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;

public class LstmHyperparameters {
    public int windowSize;
    public int lstmNeurons;
    public double dropoutRate;
    public double learningRate;
    public int numEpochs;
    public int patience;
    public double minDelta;
    public String optimizer;

    public LstmHyperparameters() {}

    public LstmHyperparameters(int windowSize, int lstmNeurons, double dropoutRate, double learningRate, int numEpochs, int patience, double minDelta, String optimizer) {
        this.windowSize = windowSize;
        this.lstmNeurons = lstmNeurons;
        this.dropoutRate = dropoutRate;
        this.learningRate = learningRate;
        this.numEpochs = numEpochs;
        this.patience = patience;
        this.minDelta = minDelta;
        this.optimizer = optimizer;
    }

    public void saveToFile(String path) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.writeValue(new File(path), this);
    }

    public static LstmHyperparameters loadFromFile(String path) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(new File(path), LstmHyperparameters.class);
    }
}

