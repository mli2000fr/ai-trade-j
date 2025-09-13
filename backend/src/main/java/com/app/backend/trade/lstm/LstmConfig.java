package com.app.backend.trade.lstm;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;
import org.springframework.stereotype.Component;

/**
 * Classe de configuration pour les hyperparamètres du modèle LSTM.
 * <p>
 * Les paramètres sont chargés dynamiquement depuis le fichier lstm-config.properties
 * situé dans le dossier resources. Permet de centraliser et modifier facilement
 * les hyperparamètres du modèle sans changer le code source.
 * </p>
 */
@Component
public class LstmConfig {
    private int windowSize;
    private int lstmNeurons;
    private double dropoutRate;
    private double learningRate;
    private int numEpochs;
    private int patience;
    private double minDelta;
    private String optimizer;

    /**
     * Constructeur. Charge les hyperparamètres depuis le fichier lstm-config.properties.
     * @throws RuntimeException si le fichier de configuration ne peut pas être chargé
     */
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

    /**
     * @return la taille de la fenêtre (windowSize) utilisée pour les séquences LSTM
     */
    public int getWindowSize() { return windowSize; }
    /**
     * @return le nombre de neurones dans la couche LSTM
     */
    public int getLstmNeurons() { return lstmNeurons; }
    /**
     * @return le taux de dropout appliqué au modèle LSTM
     */
    public double getDropoutRate() { return dropoutRate; }
    /**
     * @return le taux d'apprentissage utilisé pour l'optimiseur
     */
    public double getLearningRate() { return learningRate; }
    /**
     * @return le nombre d'epochs pour l'entraînement du modèle
     */
    public int getNumEpochs() { return numEpochs; }
    /**
     * @return le nombre d'epochs sans amélioration avant early stopping
     */
    public int getPatience() { return patience; }
    /**
     * @return l'amélioration minimale du score pour considérer une epoch comme meilleure
     */
    public double getMinDelta() { return minDelta; }
    /**
     * @return le nom de l'optimiseur utilisé (adam, rmsprop, sgd, etc.)
     */
    public String getOptimizer() { return optimizer; }
}
