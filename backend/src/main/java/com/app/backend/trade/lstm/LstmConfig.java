package com.app.backend.trade.lstm;

import java.io.IOException;
import java.io.InputStream;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Properties;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.springframework.stereotype.Component;

/**
 * Classe de configuration pour les hyperparamètres du modèle LSTM.
 * <p>
 * Les paramètres sont chargés dynamiquement depuis le fichier lstm-config.properties
 * situé dans le dossier resources. Permet de centraliser et modifier facilement
 * les hyperparamètres du modèle sans changer le code source.
 * </p>
 */
@Setter
@Getter
@Component
public class LstmConfig {
    /**
     * Nombre de périodes utilisées pour chaque séquence d'entrée du LSTM.
     * Représente la taille de la fenêtre glissante (ex : 20 jours).
     */
    private int windowSize;

    /**
     * Nombre de neurones dans la couche LSTM.
     * Plus ce nombre est élevé, plus le modèle peut apprendre des patterns complexes.
     */
    private int lstmNeurons;

    /**
     * Taux de dropout pour la régularisation.
     * Pourcentage de neurones désactivés aléatoirement à chaque itération afin d'éviter le sur-apprentissage.
     */
    private double dropoutRate;

    /**
     * Taux d'apprentissage du modèle.
     * Contrôle la vitesse d'ajustement des poids lors de l'entraînement.
     */
    private double learningRate;

    /**
     * Nombre d'époques d'entraînement.
     * Définit combien de fois le modèle parcourt l'ensemble des données.
     */
    private int numEpochs;

    /**
     * Patience pour l'arrêt anticipé (early stopping).
     * Nombre d'époques sans amélioration avant d'arrêter l'entraînement.
     */
    private int patience;

    /**
     * Amélioration minimale pour l'arrêt anticipé.
     * Seuil de progression requis pour considérer qu'il y a amélioration.
     */
    private double minDelta;

    /**
     * Nombre de folds pour la validation croisée (cross-validation).
     * Permet d'évaluer la robustesse du modèle sur plusieurs sous-ensembles.
     */
    private int kFolds;

    /**
     * Algorithme d'optimisation utilisé pour l'entraînement.
     * Exemple : "adam", "sgd", "rmsprop".
     */
    private String optimizer;

    /**
     * Coefficient de régularisation L1.
     * Contrôle la pénalité appliquée aux poids pour éviter le sur-apprentissage.
     */
    private double l1;

    /**
     * Coefficient de régularisation L2.
     * Contrôle la pénalité appliquée aux poids pour éviter le sur-apprentissage.
     */
    private double l2;

    /**
     * Portée de la normalisation : "window" (fenêtre locale) ou "global" (toute la série).
     */
    private String normalizationScope = "window";

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
                kFolds = Integer.parseInt(props.getProperty("kFolds", "5"));
                l1 = Double.parseDouble(props.getProperty("l1", "0.0"));
                l2 = Double.parseDouble(props.getProperty("l2", "0.0"));
                normalizationScope = props.getProperty("normalizationScope", "window");
            }
            optimizer = props.getProperty("optimizer", "adam");
        } catch (IOException e) {
            throw new RuntimeException("Impossible de charger lstm-config.properties", e);
        }
    }

    /**
     * Constructeur. Initialise les hyperparamètres à partir d'un ResultSet.
     * @param rs ResultSet contenant les valeurs des hyperparamètres
     * @throws SQLException si une erreur se produit lors de l'accès aux données du ResultSet
     */
    public LstmConfig(ResultSet rs) throws SQLException {
        windowSize = rs.getInt("window_size");
        lstmNeurons = rs.getInt("lstm_neurons");
        dropoutRate = rs.getDouble("dropout_rate");
        learningRate = rs.getDouble("learning_rate");
        numEpochs = rs.getInt("num_epochs");
        patience = rs.getInt("patience");
        minDelta = rs.getDouble("min_delta");
        kFolds = rs.getInt("k_folds");
        optimizer = rs.getString("optimizer");
        l1 = rs.getDouble("l1");
        l2 = rs.getDouble("l2");
        normalizationScope = rs.getString("normalization_scope");
    }
}
