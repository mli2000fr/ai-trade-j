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
     * Méthode de normalisation utilisée : "minmax", "zscore", etc.
     */
    private String normalizationMethod = "minmax";

    /**
     * Type de swing trade : "range", "breakout", "mean_reversion", etc.
     */
    private String swingTradeType = "range";

    /**
     * Liste des features à inclure dans la séquence d'entrée (ex : close, volume, rsi, sma, ema, macd, atr, bollinger_high, bollinger_low, stochastic, cci, momentum, day_of_week, month, session).
     */
    private java.util.List<String> features = java.util.Arrays.asList(
        "close", "volume", "rsi", "sma", "ema", "macd",
        "atr", "bollinger_high", "bollinger_low",
        "stochastic", "cci", "momentum",
        "day_of_week", "month", "session"
    );

    /**
     * Nombre de couches LSTM empilées (stacked LSTM)
     */
    private int numLstmLayers = 2;

    /**
     * Active le mode bidirectionnel (Bidirectional LSTM)
     */
    private boolean bidirectional = false;

    /**
     * Active la couche d'attention (si supportée)
     */
    private boolean attention = false;

    /**
     * Horizon de prédiction pour la classification swing (nombre de barres à l'avance)
     */
    private int horizonBars = 5; // Par défaut 5, configurable

    /**
     * Type de seuil swing trade : "ATR" ou "returns" (volatilité des log-returns)
     */
    private String thresholdType = "ATR";

    /**
     * Coefficient multiplicateur pour le seuil (k)
     */
    private double thresholdK = 1.0;

    /**
     * Pourcentage de limitation de la prédiction autour du close (ex: 0.1 pour ±10%). 0 = désactivé.
     */
    private double limitPredictionPct = 0.0;

    /**
     * Taille du batch pour l'entraînement (par défaut 64)
     */
    private int batchSize = 64;

    /**
     * Mode de validation croisée : split, timeseries, kfold
     */
    private String cvMode = "split";

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
                normalizationMethod = props.getProperty("normalizationMethod", "auto");
                swingTradeType = props.getProperty("swingTradeType", "range");
                numLstmLayers = Integer.parseInt(props.getProperty("numLstmLayers", "2"));
                bidirectional = Boolean.parseBoolean(props.getProperty("bidirectional", "false"));
                attention = Boolean.parseBoolean(props.getProperty("attention", "false"));
                horizonBars = Integer.parseInt(props.getProperty("horizonBars", "5"));
                thresholdType = props.getProperty("thresholdType", "ATR");
                thresholdK = Double.parseDouble(props.getProperty("thresholdK", "1.0"));
                limitPredictionPct = Double.parseDouble(props.getProperty("limitPredictionPct", "0.0"));
                batchSize = Integer.parseInt(props.getProperty("batchSize", "64"));
                cvMode = props.getProperty("cvMode", "split");
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
        normalizationMethod = rs.getString("normalization_method") != null ? rs.getString("normalization_method") : "auto";
        swingTradeType = rs.getString("swing_trade_type") != null ? rs.getString("swing_trade_type") : "range";
        numLstmLayers = rs.getInt("num_lstm_layers");
        bidirectional = rs.getBoolean("bidirectional");
        attention = rs.getBoolean("attention");
        horizonBars = rs.getInt("horizon_bars");
        thresholdType = rs.getString("threshold_type") != null ? rs.getString("threshold_type") : "ATR";
        thresholdK = rs.getDouble("threshold_k");
        limitPredictionPct = rs.getDouble("limit_prediction_pct");
        batchSize = rs.getInt("batch_size");
        cvMode = rs.getString("cv_mode") != null ? rs.getString("cv_mode") : "split";
    }

    /**
     * Retourne la méthode de normalisation adaptée selon le type de swing trade si "auto".
     */
    public String getNormalizationMethod() {
        if ("auto".equalsIgnoreCase(normalizationMethod)) {
            if ("mean_reversion".equalsIgnoreCase(swingTradeType)) {
                return "zscore";
            } else {
                // Par défaut : range, breakout, etc.
                return "minmax";
            }
        }
        return normalizationMethod;
    }
}
