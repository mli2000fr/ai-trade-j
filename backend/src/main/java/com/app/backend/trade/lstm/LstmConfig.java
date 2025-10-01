package com.app.backend.trade.lstm;

import java.io.IOException;
import java.io.InputStream;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Properties;

// Lombok génère automatiquement getters et setters via @Getter/@Setter ci-dessous.
// Aucun code manuel de getter/setter n'est nécessaire, mais gardez bien les annotations.
import lombok.Getter;
import lombok.Setter;
import org.springframework.stereotype.Component;

/**
 * ===================================== DESCRIPTION GENERALE =====================================
 * Classe de configuration centralisant tous les hyperparamètres utilisés par le modèle LSTM.
 *
 *  RAPIDE (TL;DR débutant) :
 *   - Modifier un paramètre pour l'entraînement ? => fichier resources/lstm-config.properties
 *   - Ajouter un nouveau paramètre ? => (1) champ + getter/setter (Lombok déjà OK) (2) lire valeur
 *       dans le constructeur fichier (avec valeur par défaut) (3) idem dans constructeur ResultSet
 *       avec try/catch si colonne optionnelle (4) mettre à jour ce bloc de doc.
 *   - Accéder à la config dans un service Spring ? => @Autowired LstmConfig config;
 *   - Méthode de normalisation effective ? => config.getNormalizationMethod(); (gère 'auto').
 *
 *  OBJECTIFS : centralisation, lisibilité, reproductibilité.
 *  NE PAS MODIFIER L'ORDRE DES AFFECTATIONS sans raison technique claire (risque régression).
 *
 * ================================================================================================
 * DETAILS (déjà commentés champ par champ plus bas) :
 *  - Chargement depuis fichier properties OU depuis une ligne SQL (ResultSet)
 *  - Valeurs par défaut robustes si fichier absent
 *  - Filet de sécurité supplémentaire pour l'optimiseur
 *  - Méthode utilitaire getNormalizationMethod() applique heuristique quand 'auto'
 * ================================================================================================
 */
@Setter
@Getter
@Component
public class LstmConfig {
    // =============================== Fenêtre & Structure du réseau ===============================
    /**
     * Taille de la séquence d'entrée (longueur de la fenêtre glissante) utilisée comme contexte
     * pour prédire la valeur future. Exemple : 20 => on regarde les 20 dernières barres.
     */
    private int windowSize;

    /**
     * Nombre de neurones (unités) dans la (ou les) couche(s) LSTM. Plus il est élevé, plus la
     * capacité de représentation est grande (mais risque de sur-apprentissage + temps de calcul).
     */
    private int lstmNeurons;

    /**
     * Taux de dropout (0.0 à 1.0). Pour chaque batch d'entraînement, un pourcentage de neurones
     * est désactivé aléatoirement pour réduire le sur-apprentissage (regularisation).
     */
    private double dropoutRate;

    // =============================== Optimisation & Entraînement ===============================
    /**
     * Taux d'apprentissage (learning rate). Contrôle la vitesse d'ajustement des poids.
     * Trop haut => divergence / instabilité. Trop bas => entraînement très lent.
     */
    private double learningRate;

    /**
     * Nombre total de passes (époques/epochs) sur l'ensemble des données d'entraînement.
     */
    private int numEpochs;

    /**
     * Patience pour l'Early Stopping : nombre d'époques sans amélioration avant arrêt anticipé.
     */
    private int patience;

    /**
     * Amélioration minimale (min_delta) exigée pour considérer qu'il y a progrès (early stopping).
     */
    private double minDelta;

    /**
     * Nombre de folds pour la validation croisée (k-fold). Permet de mieux évaluer la robustesse
     * (attention : coûteux en temps). Non utilisé si cvMode ne l'exige pas.
     */
    private int kFolds;

    /**
     * Nom de l'optimiseur utilisé (ex: adam, sgd, rmsprop). C'est une chaîne car le framework
     * d'entraînement peut mapper ce texte vers une implémentation concrète.
     */
    private String optimizer;

    // =============================== Régularisation (L1 / L2) ===============================
    /**
     * Coefficient L1 : pénalise la somme des valeurs absolues des poids => favorise la sparsité.
     */
    private double l1;

    /**
     * Coefficient L2 : pénalise la somme des carrés des poids => favorise des poids plus petits.
     */
    private double l2;

    // =============================== Normalisation / Prétraitement ===============================
    /**
     * Portée (scope) de la normalisation : "window" = recalcul sur chaque sous-séquence,
     * "global" = une seule transformation pour toute la série.
     */
    private String normalizationScope = "window";

    /**
     * Méthode de normalisation demandée. Valeurs possibles : "minmax", "zscore", "auto" (détermine
     * automatiquement selon swingTradeType). Voir getNormalizationMethod().
     */
    private String normalizationMethod = "minmax";

    // =============================== Type de stratégie swing ===============================
    /**
     * Type de swing trade ciblé (ex: range, breakout, mean_reversion). Influence parfois la
     * normalisation ou la construction des labels.
     */
    private String swingTradeType = "range";

    // =============================== Features (Entrées du modèle) ===============================
    /**
     * Liste des noms de colonnes/features incluses dans chaque vecteur temporel. L'ordre est
     * significatif : il doit correspondre au pipeline de préparation des tenseurs.
     * IMPORTANT : Si on ajoute une feature côté data engineering, il faut la rajouter ici ET
     * adapter le code de construction des tenseurs.
     * Features optimisées pour swing trade professionnel (3-10 jours)
     */
    private java.util.List<String> features = java.util.Arrays.asList(
        "close", "volume", "high", "low", "open",
        "rsi", "rsi_14", "rsi_21",
        "sma", "sma_20", "sma_50",
        "ema", "ema_12", "ema_26", "ema_50",
        "macd", "macd_signal", "macd_histogram",
        "atr", "atr_14", "atr_21",
        "bollinger_high", "bollinger_low", "bollinger_width",
        "stochastic", "stochastic_d", "williams_r",
        "cci", "momentum", "roc",
        "adx", "di_plus", "di_minus",
        "obv", "volume_ratio",
        "price_position", "volatility_regime",
        "day_of_week", "month", "quarter"
    );

    // =============================== Architecture avancée ===============================
    /** Nombre de couches LSTM empilées (stack). */
    private int numLstmLayers = 2;
    /** Active (true) ou non (false) un LSTM bidirectionnel (avant + arrière). */
    private boolean bidirectional = false;
    /** Active une couche d'attention au-dessus des sorties LSTM (si supportée par le backend). */
    private boolean attention = false;

    // =============================== Labeling & Prédiction ===============================
    /** Horizon temporel (nombre de barres à l'avance) pour lequel on veut prédire la dynamique. */
    private int horizonBars = 5; // Valeur par défaut modifiable dans properties
    /** Type de seuil utilisé pour définir zones / signaux (ATR ou returns). */
    private String thresholdType = "ATR";
    /** Multiplicateur k appliqué au seuil de volatilité / amplitude. */
    private double thresholdK = 1.0;
    /** Limitation relative de la prédiction autour du cours de clôture. 0 = pas de limitation. */
    private double limitPredictionPct = 0.0;

    // =============================== Entraînement (Batch / Validation) ===============================
    /** Taille des mini-lots (batch) durant l'entraînement. */
    private int batchSize = 128; // Remplacé si override dans les properties
    /** Mode de validation croisée : split, timeseries, kfold. */
    private String cvMode = "split";

    // =============================== Pipeline V2 (options évoluées) ===============================
    /** Active pipeline étiquettes scalaires + walk-forward. */
    private boolean useScalarV2 = false;
    /** Utilise le log-return comme cible (target) plutôt que le prix normalisé. */
    private boolean useLogReturnTarget = true;
    /** Active la validation walk-forward (utilisé si useScalarV2 = true). */
    private boolean useWalkForwardV2 = true;
    /** Nombre de segments (splits) dans la validation walk-forward. */
    private int walkForwardSplits = 4;
    /** Barres d'embargo (séparation temporelle) pour éviter la fuite entre train/test. */
    private int embargoBars = 0;
    /** Graine (seed) pour reproductibilité (initialisation aléatoire). */
    private long seed = 42L;
    /** Cap (plafond) du profit factor dans le score métier V2. */
    private double businessProfitFactorCap = 3.0;
    /** Paramètre gamma pour pénaliser le drawdown dans le score métier V2. */
    private double businessDrawdownGamma = 1.2;

    // =============================== Paramètres Trading (Simulation / Backtest) ===============================
    /** Capital total simulé pour calculer la taille de position. */
    private double capital = 10000.0;
    /** Pourcentage du capital risqué par trade (ex: 0.01 = 1%). */
    private double riskPct = 0.01;
    /** Facteur multiplicateur dans la formule de sizing basée sur l'ATR. */
    private double sizingK = 1.0;
    /** Frais (commission) proportionnels par trade (ex: 0.0005 = 0.05%). */
    private double feePct = 0.0005;
    /** Slippage moyen estimé par trade (ex: 0.0002 = 0.02%). */
    private double slippagePct = 0.0002;

    // =============================== Détection de dérive (Drift) ===============================
    /** Seuil de divergence KL (Kullback-Leibler) pour signaler un drift de distribution. */
    private double klDriftThreshold = 0.15;
    /** Seuil de shift de moyenne exprimé en nombre d'écarts-types. */
    private double meanShiftSigmaThreshold = 2.0;

    /**
     * Constructeur par défaut : charge les hyperparamètres depuis le fichier
     * resources/lstm-config.properties si présent. Chaque paramètre possède une valeur de secours
     * (fallback) afin de fonctionner même si une clé manque dans le fichier.
     *
     * NOTE IMPORTANTE : Ne pas supprimer la seconde affectation de 'optimizer' en fin de bloc,
     * elle assure que même si aucune propriété n'est chargée (input == null), on a "adam" par défaut.
     * (Ceci est un filet de sécurité supplémentaire.)
     *
     * En cas d'erreur d'I/O, on remonte une RuntimeException pour signaler l'impossibilité de
     * poursuivre (erreur de configuration critique).
     */
    public LstmConfig() {
        Properties props = new Properties();
        try (InputStream input = getClass().getClassLoader().getResourceAsStream("lstm-config.properties")) {
            if (input != null) {
                // Chargement du fichier de propriétés
                props.load(input);
                // Lecture des propriétés avec valeurs par défaut. IMPORTANT : Ne pas changer l'ordre
                // sauf nécessité. Chaque conversion parse la chaîne vers le type désiré.
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
                batchSize = Integer.parseInt(props.getProperty("batchSize", "256"));
                cvMode = props.getProperty("cvMode", "split");
                useScalarV2 = Boolean.parseBoolean(props.getProperty("useScalarV2", "false"));
                useLogReturnTarget = Boolean.parseBoolean(props.getProperty("useLogReturnTarget", "true"));
                useWalkForwardV2 = Boolean.parseBoolean(props.getProperty("useWalkForwardV2", "true"));
                walkForwardSplits = Integer.parseInt(props.getProperty("walkForwardSplits", "4"));
                embargoBars = Integer.parseInt(props.getProperty("embargoBars", "0"));
                seed = Long.parseLong(props.getProperty("seed", "42"));
                businessProfitFactorCap = Double.parseDouble(props.getProperty("businessProfitFactorCap", "3.0"));
                businessDrawdownGamma = Double.parseDouble(props.getProperty("businessDrawdownGamma", "1.2"));
                capital = Double.parseDouble(props.getProperty("capital", "10000.0"));
                riskPct = Double.parseDouble(props.getProperty("riskPct", "0.01"));
                sizingK = Double.parseDouble(props.getProperty("sizingK", "1.0"));
                feePct = Double.parseDouble(props.getProperty("feePct", "0.0005"));
                slippagePct = Double.parseDouble(props.getProperty("slippagePct", "0.0002"));
                klDriftThreshold = Double.parseDouble(props.getProperty("klDriftThreshold", "0.15"));
                meanShiftSigmaThreshold = Double.parseDouble(props.getProperty("meanShiftSigmaThreshold", "2.0"));
            }
            // FILET DE SECURITE : même si le fichier n'existe pas, on veut un optimizer par défaut.
            optimizer = props.getProperty("optimizer", "adam");
        } catch (IOException e) {
            throw new RuntimeException("Impossible de charger lstm-config.properties", e);
        }
    }

    /**
     * Constructeur basé sur un ResultSet (lecture depuis base de données). On suppose que les
     * colonnes existent. Pour les colonnes optionnelles, on encapsule dans des try/catch afin de
     * ne pas provoquer d'erreur si la colonne n'est pas encore déployée. Ceci permet une migration
     * progressive du schéma. Ne modifiez pas les noms de colonnes sans aligner la BDD.
     *
     * @param rs ResultSet pointant sur une ligne contenant une configuration complète.
     * @throws SQLException en cas d'échec d'accès aux colonnes obligatoires.
     */
    public LstmConfig(ResultSet rs) throws SQLException {
        // Champs obligatoires (présumés toujours disponibles)
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
        // Valeur de repli si champ NULL (ex: champ non renseigné)
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
        // Champs optionnels : on tente, sinon on ignore silencieusement (retro-compatibilité schéma)
        try { this.useScalarV2 = rs.getBoolean("use_scalar_v2"); } catch (Exception ignored) {}
        try { this.useLogReturnTarget = rs.getBoolean("use_log_return_target"); } catch (Exception ignored) {}
        try { this.useWalkForwardV2 = rs.getBoolean("use_walk_forward_v2"); } catch (Exception ignored) {}
        try { this.walkForwardSplits = rs.getInt("walk_forward_splits"); } catch (Exception ignored) {}
        try { this.embargoBars = rs.getInt("embargo_bars"); } catch (Exception ignored) {}
        try { this.seed = rs.getLong("seed"); } catch (Exception ignored) {}
        try { this.businessProfitFactorCap = rs.getDouble("business_profit_factor_cap"); } catch (Exception ignored) {}
        try { this.businessDrawdownGamma = rs.getDouble("business_drawdown_gamma"); } catch (Exception ignored) {}
        try { this.capital = rs.getDouble("capital"); } catch (Exception ignored) {}
        try { this.riskPct = rs.getDouble("risk_pct"); } catch (Exception ignored) {}
        try { this.sizingK = rs.getDouble("sizing_k"); } catch (Exception ignored) {}
        try { this.feePct = rs.getDouble("fee_pct"); } catch (Exception ignored) {}
        try { this.slippagePct = rs.getDouble("slippage_pct"); } catch (Exception ignored) {}
        try { this.klDriftThreshold = rs.getDouble("kl_drift_threshold"); } catch (Exception ignored) {}
        try { this.meanShiftSigmaThreshold = rs.getDouble("mean_shift_sigma_threshold"); } catch (Exception ignored) {}
    }

    /**
     * Fournit la méthode de normalisation effectivement utilisée. Si l'utilisateur a mis
     * "auto" dans le fichier de configuration ou la base, on applique une heuristique :
     *  - mean_reversion => zscore (centrage-réduction adapté aux écarts autour d'une moyenne)
     *  - sinon => minmax (mise à l'échelle dans [0,1])
     *
     * Cette méthode ne modifie pas l'état interne : elle renvoie juste la valeur choisie.
     * @return la chaîne représentant la méthode de normalisation effective.
     */
    public String getNormalizationMethod() {
        if ("auto".equalsIgnoreCase(normalizationMethod)) {
            if ("mean_reversion".equalsIgnoreCase(swingTradeType)) {
                return "zscore";
            } else {
                // Pour range, breakout, etc. => défaut minmax
                return "minmax";
            }
        }
        return normalizationMethod;
    }
}
