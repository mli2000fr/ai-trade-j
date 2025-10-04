package com.app.backend.trade.lstm;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;

/**
 * Repository (accès base de données) dédié au stockage et à la récupération :
 * 1. Des hyperparamètres d'un modèle LSTM (table: lstm_hyperparams)
 * 2. Des métriques de tuning associées (table: lstm_tuning_metrics)
 *
 * Points importants pour un débutant :
 * - Cette classe N'IMPLÉMENTE PAS de logique métier du modèle. Elle ne fait que lire/écrire en base.
 * - On utilise JdbcTemplate (Spring) pour exécuter des requêtes SQL préparées (sécurité + robustesse).
 * - Les listes (features) sont stockées en JSON via Gson (sérialisation/désérialisation).
 * - L'ordre des colonnes dans les requêtes SQL DOIT ABSOLUMENT correspondre à l'ordre des paramètres fournis dans jdbcTemplate.update(...).
 *   NE PAS MODIFIER SANS RAISON -> RISQUE DE RÉGRESSION SILENCIEUSE.
 * - REPLACE INTO (MySQL) = si une ligne avec la même clé primaire (symbol) existe, elle est supprimée puis recréée.
 *   Conséquence : updated_date est rafraîchie automatiquement (CURRENT_TIMESTAMP ici).
 */
@Repository
public class LstmHyperparamsRepository {

    // =========================================================
    // Dépendances
    // =========================================================

    /**
     * JdbcTemplate fourni par Spring. Sert à envoyer des requêtes SQL à la base.
     * Il gère :
     * - la connexion
     * - la préparation des statements
     * - le mapping minimal des résultats
     */
    private final JdbcTemplate jdbcTemplate;

    /**
     * Gson sert à convertir (sérialiser/désérialiser) une liste Java -> JSON (pour la colonne 'features').
     */
    private final Gson gson = new Gson();

    // =========================================================
    // Constantes SQL (extraites pour lisibilité uniquement — LOGIQUE IDENTIQUE)
    // Très important : ne pas changer l'ordre des colonnes sans ajuster l'ordre des paramètres.
    // =========================================================

    /**
     * Requête d'UPSERT (via REPLACE INTO) des hyperparamètres.
     * CURRENT_TIMESTAMP met à jour automatiquement la colonne updated_date.
     */
    private static final String SQL_SAVE_HYPERPARAMS =
            "REPLACE INTO lstm_hyperparams (" +
                    "symbol, window_size, lstm_neurons, dropout_rate, learning_rate, " +
                    "num_epochs, patience, min_delta, k_folds, optimizer, l1, l2, normalization_scope, " +
                    "normalization_method, swing_trade_type, num_layers, bidirectional, attention, features, horizon_bars, " +
                    "threshold_type, threshold_k, limit_prediction_pct, batch_size, cv_mode, use_scalar_v2, use_log_return_target, use_walk_forward_v2, " +
                    "walk_forward_splits, embargo_bars, seed, business_profit_factor_cap, business_drawdown_gamma, capital, risk_pct, sizing_k, fee_pct, slippage_pct, " +
                    "kl_drift_threshold, mean_shift_sigma_threshold, use_multi_horizon_avg, entry_threshold_factor, updated_date) " +
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)";

    /**
     * Requête de sélection d'un jeu d'hyperparamètres selon le symbole (clé).
     */
    private static final String SQL_LOAD_HYPERPARAMS =
            "SELECT * FROM lstm_hyperparams WHERE symbol = ?";

    /**
     * Requête d'insertion des métriques de tuning. Pas de REPLACE ici : on garde l'historique.
     * tested_date est auto-renseignée (CURRENT_TIMESTAMP).
     */
    private static final String SQL_SAVE_TUNING_METRICS =
            "INSERT INTO lstm_tuning_metrics (" +
                    "symbol, window_size, lstm_neurons, dropout_rate, learning_rate, l1, l2, num_epochs, patience, min_delta, optimizer, " +
                    "normalization_scope, normalization_method, swing_trade_type, features, mse, rmse, horizon_bars, " +
                    "profit_total, profit_factor, win_rate, max_drawdown, num_trades, business_score, sortino, calmar, turnover, avg_bars_in_position, use_multi_horizon_avg, entry_threshold_factor, tested_date" +
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)";


    // =========================================================
    // Constructeur
    // =========================================================

    /**
     * Injection du JdbcTemplate (Spring s'en charge automatiquement).
     * @param jdbcTemplate outil pour exécuter les requêtes SQL.
     */
    public LstmHyperparamsRepository(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    // =========================================================
    // Méthodes publiques (API du repository)
    // =========================================================

    /**
     * Sauvegarde (ou remplace) les hyperparamètres associés à un symbole.
     * REPLACE INTO => si la clé existe déjà, elle est remplacée (équivalent à un UPSERT).
     *
     * @param symbol symbole boursier (clé primaire attendue dans la table).
     * @param config objet contenant tous les hyperparamètres.
     */
    public void saveHyperparams(String symbol, LstmConfig config) {
        // Important : l'ordre des paramètres DOIT correspondre strictement
        // à l'ordre des colonnes dans SQL_SAVE_HYPERPARAMS ci-dessus.
        jdbcTemplate.update(
                SQL_SAVE_HYPERPARAMS,
                symbol,
                config.getWindowSize(),
                config.getLstmNeurons(),
                config.getDropoutRate(),
                config.getLearningRate(),
                config.getNumEpochs(),
                config.getPatience(),
                config.getMinDelta(),
                config.getKFolds(),
                config.getOptimizer(),
                config.getL1(),
                config.getL2(),
                config.getNormalizationScope(),
                config.getNormalizationMethod(),
                config.getSwingTradeType(),
                config.getNumLstmLayers(),
                config.isBidirectional(),
                config.isAttention(),
                gson.toJson(config.getFeatures()), // Conversion liste -> JSON
                config.getHorizonBars(),
                config.getThresholdType(),
                config.getThresholdK(),
                config.getLimitPredictionPct(),
                config.getBatchSize(),
                config.getCvMode(),
                config.isUseScalarV2(),
                config.isUseLogReturnTarget(),
                config.isUseWalkForwardV2(),
                config.getWalkForwardSplits(),
                config.getEmbargoBars(),
                config.getSeed(),
                config.getBusinessProfitFactorCap(),
                config.getBusinessDrawdownGamma(),
                config.getCapital(),
                config.getRiskPct(),
                config.getSizingK(),
                config.getFeePct(),
                config.getSlippagePct(),
                config.getKlDriftThreshold(),
                config.getMeanShiftSigmaThreshold(),
                config.isUseMultiHorizonAvg(),
                config.getEntryThresholdFactor()
        );
    }

    /**
     * Charge les hyperparamètres pour un symbole donné.
     * Retourne null si aucun enregistrement trouvé (le code appelant doit gérer ce cas).
     *
     * @param symbol symbole boursier.
     * @return LstmConfig ou null.
     */
    public LstmConfig loadHyperparams(String symbol) {
        // query(..., ResultSetExtractor, params...) :
        // rs.next() ? ... : null => on récupère la première ligne si disponible.
        return jdbcTemplate.query(
                SQL_LOAD_HYPERPARAMS,
                rs -> rs.next() ? mapRowToConfig(rs) : null,
                symbol
        );
    }

    /**
     * Enregistre une ligne de métriques de tuning (historisation).
     * On ne modifie jamais les anciennes lignes => utile pour analyser l'évolution.
     *
     * @param symbol symbole testé
     * @param config hyperparamètres utilisés lors du test (on snapshot certains champs)
     * @param mse erreur quadratique moyenne du modèle
     * @param rmse racine de l'erreur quadratique moyenne
     * @param profitTotal profit agrégé sur la simulation
     * @param profitFactor ratio (gains / pertes)
     * @param winRate pourcentage de trades gagnants
     * @param maxDrawdown drawdown maximum (risque)
     * @param numTrades nombre total de trades exécutés
     * @param businessScore score métier agrégé (logique externe)
     * @param sortino ratio Sortino
     * @param calmar ratio Calmar
     * @param turnover rotation du portefeuille
     * @param avgBarsInPosition durée moyenne en position (en barres)
     */
    public void saveTuningMetrics(
            String symbol,
            LstmConfig config,
            double mse,
            double rmse,
            double profitTotal,
            double profitFactor,
            double winRate,
            double maxDrawdown,
            int numTrades,
            double businessScore,
            double sortino,
            double calmar,
            double turnover,
            double avgBarsInPosition
    ) {
        jdbcTemplate.update(
                SQL_SAVE_TUNING_METRICS,
                symbol,
                config.getWindowSize(),
                config.getLstmNeurons(),
                config.getDropoutRate(),
                config.getLearningRate(),
                config.getL1(),
                config.getL2(),
                config.getNumEpochs(),
                config.getPatience(),
                config.getMinDelta(),
                config.getOptimizer(),
                config.getNormalizationScope(),
                config.getNormalizationMethod(),
                config.getSwingTradeType(),
                gson.toJson(config.getFeatures()),
                mse,
                rmse,
                config.getHorizonBars(),
                profitTotal,
                profitFactor,
                winRate,
                maxDrawdown,
                numTrades,
                businessScore,
                sortino,
                calmar,
                turnover,
                avgBarsInPosition,
                config.isUseMultiHorizonAvg(),
                config.getEntryThresholdFactor()
        );
    }

    // =========================================================
    // Méthode interne (mapping ResultSet -> objet métier)
    // =========================================================

    /**
     * Construit un objet LstmConfig à partir d'une ligne SQL.
     * Toute valeur potentiellement NULL en base est soit lue telle quelle,
     * soit remplacée par une valeur par défaut (ex : normalization_method -> "auto").
     *
     * Attention : si vous ajoutez une nouvelle colonne en base :
     * 1. Ajoutez-la dans la requête SQL (si nécessaire)
     * 2. Ajoutez le setXXX correspondant ici
     * 3. Testez soigneusement (risque de NullPointer ou d'ordre des paramètres)
     *
     * @param rs ligne courante du ResultSet
     * @return LstmConfig rempli
     * @throws SQLException en cas d'accès colonne incorrect
     */
    private LstmConfig mapRowToConfig(ResultSet rs) throws SQLException {
        LstmConfig config = new LstmConfig();

        // Extraction simple (types primitifs ou wrappers)
        config.setWindowSize(rs.getInt("window_size"));
        config.setLstmNeurons(rs.getInt("lstm_neurons"));
        config.setDropoutRate(rs.getDouble("dropout_rate"));
        config.setLearningRate(rs.getDouble("learning_rate"));
        config.setNumEpochs(rs.getInt("num_epochs"));
        config.setPatience(rs.getInt("patience"));
        config.setMinDelta(rs.getDouble("min_delta"));
        config.setKFolds(rs.getInt("k_folds"));
        config.setOptimizer(rs.getString("optimizer"));
        config.setL1(rs.getDouble("l1"));
        config.setL2(rs.getDouble("l2"));

        // num_layers en base -> numLstmLayers en objet
        config.setNumLstmLayers(rs.getInt("num_layers"));

        // Flags booléens
        config.setBidirectional(rs.getBoolean("bidirectional"));
        config.setAttention(rs.getBoolean("attention"));

        // Paramètres horizon / seuils
        config.setHorizonBars(rs.getInt("horizon_bars"));
        config.setThresholdK(rs.getDouble("threshold_k"));
        config.setThresholdType(rs.getString("threshold_type"));

        // Valeur limite sur prédiction (contrôle amplitude)
        config.setLimitPredictionPct(rs.getDouble("limit_prediction_pct"));

        // Normalisation (avec fallback si NULL)
        config.setNormalizationScope(rs.getString("normalization_scope"));
        config.setNormalizationMethod(
                rs.getString("normalization_method") != null
                        ? rs.getString("normalization_method")
                        : "auto"
        );

        // Type swing trade (fallback)
        config.setSwingTradeType(
                rs.getString("swing_trade_type") != null
                        ? rs.getString("swing_trade_type")
                        : "range"
        );

        // Batch / Cross-validation
        config.setBatchSize(rs.getInt("batch_size"));
        config.setCvMode(
                rs.getString("cv_mode") != null
                        ? rs.getString("cv_mode")
                        : "split"
        );

        // Options booléennes supplémentaires
        config.setUseScalarV2(rs.getBoolean("use_scalar_v2"));
        config.setUseLogReturnTarget(rs.getBoolean("use_log_return_target"));
        config.setUseWalkForwardV2(rs.getBoolean("use_walk_forward_v2"));

        // Walk-forward / embargo
        config.setWalkForwardSplits(rs.getInt("walk_forward_splits"));
        config.setEmbargoBars(rs.getInt("embargo_bars"));

        // Reproductibilité
        config.setSeed(rs.getLong("seed"));

        // Paramètres business & risque
        config.setBusinessProfitFactorCap(rs.getDouble("business_profit_factor_cap"));
        config.setBusinessDrawdownGamma(rs.getDouble("business_drawdown_gamma"));
        config.setCapital(rs.getDouble("capital"));
        config.setRiskPct(rs.getDouble("risk_pct"));
        config.setSizingK(rs.getDouble("sizing_k"));
        config.setFeePct(rs.getDouble("fee_pct"));
        config.setSlippagePct(rs.getDouble("slippage_pct"));

        // Détection dérive / drift
        config.setKlDriftThreshold(rs.getDouble("kl_drift_threshold"));
        config.setMeanShiftSigmaThreshold(rs.getDouble("mean_shift_sigma_threshold"));

        // Champs ajoutés : useMultiHorizonAvg et entryThresholdFactor
        try { config.setUseMultiHorizonAvg(rs.getBoolean("use_multi_horizon_avg")); } catch (Exception ignored) {}
        try { config.setEntryThresholdFactor(rs.getDouble("entry_threshold_factor")); } catch (Exception ignored) {}

        // Désérialisation des features si non null
        String featuresJson = rs.getString("features");
        if (featuresJson != null) {
            config.setFeatures(
                    gson.fromJson(
                            featuresJson,
                            new TypeToken<List<String>>() {}.getType()
                    )
            );
        }

        return config;
    }

}
