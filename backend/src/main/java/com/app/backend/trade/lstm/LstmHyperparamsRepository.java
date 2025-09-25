package com.app.backend.trade.lstm;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;

@Repository
public class LstmHyperparamsRepository {
    private final JdbcTemplate jdbcTemplate;
    private final Gson gson = new Gson();

    public LstmHyperparamsRepository(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    public void saveHyperparams(String symbol, LstmConfig config) {
        String sql = "REPLACE INTO lstm_hyperparams (" +
                "symbol, window_size, lstm_neurons, dropout_rate, learning_rate, " +
                "num_epochs, patience, min_delta, k_folds, optimizer, l1, l2, normalization_scope, " +
                "normalization_method, swing_trade_type, num_layers, bidirectional, attention, features, horizon_bars, " +
                "threshold_type, threshold_k, limit_prediction_pct, batch_size, cv_mode, use_scalar_v2, use_log_return_target, use_walk_forward_v2, " +
                "walk_forward_splits, embargo_bars, seed, business_profit_factor_cap, business_drawdown_gamma, capital, risk_pct, sizing_k, fee_pct, slippage_pct, " +
                "kl_drift_threshold, mean_shift_sigma_threshold, updated_date) " +
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)";

        jdbcTemplate.update(sql,
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
            gson.toJson(config.getFeatures()),
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
            config.getMeanShiftSigmaThreshold()
        );
    }

    public LstmConfig loadHyperparams(String symbol) {
        String sql = "SELECT * FROM lstm_hyperparams WHERE symbol = ?";
        return jdbcTemplate.query(sql, rs -> rs.next() ? mapRowToConfig(rs) : null, symbol);
    }

    private LstmConfig mapRowToConfig(ResultSet rs) throws SQLException {
        LstmConfig config = new LstmConfig();
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
        config.setNumLstmLayers(rs.getInt("num_layers")); // colonne = num_layers
        config.setBidirectional(rs.getBoolean("bidirectional"));
        config.setAttention(rs.getBoolean("attention"));
        config.setHorizonBars(rs.getInt("horizon_bars"));
        config.setThresholdK(rs.getDouble("threshold_k"));
        config.setThresholdType(rs.getString("threshold_type"));
        config.setLimitPredictionPct(rs.getDouble("limit_prediction_pct"));
        config.setNormalizationScope(rs.getString("normalization_scope"));
        config.setNormalizationMethod(rs.getString("normalization_method") != null ? rs.getString("normalization_method") : "auto");
        config.setSwingTradeType(rs.getString("swing_trade_type") != null ? rs.getString("swing_trade_type") : "range");
        config.setBatchSize(rs.getInt("batch_size"));
        config.setCvMode(rs.getString("cv_mode") != null ? rs.getString("cv_mode") : "split");
        config.setUseScalarV2(rs.getBoolean("use_scalar_v2"));
        config.setUseLogReturnTarget(rs.getBoolean("use_log_return_target"));
        config.setUseWalkForwardV2(rs.getBoolean("use_walk_forward_v2"));
        config.setWalkForwardSplits(rs.getInt("walk_forward_splits"));
        config.setEmbargoBars(rs.getInt("embargo_bars"));
        config.setSeed(rs.getLong("seed"));
        config.setBusinessProfitFactorCap(rs.getDouble("business_profit_factor_cap"));
        config.setBusinessDrawdownGamma(rs.getDouble("business_drawdown_gamma"));
        config.setCapital(rs.getDouble("capital"));
        config.setRiskPct(rs.getDouble("risk_pct"));
        config.setSizingK(rs.getDouble("sizing_k"));
        config.setFeePct(rs.getDouble("fee_pct"));
        config.setSlippagePct(rs.getDouble("slippage_pct"));
        config.setKlDriftThreshold(rs.getDouble("kl_drift_threshold"));
        config.setMeanShiftSigmaThreshold(rs.getDouble("mean_shift_sigma_threshold"));
        String featuresJson = rs.getString("features");
        if (featuresJson != null) {
            config.setFeatures(gson.fromJson(featuresJson, new TypeToken<List<String>>() {}.getType()));
        }
        return config;
    }

    // Sauvegarde des métriques de tuning étendue avec Sortino, Calmar, turnover, avgBarsInPosition
    public void saveTuningMetrics(String symbol, LstmConfig config, double mse, double rmse, String direction,
                                  double profitTotal, double profitFactor, double winRate, double maxDrawdown, int numTrades, double businessScore,
                                  double sortino, double calmar, double turnover, double avgBarsInPosition) {
        String sql = "INSERT INTO lstm_tuning_metrics (symbol, window_size, lstm_neurons, dropout_rate, learning_rate, l1, l2, num_epochs, patience, min_delta, optimizer, normalization_scope, normalization_method, swing_trade_type, features, mse, rmse, direction, horizon_bars, profit_total, profit_factor, win_rate, max_drawdown, num_trades, business_score, sortino, calmar, turnover, avg_bars_in_position, tested_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)";
        jdbcTemplate.update(sql,
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
            direction,
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
            avgBarsInPosition
        );
    }

}
