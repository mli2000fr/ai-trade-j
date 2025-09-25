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

    public LstmHyperparamsRepository(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    public void saveHyperparams(String symbol, LstmConfig config) {
        String sql = "REPLACE INTO lstm_hyperparams (symbol, window_size, lstm_neurons, dropout_rate, learning_rate, " +
                "num_epochs, patience, min_delta, k_folds, optimizer, l1, l2, normalization_scope, " +
                "normalization_method, swing_trade_type, num_layers, bidirectional, attention, features, " +
                "threshold_type, threshold_k, limit_prediction_pct, updated_date) " +
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)";
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
            new Gson().toJson(config.getFeatures()),
            config.getThresholdType(),
            config.getThresholdK(),
            config.getLimitPredictionPct()
        );
    }

    public LstmConfig loadHyperparams(String symbol) {
        String sql = "SELECT * FROM lstm_hyperparams WHERE symbol = ?";
        return jdbcTemplate.query(sql, rs -> {
            if (rs.next()) {
                return mapRowToConfig(rs);
            } else {
                return null;
            }
        }, symbol);
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
        config.setNumLstmLayers(rs.getInt("num_layers"));
        config.setBidirectional(rs.getBoolean("bidirectional"));
        config.setAttention(rs.getBoolean("attention"));
        config.setHorizonBars(rs.getInt("horizon_bars"));
        config.setThresholdK(rs.getDouble("threshold_k"));
        config.setThresholdType(rs.getString("threshold_type"));
        config.setHorizonBars(rs.getInt("horizon_bars"));
        config.setLimitPredictionPct(rs.getDouble("limit_prediction_pct"));
        config.setNormalizationScope(rs.getString("normalization_scope"));
        config.setNormalizationMethod(rs.getString("normalization_method") != null ? rs.getString("normalization_method") : "auto");
        config.setSwingTradeType(rs.getString("swing_trade_type") != null ? rs.getString("swing_trade_type") : "range");
        config.setFeatures(rs.getString("features") != null ? new Gson().fromJson(rs.getString("features"),new TypeToken<List<String>>() {}.getType()) : null);
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
            new com.google.gson.Gson().toJson(config.getFeatures()),
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
