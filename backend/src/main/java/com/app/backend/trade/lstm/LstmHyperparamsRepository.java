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
        String sql = "REPLACE INTO lstm_hyperparams (symbol, window_size, lstm_neurons, dropout_rate, learning_rate, num_epochs, patience, min_delta, k_folds, optimizer, l1, l2, normalization_scope, normalization_method, swing_trade_type, features, updated_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)";
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
            new Gson().toJson(config.getFeatures())
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
        config.setNormalizationScope(rs.getString("normalization_scope"));
        config.setNormalizationMethod(rs.getString("normalization_method") != null ? rs.getString("normalization_method") : "auto");
        config.setSwingTradeType(rs.getString("swing_trade_type") != null ? rs.getString("swing_trade_type") : "range");
        config.setSwingTradeType(rs.getString("features") != null ? new Gson().fromJson(rs.getString("features"),new TypeToken<List<String>>() {}.getType()) : null);
        return config;
    }
}
