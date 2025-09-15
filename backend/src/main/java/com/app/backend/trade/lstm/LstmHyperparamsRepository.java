package com.app.backend.trade.lstm;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;
import java.sql.ResultSet;
import java.sql.SQLException;

@Repository
public class LstmHyperparamsRepository {
    private final JdbcTemplate jdbcTemplate;

    public LstmHyperparamsRepository(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    public void saveHyperparams(String symbol, LstmConfig config) {
        String sql = "REPLACE INTO lstm_hyperparams (symbol, window_size, lstm_neurons, dropout_rate, learning_rate, num_epochs, patience, min_delta, k_folds, optimizer, l1, l2, updated_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)";
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
            config.getL2()
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
        return config;
    }
}

