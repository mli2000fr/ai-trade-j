package com.app.backend.trade.controller;


import com.app.backend.trade.lstm.LstmTradePredictor;
import com.app.backend.trade.model.DailyValue;
import com.app.backend.trade.util.TradeUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Controller;
import org.ta4j.core.BarSeries;

import java.util.Collections;
import java.util.List;

@Controller
public class LsdmHelper {


    private JdbcTemplate jdbcTemplate;
    private LstmTradePredictor lstmTradePredictor;

    @Autowired
    public LsdmHelper(JdbcTemplate jdbcTemplate, LstmTradePredictor lstmTradePredictor) {
        this.jdbcTemplate = jdbcTemplate;
        this.lstmTradePredictor = lstmTradePredictor;
    }



    public BarSeries getBarBySymbol(String symbol, Integer limit) {
        String sql = "SELECT date, open, high, low, close, volume, number_of_trades, volume_weighted_average_price " +
                "FROM daily_value WHERE symbol = ? ORDER BY date ASC";
        if (limit != null && limit > 0) {
            sql = "SELECT date, open, high, low, close, volume, number_of_trades, volume_weighted_average_price " +
                    "FROM daily_value WHERE symbol = ? ORDER BY date DESC LIMIT " + limit;
        }
        List<DailyValue> results = jdbcTemplate.query(sql, new Object[]{symbol}, (rs, rowNum) -> {
            return DailyValue.builder()
                    .date(rs.getDate("date").toString())
                    .open(rs.getString("open"))
                    .high(rs.getString("high"))
                    .low(rs.getString("low"))
                    .close(rs.getString("close"))
                    .volume(rs.getString("volume"))
                    .numberOfTrades(rs.getString("number_of_trades"))
                    .volumeWeightedAveragePrice(rs.getString("volume_weighted_average_price"))
                    .build();
        });

        // Inverser la liste pour avoir les dates en ordre croissant
        if (limit != null && limit > 0) {
            Collections.reverse(results);
        }
        return TradeUtils.mapping(results);
    }

    // Entraînement LSTM
    public void trainLstm(String symbol, int windowSize, int numEpochs) {
        BarSeries series = getBarBySymbol(symbol, null);
        lstmTradePredictor.initModel(windowSize, 1);
        lstmTradePredictor.trainLstm(series, windowSize, numEpochs);
        // Sauvegarder le modèle après entraînement
        try {
            String modelPath = "models/lstm_model_" + symbol + ".zip";
            lstmTradePredictor.saveModel(modelPath);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Prédiction LSTM
    public double predictNextClose(String symbol, int windowSize) {
        String modelPath = "models/lstm_model_" + symbol + ".zip";
        boolean modelLoaded = false;
        try {
            lstmTradePredictor.loadModel(modelPath);
            modelLoaded = true;
        } catch (Exception e) {
            // Le modèle n'existe pas ou erreur de chargement
            modelLoaded = false;
        }
        BarSeries series = getBarBySymbol(symbol, null);
        if (!modelLoaded) {
            // Entraînement automatique si le modèle n'existe pas
            lstmTradePredictor.initModel(windowSize, 1);
            lstmTradePredictor.trainLstm(series, windowSize, 10); // 10 epochs par défaut
            try {
                lstmTradePredictor.saveModel(modelPath);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return lstmTradePredictor.predictNextClose(series, windowSize);
    }
}
