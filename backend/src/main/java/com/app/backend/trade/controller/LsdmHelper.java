package com.app.backend.trade.controller;


import com.app.backend.trade.lstm.LstmConfig;
import com.app.backend.trade.lstm.LstmTradePredictor;
import com.app.backend.trade.model.DailyValue;
import com.app.backend.trade.model.PreditLsdm;
import com.app.backend.trade.util.TradeUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Controller;
import org.ta4j.core.BarSeries;

import java.util.Collections;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Controller
public class LsdmHelper {


    private JdbcTemplate jdbcTemplate;
    private LstmTradePredictor lstmTradePredictor;

    private static final Logger logger = LoggerFactory.getLogger(LsdmHelper.class);

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
    public void trainLstm(String symbol) {
        BarSeries series = getBarBySymbol(symbol, null);
        lstmTradePredictor.initWithConfig(new LstmConfig());
        MultiLayerNetwork model = lstmTradePredictor.trainLstm(series);
        // Sauvegarder le modèle dans la base après entraînement
        try {
            lstmTradePredictor.saveModelToDb(symbol, model, jdbcTemplate);
        } catch (Exception e) {
            logger.error("Erreur lors de la sauvegarde du modèle : {}", e.getMessage());
        }
    }

    // Prédiction LSTM
    public PreditLsdm getPredit(String symbol) {
        boolean modelLoaded;
        try {
            lstmTradePredictor.loadModelFromDb(symbol, jdbcTemplate);
            modelLoaded = true;
        } catch (Exception e) {
            modelLoaded = false;
        }
        BarSeries series = getBarBySymbol(symbol, null);
        if (!modelLoaded) {
            lstmTradePredictor.initWithConfig(new LstmConfig());
            MultiLayerNetwork model = lstmTradePredictor.trainLstm(series);
            try {
                lstmTradePredictor.saveModelToDb(symbol, model, jdbcTemplate);
            } catch (Exception ex) {
                logger.error("Erreur lors de la sauvegarde du modèle : {}", ex.getMessage());
            }
        }
        return lstmTradePredictor.getPredit(series);
    }

    /**
     * Lance une validation croisée k-fold sur le modèle LSTM pour un symbole donné.
     * Les résultats sont loggés (voir app.log).
     */
    public void crossValidateLstm(String symbol, int windowSize, int numEpochs, int kFolds, int lstmNeurons, double dropoutRate, int patience, double minDelta, double learningRate, String optimizer) {
        BarSeries series = getBarBySymbol(symbol, null);
        lstmTradePredictor.crossValidateLstm(series, windowSize, numEpochs, kFolds, lstmNeurons, dropoutRate, patience, minDelta, learningRate, optimizer);
    }
}
