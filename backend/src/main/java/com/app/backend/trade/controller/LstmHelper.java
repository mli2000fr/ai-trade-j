package com.app.backend.trade.controller;


import com.app.backend.trade.lstm.LstmConfig;
import com.app.backend.trade.lstm.LstmTradePredictor;
import com.app.backend.trade.lstm.LstmTuningService;
import com.app.backend.trade.model.DailyValue;
import com.app.backend.trade.model.PreditLsdm;
import com.app.backend.trade.model.SignalType;
import com.app.backend.trade.util.TradeUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Controller;
import org.ta4j.core.BarSeries;

import java.io.IOException;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Controller
public class LstmHelper {


    private JdbcTemplate jdbcTemplate;
    private LstmTradePredictor lstmTradePredictor;

    private static final Logger logger = LoggerFactory.getLogger(LstmHelper.class);

    @Autowired
    public LstmHelper(JdbcTemplate jdbcTemplate, LstmTradePredictor lstmTradePredictor) {
        this.jdbcTemplate = jdbcTemplate;
        this.lstmTradePredictor = lstmTradePredictor;
    }

    @Autowired
    private LstmTuningService lstmTuningService;

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
        boolean useRandomGrid = true;
        List<LstmConfig> grid;
        if (useRandomGrid) {
            grid = lstmTuningService.generateRandomSwingTradeGrid(10);
        } else {
            grid = lstmTuningService.generateSwingTradeGrid();
        }
        lstmTuningService.tuneAllSymbols(Arrays.asList(symbol), grid, jdbcTemplate, sym -> getBarBySymbol(sym, null));
    }

    // Prédiction LSTM
    public PreditLsdm getPredit(String symbol) throws IOException {
        LstmConfig config = lstmTuningService.hyperparamsRepository.loadHyperparams(symbol);
        if(config == null){
            logger.info("Hyperparamètres existants trouvés pour {}. Ignorer le tuning.", symbol);
            return PreditLsdm.builder().lastClose(0).predictedClose(0).signal(SignalType.NONE).position("").lastDate("").build();
        }
        MultiLayerNetwork model = lstmTradePredictor.loadModelFromDb(symbol, jdbcTemplate);

        BarSeries series = getBarBySymbol(symbol, null);
        return lstmTradePredictor.getPredit(symbol, series, config, model);
    }

    // Entraînement LSTM avec personnalisation des features
    public void trainLstm(String symbol, List<String> features) {
        BarSeries series = getBarBySymbol(symbol, null);
        LstmConfig config = new LstmConfig();
        if (features != null && !features.isEmpty()) {
            config.setFeatures(features);
        }
        MultiLayerNetwork model = lstmTradePredictor.initModel(
            config.getWindowSize(),
            1,
            config.getLstmNeurons(),
            config.getDropoutRate(),
            config.getLearningRate(),
            config.getOptimizer(),
            config.getL1(),
            config.getL2()
        );
        model = lstmTradePredictor.trainLstm(series, config, model);
        try {
            lstmTradePredictor.saveModelToDb(symbol, model, jdbcTemplate, config);
        } catch (Exception e) {
            logger.error("Erreur lors de la sauvegarde du modèle : {}", e.getMessage());
        }
    }

    // Prédiction LSTM avec personnalisation des features
    public PreditLsdm getPredit(String symbol, List<String> features) throws IOException {
        LstmConfig config = lstmTuningService.tuneSymbol(symbol, lstmTuningService.generateSwingTradeGrid(), getBarBySymbol(symbol, null), jdbcTemplate);
        if (features != null && !features.isEmpty()) {
            config.setFeatures(features);
        }
        MultiLayerNetwork model = lstmTradePredictor.loadModelFromDb(symbol, jdbcTemplate);
        BarSeries series = getBarBySymbol(symbol, null);
        return lstmTradePredictor.getPredit(symbol, series, config, model);
    }

    /**
     * Lance une validation croisée k-fold sur le modèle LSTM pour un symbole donné.
     * Les résultats sont loggés (voir app.log).
     */
    public void crossValidateLstm(String symbol, int windowSize, int numEpochs, int kFolds, int lstmNeurons, double dropoutRate, int patience, double minDelta, double learningRate, String optimizer) {
        BarSeries series = getBarBySymbol(symbol, null);
        LstmConfig config = new LstmConfig();
        config.setWindowSize(windowSize);
        config.setNumEpochs(numEpochs);
        config.setKFolds(kFolds);
        config.setLstmNeurons(lstmNeurons);
        config.setDropoutRate(dropoutRate);
        config.setPatience(patience);
        config.setMinDelta(minDelta);
        config.setLearningRate(learningRate);
        config.setOptimizer(optimizer);
        lstmTradePredictor.crossValidateLstm(series, config);
    }

    /**
     * Lance le tuning automatique pour une liste de symboles.
     * Les résultats sont loggés et la meilleure config est sauvegardée pour chaque symbole.
     * @param useRandomGrid true pour utiliser une grille aléatoire (random search), false pour grid search complet
     * @param randomGridSize nombre de configurations aléatoires à tester (si useRandomGrid=true)
     */
    public void tuneAllSymbols(boolean useRandomGrid, int randomGridSize) {
        List<String> symbols = getSymbolFitredFromTabSingle("rendement_score");
        List<LstmConfig> grid;
        if (useRandomGrid) {
            grid = lstmTuningService.generateRandomSwingTradeGrid(randomGridSize);
        } else {
            grid = lstmTuningService.generateSwingTradeGrid();
        }
        lstmTuningService.tuneAllSymbols(symbols, grid, jdbcTemplate, symbol -> getBarBySymbol(symbol, null));
    }

    // Méthode existante conservée pour compatibilité
    public void tuneAllSymbols() {
        tuneAllSymbols(true, 10);
    }

    public List<String> getSymbolFitredFromTabSingle(String sort) {
        String orderBy = sort == null ? "rendement_score" : sort;
        String sql = "select symbol from trade_ai.best_in_out_single_strategy where fltred_out = 'false'";
        sql += " ORDER BY " + orderBy + " DESC";
        return jdbcTemplate.queryForList(sql, String.class);
    }
}
