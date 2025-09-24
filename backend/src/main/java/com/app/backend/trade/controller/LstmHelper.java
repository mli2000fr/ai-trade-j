package com.app.backend.trade.controller;


import com.app.backend.trade.lstm.LstmConfig;
import com.app.backend.trade.lstm.LstmTradePredictor;
import com.app.backend.trade.lstm.LstmTuningService;
import com.app.backend.trade.model.*;
import com.app.backend.trade.util.TradeUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.ta4j.core.BarSeries;
import java.io.IOException;
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
        PreditLsdm preditLsdmDb = this.getPreditFromDB(symbol);
        if(preditLsdmDb != null){
            return preditLsdmDb;
        }

        LstmConfig config = lstmTuningService.hyperparamsRepository.loadHyperparams(symbol);
        if (config == null) {
            logger.info("Hyperparamètres existants trouvés pour {}. Ignorer le tuning.", symbol);
            return PreditLsdm.builder().lastClose(0).predictedClose(0).signal(SignalType.NONE).position("").lastDate("").build();
        }
        MultiLayerNetwork model = lstmTradePredictor.loadModelFromDb(symbol, jdbcTemplate);

        BarSeries series = getBarBySymbol(symbol, null);
        PreditLsdm preditLsdm = lstmTradePredictor.getPredit(symbol, series, config, model);
        saveSignalHistory(symbol, preditLsdm);
        return preditLsdm;
    }
    public void saveSignalHistory(String symbol, PreditLsdm preditLsdm) {
        java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(java.time.LocalDate.now());
        String insertSql = "INSERT INTO signal_lstm (symbol, signal_lstm, price_lstm, price_clo, position_lstm, lstm_created_at) VALUES (?, ?, ?, ?, ?, ?)";
        jdbcTemplate.update(insertSql,
                symbol,
                preditLsdm.getSignal().name(),
                preditLsdm.getPredictedClose(),
                preditLsdm.getLastClose(),
                preditLsdm.getPosition(),
                java.sql.Date.valueOf(lastTradingDay));
    }
    public PreditLsdm getPreditFromDB(String symbol) {
        String sql = "SELECT * FROM signal_lstm WHERE symbol = ? ORDER BY lstm_created_at DESC LIMIT 1";
        try {
            return jdbcTemplate.query(sql, ps -> ps.setString(1, symbol), rs -> {
                if (rs.next()) {
                    String signalStr = rs.getString("signal_lstm");
                    double priceLstm = rs.getDouble("price_lstm");
                    double priceClo = rs.getDouble("price_clo");
                    String positionLstm = rs.getString("position_lstm");
                    java.sql.Date lastDate = rs.getDate("lstm_created_at");
                    SignalType type;
                    try {
                        type = SignalType.valueOf(signalStr);
                    } catch (Exception e) {
                        logger.warn("SignalType inconnu en base: {}", signalStr);
                        type = null;
                    }

                    java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(java.time.LocalDate.now());
                    java.time.LocalDate lastKnown = lastDate.toLocalDate();
                    // Si la dernière date connue est le dernier jour de cotation, la base est à jour
                    if (lastKnown.isEqual(lastTradingDay) || lastKnown.isAfter(lastTradingDay)) {
                        String dateSavedStr = lastKnown.format(java.time.format.DateTimeFormatter.ofPattern("dd/MM"));
                        return PreditLsdm.builder().signal(type).lastClose(priceClo).lastDate(dateSavedStr).predictedClose(priceLstm).position(positionLstm).build();
                    }else{
                        return null;
                    }
                }
                return null;
            });
        } catch (Exception e) {
            logger.warn("Erreur SQL getPreditFromDB pour {}: {}", symbol, e.getMessage());
            return null;
        }
    }


    // Méthode existante conservée pour compatibilité
    public void tuneAllSymbols() {
        tuneAllSymbols(true, 5);
    }
    /**
     * Lance le tuning automatique pour une liste de symboles.
     * Les résultats sont loggés et la meilleure config est sauvegardée pour chaque symbole.
     *
     * @param useRandomGrid  true pour utiliser une grille aléatoire (random search), false pour grid search complet
     * @param randomGridSize nombre de configurations aléatoires à tester (si useRandomGrid=true)
     */
    public void tuneAllSymbols(boolean useRandomGrid, int randomGridSize) {
        List<String> symbols = getSymbolFitredFromTabSingle("score_swing_trade");
        List<LstmConfig> grid;
        if (useRandomGrid) {
            grid = lstmTuningService.generateRandomSwingTradeGrid(randomGridSize);
        } else {
            grid = lstmTuningService.generateSwingTradeGrid();
        }
        lstmTuningService.tuneAllSymbolsMultiThread(symbols, grid, jdbcTemplate, symbol -> getBarBySymbol(symbol, null));
    }

    public void tuneAllSymbolsBis() {
        tuneAllSymbolsBis(true, 1);
    }
    public void tuneAllSymbolsBis(boolean useRandomGrid, int randomGridSize) {
        List<String> symbols = getSymbolFitredFromTabSingle("score_swing_trade");
        List<LstmConfig> grid;
        if (useRandomGrid) {
            grid = lstmTuningService.generateRandomSwingTradeGrid(randomGridSize);
        } else {
            grid = lstmTuningService.generateSwingTradeGrid();
        }
        lstmTuningService.tuneAllSymbols(symbols, grid, jdbcTemplate, symbol -> getBarBySymbol(symbol, null));
    }


    public List<String> getSymbolFitredFromTabSingle(String sort) {
        String orderBy = sort == null ? "score_swing_trade" : sort;
        String sql = "select symbol from best_in_out_single_strategy";
        sql += " ORDER BY " + orderBy + " DESC";
        return jdbcTemplate.queryForList(sql, String.class);
    }


    public List<LstmTuningService.TuningExceptionReportEntry> getTuningExceptionReport() {
        return lstmTuningService.getTuningExceptionReport();
    }

}
