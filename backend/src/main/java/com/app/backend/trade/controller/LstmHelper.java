package com.app.backend.trade.controller;


import com.app.backend.trade.lstm.LstmConfig;
import com.app.backend.trade.lstm.LstmTradePredictor;
import com.app.backend.trade.lstm.LstmTuningService;
import com.app.backend.trade.model.*;
import com.app.backend.trade.util.TradeUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Controller;
import org.ta4j.core.BarSeries;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Controller
public class LstmHelper {


    private final JdbcTemplate jdbcTemplate;
    private final LstmTradePredictor lstmTradePredictor;
    private final LstmTuningService lstmTuningService;

    private static final Logger logger = LoggerFactory.getLogger(LstmHelper.class);

    // File centralisée des rapports de drift
    private final java.util.concurrent.ConcurrentLinkedQueue<LstmTradePredictor.DriftReportEntry> driftReports = new java.util.concurrent.ConcurrentLinkedQueue<>();

    public LstmHelper(JdbcTemplate jdbcTemplate, LstmTradePredictor lstmTradePredictor, LstmTuningService lstmTuningService) {
        this.jdbcTemplate = jdbcTemplate;
        this.lstmTradePredictor = lstmTradePredictor;
        this.lstmTuningService = lstmTuningService;
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
        // Charger modèle + scalers
        LstmTradePredictor.LoadedModel loaded = null;
        try {
            loaded = lstmTradePredictor.loadModelAndScalersFromDb(symbol, jdbcTemplate);
        } catch (Exception e) {
            logger.warn("Impossible de charger le modèle/scalers depuis la base : {}", e.getMessage());
        }
        MultiLayerNetwork model = loaded != null ? loaded.model : null;
        LstmTradePredictor.ScalerSet scalers = loaded != null ? loaded.scalers : null;
        BarSeries series = getBarBySymbol(symbol, null);
        // Vérification drift et retrain éventuel + reporting détaillé
        if (model != null && scalers != null) {
            try {
                java.util.List<LstmTradePredictor.DriftReportEntry> reports = lstmTradePredictor.checkDriftAndRetrainWithReport(series, config, model, scalers, symbol);
                if(!reports.isEmpty()){
                    for(LstmTradePredictor.DriftReportEntry r : reports){
                        driftReports.add(r);
                        // Insertion BD best-effort (table optionnelle lstm_drift_report)
                        try {
                            String sql = "INSERT INTO lstm_drift_report (event_date, symbol, feature, drift_type, kl, mean_shift, mse_before, mse_after, retrained) VALUES (?,?,?,?,?,?,?,?,?)";
                            jdbcTemplate.update(sql,
                                java.sql.Timestamp.from(r.eventDate),
                                r.symbol,
                                r.feature,
                                r.driftType,
                                r.kl,
                                r.meanShift,
                                r.mseBefore,
                                r.mseAfter,
                                r.retrained
                            );
                        } catch (Exception ex){
                            logger.debug("Table lstm_drift_report absente ou insertion échouée: {}", ex.getMessage());
                        }
                        logger.info("[DRIFT-REPORT] symbol={} feature={} type={} kl={} meanShift={}σ mseBefore={} mseAfter={} retrained={}", r.symbol, r.feature, r.driftType, r.kl, r.meanShift, r.mseBefore, r.mseAfter, r.retrained);
                    }
                    // Sauvegarder le modèle mis à jour si retrain effectué (au moins un retrained true)
                    boolean retrained = reports.stream().anyMatch(rr -> rr.retrained);
                    if(retrained){
                        try {
                            lstmTradePredictor.saveModelToDb(symbol, model, jdbcTemplate, config, scalers);
                        } catch (Exception e) {
                            logger.warn("Erreur lors de la sauvegarde du modèle/scalers après retrain : {}", e.getMessage());
                        }
                    }
                }
            } catch (Exception e){
                logger.warn("Erreur pendant le drift reporting: {}", e.getMessage());
            }
        }
        PreditLsdm preditLsdm = lstmTradePredictor.getPredit(symbol, series, config, model, scalers);
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
    public List<String> getSymbolFitredFromTabSingle(String sort) {
        String orderBy = sort == null ? "score_swing_trade" : sort;
        String sql = "select symbol from best_in_out_single_strategy s where s.avg_pnl > 0 AND s.profit_factor > 1 AND s.win_rate > 0.5 AND s.max_drawdown < 0.2 AND s.sharpe_ratio > 1 AND s.rendement > 0.05";
        sql += " ORDER BY " + orderBy + " DESC";
        return jdbcTemplate.queryForList(sql, String.class);
    }

    // Méthode existante conservée pour compatibilité
    public void tuneAllSymbols() {
        tuneAllSymbols(true, 60);
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
        int[] horizonBars = {3, 5, 10};
        int[] numLstmLayers = {1, 2, 3};
        int[] batchSizes = {64, 128, 256};
        boolean[] bidirectionals = {true, false};
        boolean[] attentions = {true, false};
        for (String symbol : symbols) {
            List<String> features = getFeaturesForSymbol(symbol);
            List<LstmConfig> grid;
            if (useRandomGrid) {
                grid = lstmTuningService.generateRandomSwingTradeGrid(
                    randomGridSize, features, horizonBars, numLstmLayers, batchSizes, bidirectionals, attentions
                );
            } else {
                grid = lstmTuningService.generateSwingTradeGrid(
                    features, horizonBars, numLstmLayers, batchSizes, bidirectionals, attentions
                );
            }
            lstmTuningService.tuneAllSymbols(Collections.singletonList(symbol), grid, jdbcTemplate, s -> getBarBySymbol(s, null));
        }
    }

    /**
     * Retourne la liste des features à utiliser pour un symbole donné.
     * Peut être personnalisée selon le secteur, le marché ou le symbole.
     */
    public List<String> getFeaturesForSymbol(String symbol) {
        if (symbol.startsWith("BTC") || symbol.startsWith("ETH")) {
            return Arrays.asList("close", "volume", "rsi", "macd", "atr", "momentum", "day_of_week", "month");
        } else if (symbol.startsWith("CAC") || symbol.startsWith("SPX")) {
            return Arrays.asList("close", "sma", "ema", "rsi", "atr", "bollinger_high", "bollinger_low", "month");
        } else if (symbol.startsWith("AAPL") || symbol.startsWith("MSFT")) {
            return Arrays.asList("close", "volume", "rsi", "sma", "ema", "macd", "atr", "bollinger_high", "bollinger_low", "stochastic", "cci", "momentum", "day_of_week", "month");
        }
        return Arrays.asList("close", "volume", "rsi", "sma", "ema", "macd", "atr", "bollinger_high", "bollinger_low", "stochastic", "cci", "momentum", "day_of_week", "month");
    }


    public List<LstmTuningService.TuningExceptionReportEntry> getTuningExceptionReport() {
        return lstmTuningService.getTuningExceptionReport();
    }

    // Accès aux rapports de drift en mémoire
    public java.util.List<LstmTradePredictor.DriftReportEntry> getDriftReports(){
        return new java.util.ArrayList<>(driftReports);
    }

}
