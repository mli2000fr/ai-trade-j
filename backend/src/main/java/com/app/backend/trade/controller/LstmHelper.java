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
                config.getL2(),
                config
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
        LstmConfig config = lstmTuningService.tuneSymbol(symbol, lstmTuningService.generateSwingTradeGrid(), getBarBySymbol(symbol, null), jdbcTemplate, true);
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

    // Méthode existante conservée pour compatibilité
    public void tuneAllSymbols() {
        tuneAllSymbols(true, 5);
    }

    public List<String> getSymbolFitredFromTabSingle(String sort) {
        String orderBy = sort == null ? "score_swing_trade" : sort;
        String sql = "select symbol from best_in_out_single_strategy s where s.avg_pnl > 0 AND s.profit_factor > 1 AND s.win_rate > 0.5 AND s.max_drawdown < 0.2 AND s.sharpe_ratio > 1 AND s.rendement > 0.05";
        sql += " ORDER BY " + orderBy + " DESC";
        return jdbcTemplate.queryForList(sql, String.class);
    }

    /**
     * Exporte les métriques tuning LSTM au format CSV pour un symbole ou tous les symboles.
     *
     * @param symbol     symbole à exporter (null pour tous)
     * @param outputPath chemin du fichier CSV à générer
     * @return chemin du fichier généré ou null si aucun résultat
     */
    public String exportTuningMetricsToCsv(String symbol, String outputPath) {
        return lstmTuningService.hyperparamsRepository.exportTuningMetricsToCsv(symbol, outputPath);
    }

    /**
     * Backtest LSTM sur toute la série historique.
     * Applique le modèle LSTM à chaque bar, génère les signaux et simule les trades.
     * Retourne un objet RiskResult avec les métriques de performance.
     */
    public RiskResult backtestLstm(String symbol, double initialCapital, double riskPerTrade, double stopLossPct, double takeProfitPct) throws IOException {
        LstmConfig config = lstmTuningService.hyperparamsRepository.loadHyperparams(symbol);
        if (config == null) {
            logger.info("Aucun hyperparamètre LSTM pour {}.", symbol);
            return RiskResult.builder().rendement(0).tradeCount(0).winRate(0).maxDrawdown(0).avgPnL(0).profitFactor(0).avgTradeBars(0).maxTradeGain(0).maxTradeLoss(0).scoreSwingTrade(0).sharpeRatio(0).stabilityScore(0).build();
        }
        MultiLayerNetwork model = lstmTradePredictor.loadModelFromDb(symbol, jdbcTemplate);
        BarSeries series = getBarBySymbol(symbol, null);
        boolean inPosition = false;
        double entryPrice = 0.0;
        double capital = initialCapital;
        double positionSize = 0.0;
        double maxDrawdown = 0.0;
        double peakCapital = initialCapital;
        int tradeCount = 0;
        int winCount = 0;
        double totalGain = 0.0;
        double totalLoss = 0.0;
        double sumPnL = 0.0;
        double maxGain = Double.NEGATIVE_INFINITY;
        double maxLoss = Double.POSITIVE_INFINITY;
        int totalTradeBars = 0;
        int tradeStartIndex = 0;
        java.util.List<Double> tradeReturns = new java.util.ArrayList<>();
        for (int i = 0; i < series.getBarCount(); i++) {
            // Prédire le signal LSTM pour la bougie i
            PreditLsdm pred = lstmTradePredictor.getPreditAtIndex(symbol, series, config, model, i);
            double price = series.getBar(i).getClosePrice().doubleValue();
            if (!inPosition && pred.getSignal() == SignalType.BUY) {
                // Entrée en position
                positionSize = capital * riskPerTrade;
                entryPrice = price;
                inPosition = true;
                tradeStartIndex = i;
            } else if (inPosition) {
                double stopLossPrice = entryPrice * (1 - stopLossPct);
                double takeProfitPrice = entryPrice * (1 + takeProfitPct);
                boolean stopLossHit = price <= stopLossPrice;
                boolean takeProfitHit = price >= takeProfitPrice;
                boolean exitSignal = pred.getSignal() == SignalType.SELL;
                if (stopLossHit || takeProfitHit || exitSignal) {
                    double exitPrice = price;
                    if (stopLossHit) exitPrice = stopLossPrice;
                    if (takeProfitHit) exitPrice = takeProfitPrice;
                    double pnl = positionSize * ((exitPrice - entryPrice) / entryPrice);
                    capital += pnl;
                    tradeCount++;
                    sumPnL += pnl;
                    if (pnl > 0) {
                        winCount++;
                        totalGain += pnl;
                        if (pnl > maxGain) maxGain = pnl;
                    } else {
                        totalLoss += Math.abs(pnl);
                        if (pnl < maxLoss) maxLoss = pnl;
                    }
                    totalTradeBars += (i - tradeStartIndex + 1);
                    inPosition = false;
                    if (capital > peakCapital) peakCapital = capital;
                    double drawdown = (peakCapital - capital) / peakCapital;
                    if (drawdown > maxDrawdown) maxDrawdown = drawdown;
                    tradeReturns.add(pnl / initialCapital);
                }
            }
        }
        // Si une position reste ouverte à la fin, on la clôture au dernier prix
        if (inPosition) {
            double price = series.getBar(series.getEndIndex()).getClosePrice().doubleValue();
            double pnl = positionSize * ((price - entryPrice) / entryPrice);
            capital += pnl;
            tradeCount++;
            sumPnL += pnl;
            if (pnl > 0) {
                winCount++;
                totalGain += pnl;
                if (pnl > maxGain) maxGain = pnl;
            } else {
                totalLoss += Math.abs(pnl);
                if (pnl < maxLoss) maxLoss = pnl;
            }
            totalTradeBars += (series.getEndIndex() - tradeStartIndex + 1);
        }
        double rendement = (capital / initialCapital) - 1.0;
        double winRate = tradeCount > 0 ? (double) winCount / tradeCount : 0.0;
        double avgPnL = tradeCount > 0 ? sumPnL / tradeCount : 0.0;
        double profitFactor = totalLoss > 0 ? totalGain / totalLoss : 0.0;
        double avgTradeBars = tradeCount > 0 ? (double) totalTradeBars / tradeCount : 0.0;
        double maxTradeGain = (maxGain == Double.NEGATIVE_INFINITY) ? 0.0 : maxGain;
        double maxTradeLoss = (maxLoss == Double.POSITIVE_INFINITY) ? 0.0 : maxLoss;
        double meanReturn = tradeReturns.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double stdReturn = tradeReturns.size() > 1 ? Math.sqrt(tradeReturns.stream().mapToDouble(r -> Math.pow(r - meanReturn, 2)).sum() / (tradeReturns.size() - 1)) : 0.0;
        double sharpeRatio = stdReturn > 0 ? meanReturn / stdReturn : 0.0;
        double stabilityScore = stdReturn > 0 ? 1.0 / stdReturn : 0.0;
        return RiskResult.builder()
                .rendement(rendement)
                .tradeCount(tradeCount)
                .winRate(winRate)
                .maxDrawdown(maxDrawdown)
                .avgPnL(avgPnL)
                .profitFactor(profitFactor)
                .avgTradeBars(avgTradeBars)
                .maxTradeGain(maxTradeGain)
                .maxTradeLoss(maxTradeLoss)
                .scoreSwingTrade(0)
                .sharpeRatio(sharpeRatio)
                .stabilityScore(stabilityScore)
                .build();
    }
}
