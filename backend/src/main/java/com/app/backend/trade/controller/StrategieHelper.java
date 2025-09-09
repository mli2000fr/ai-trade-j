package com.app.backend.trade.controller;

import com.app.backend.model.RiskResult;
import com.app.backend.trade.model.*;
import com.app.backend.trade.model.alpaca.AlpacaAsset;
import com.app.backend.trade.service.*;
import com.app.backend.trade.strategy.BestInOutStrategy;
import com.app.backend.trade.strategy.StrategieBackTest;
import com.app.backend.trade.util.TradeConstant;
import com.app.backend.trade.util.TradeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Controller;
import org.ta4j.core.BarSeries;
import org.ta4j.core.Rule;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Classe helper pour la gestion des stratégies, optimisation et accès base de données.
 * Toutes les méthodes utilitaires ont été déplacées dans TradeUtils.
 * Cette classe se concentre sur la logique métier et l'accès aux services.
 */
@Controller
public class StrategieHelper {
    private static final Logger logger = LoggerFactory.getLogger(StrategieHelper.class);
    private final AlpacaService alpacaService;
    private final StrategyService strategyService;
    private final JdbcTemplate jdbcTemplate;
    private final StrategieBackTest strategieBackTest;
    private static final boolean INSERT_ONLY = true;

    @Autowired
    public StrategieHelper(AlpacaService alpacaService,
                           StrategyService strategyService,
                           JdbcTemplate jdbcTemplate,
                           StrategieBackTest strategieBackTest) {
        this.alpacaService = alpacaService;
        this.strategyService = strategyService;
        this.jdbcTemplate = jdbcTemplate;
        this.strategieBackTest = strategieBackTest;
    }

    /**
     * Calcule le signal combiné d'entrée ou de sortie pour une série de prix.
     * @param series série de prix
     * @param index index à tester
     * @param isEntry true pour entrée, false pour sortie
     * @return true si le signal est satisfait
     */
    public boolean getCombinedSignal(BarSeries series, int index, boolean isEntry) {
        Rule rule = isEntry ? strategyService.getEntryRule(series) : strategyService.getExitRule(series);
        boolean result = rule.isSatisfied(index);
        String log = "Test signal " + (isEntry ? "ENTREE" : "SORTIE") +
                " | index=" + index +
                " | prix=" + (series.getBar(index) != null ? series.getBar(index).getClosePrice() : "?") +
                " | stratégies actives=" + strategyService.getActiveStrategyNames() +
                " | mode=" + strategyService.getStrategyManager().getCombinationMode().name() +
                " | résultat=" + result;
        strategyService.addLog(log);
        return result;
    }


    /**
     * Met à jour les valeurs journalières pour tous les symboles actifs en base.
     */
    public void updateDBDailyValuAllSymbols(){
        List<String> listeDbSymbols = this.getAllAssetSymbolsFromDb();
        int error = 0;
        for(String symbol : listeDbSymbols){
            try{
                List<DailyValue> listeValues = this.updateDailyValue(symbol);
                for(DailyValue dv : listeValues){
                    this.insertDailyValue(symbol, dv);
                }
                Thread.sleep(200);
            }catch(Exception e){
                error++;
                TradeUtils.log("Erreur updateDailyValue("+symbol+") : " + e.getMessage());
            }
        }
        TradeUtils.log("updateDBDailyValuAllSymbols: total "+listeDbSymbols.size()+", error" + error);
    }

    /**
     * Récupère et met à jour les valeurs journalières en base, puis retourne la série correspondante.
     * @param symbol symbole à traiter
     * @param limit nombre de valeurs à retourner
     * @return BarSeries
     */
    public BarSeries getAndUpdateDBDailyValu(String symbol, int limit){
        String sql = "SELECT MAX(date) FROM daily_value WHERE symbol = ?";
        java.sql.Date lastDate = null;
        try {
            lastDate = jdbcTemplate.queryForObject(sql, new Object[]{symbol}, java.sql.Date.class);
        } catch (Exception e) {
            logger.warn("Aucune date trouvée pour le symbole {} dans daily_value ou erreur SQL: {}", symbol, e.getMessage());
        }
        String dateStart;
        boolean isUpToDate = false;
        java.time.LocalDate today = java.time.LocalDate.now();
        if (lastDate == null) {
            // Si aucune ligne trouvée, on prend la date de start par défaut
            dateStart = TradeUtils.getStartDate(800);
        } else {
            java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(today);
            java.time.LocalDate lastKnown = lastDate.toLocalDate();
            // Si la dernière date connue est le dernier jour de cotation, la base est à jour
            if (lastKnown.isEqual(lastTradingDay) || lastKnown.isAfter(lastTradingDay)) {
                isUpToDate = true;
            }
            // Sinon, on ajoute un jour à la date la plus récente
            dateStart = lastKnown.plusDays(1).toString(); // format YYYY-MM-DD
        }
        if (!isUpToDate) {
            List<DailyValue> listeValues = this.alpacaService.getHistoricalBars(symbol, dateStart, null);
            if (listeValues == null || listeValues.isEmpty()) {
                logger.info("Aucune donnée historique récupérée d'Alpaca pour {} depuis {}", symbol, dateStart);
            } else {
                for (DailyValue dv : listeValues) {
                    try {
                        this.insertDailyValue(symbol, dv);
                    } catch (Exception e) {
                        logger.warn("Erreur lors de l'insertion de DailyValue pour {} à la date {}: {}", symbol, dv.getDate(), e.getMessage());
                    }
                }
                try{
                    Thread.sleep(200);
                }catch(Exception e){}
            }
        }
        return TradeUtils.mapping(getDailyValuesFromDb(symbol, limit));
    }


    /**
     * Met à jour la base d'actifs depuis Alpaca.
     */
    public void updateDBAssets(){
        List<AlpacaAsset> listeSymbols = this.alpacaService.getIexSymbolsFromAlpaca();
        this.saveSymbolsToDatabase(listeSymbols);
    }


    /**
     * Insère une valeur journalière en base pour un symbole.
     * @param symbol symbole
     * @param dailyValue valeur à insérer
     */
    public void insertDailyValue(String symbol, DailyValue dailyValue) {
        // Conversion de la date (ex: "2025-03-18T04:00:00Z" ou "2025-03-18") en java.sql.Date
        java.sql.Date sqlDate = null;
        if (dailyValue.getDate() != null && !dailyValue.getDate().isEmpty()) {
            String dateStr = dailyValue.getDate();
            // Extraction de la partie date (YYYY-MM-DD)
            if (dateStr.length() >= 10) {
                dateStr = dateStr.substring(0, 10);
            }
            try {
                sqlDate = java.sql.Date.valueOf(dateStr);
            } catch (Exception e) {
                logger.warn("Format de date inattendu pour DailyValue: {}", dailyValue.getDate());
            }
        }
        String sql = "INSERT INTO daily_value (symbol, date, open, high, low, close, volume, number_of_trades, volume_weighted_average_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";
        jdbcTemplate.update(sql,
                symbol,
                sqlDate,
                dailyValue.getOpen(),
                dailyValue.getHigh(),
                dailyValue.getLow(),
                dailyValue.getClose(),
                dailyValue.getVolume(),
                dailyValue.getNumberOfTrades(),
                dailyValue.getVolumeWeightedAveragePrice()
        );
    }

    /**
     * Récupère tous les symboles actifs en base.
     * @return liste de symboles
     */
    public List<String> getAllAssetSymbolsFromDb() {
        String sql = "SELECT symbol FROM alpaca_asset WHERE status = 'active'";
        return jdbcTemplate.queryForList(sql, String.class);
    }

    /**
     * Récupère tous les symboles éligibles en base.
     * @return liste de symboles
     */
    public List<String> getAllAssetSymbolsEligibleFromDb() {
        String sql = "SELECT symbol FROM trade_ai.alpaca_asset WHERE status = 'active' and eligible = true;";
        return jdbcTemplate.queryForList(sql, String.class);
    }

    /**
     * Récupère les symboles manquants dans la table daily_value.
     * @return liste de symboles
     */
    public List<String> getAllAssetSymbolsComplementFromDb() {
        String sql = "SELECT symbol FROM trade_ai.alpaca_asset WHERE symbol NOT IN (SELECT symbol FROM trade_ai.daily_value);";
        return jdbcTemplate.queryForList(sql, String.class);
    }

    /**
     * Récupère les valeurs journalières d'un symbole depuis la base, avec limite.
     * @param symbol symbole
     * @param limit nombre de valeurs
     * @return liste de DailyValue
     */
    public List<DailyValue> getDailyValuesFromDb(String symbol, Integer limit) {
        String sql = "SELECT date, open, high, low, close, volume, number_of_trades, volume_weighted_average_price " +
                "FROM daily_value WHERE symbol = ? ORDER BY date DESC";
        if (limit != null && limit > 0) {
            sql += " LIMIT " + limit;
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
        Collections.reverse(results);
        return results;
    }

    /**
     * Sauvegarde une liste d'actifs Alpaca en base.
     * @param assets liste d'actifs
     */
    public void saveSymbolsToDatabase(List<AlpacaAsset> assets) {
        String sql = "INSERT IGNORE INTO alpaca_asset (id, symbol, exchange, status, name, created_at) VALUES (?, ?, ?, ?, ?, ?)";
        for (AlpacaAsset asset : assets) {
            jdbcTemplate.update(sql,
                    asset.getId(),
                    asset.getSymbol(),
                    asset.getExchange(),
                    asset.getStatus(),
                    asset.getName(),
                    new java.sql.Timestamp(System.currentTimeMillis())
            );
        }
    }


    /**
     * Met à jour les valeurs journalières pour un symbole.
     * @param symbol symbole
     * @return liste de DailyValue ajoutées
     */
    public  List<DailyValue> updateDailyValue(String symbol) {

        // 1. Chercher la date la plus récente pour ce symbol dans la table daily_value
        String sql = "SELECT MAX(date) FROM daily_value WHERE symbol = ?";
        java.sql.Date lastDate = null;
        try {
            lastDate = jdbcTemplate.queryForObject(sql, new Object[]{symbol}, java.sql.Date.class);
        } catch (Exception e) {
            logger.warn("Aucune date trouvée pour le symbole {} dans daily_value ou erreur SQL: {}", symbol, e.getMessage());
        }
        String dateStart;
        boolean isUpToDate = false;
        java.time.LocalDate today = java.time.LocalDate.now();
        if (lastDate == null) {
            // Si aucune ligne trouvée, on prend la date de start par défaut
            dateStart = TradeUtils.getStartDate(TradeConstant.HISTORIQUE_DAILY_VALUE);
        } else {
            java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(today);
            java.time.LocalDate lastKnown = lastDate.toLocalDate();
            // Si la dernière date connue est le dernier jour de cotation, la base est à jour
            if (lastKnown.isEqual(lastTradingDay) || lastKnown.isAfter(lastTradingDay)) {
                isUpToDate = true;
            }
            // Sinon, on ajoute un jour à la date la plus récente
            dateStart = lastKnown.plusDays(1).toString(); // format YYYY-MM-DD
        }
        if (!isUpToDate) {
            return this.alpacaService.getHistoricalBars(symbol, dateStart, null);
        }
        return  new ArrayList<>();
    }


    /**
     * Calcule les stratégies croisées pour tous les symboles éligibles.
     */
    public void calculCroisedStrategies(){
        List<String> listeDbSymbols = this.getAllAssetSymbolsEligibleFromDb();
        int nbThreads = Math.max(2, Runtime.getRuntime().availableProcessors());
        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(nbThreads);
        java.util.concurrent.atomic.AtomicInteger error = new java.util.concurrent.atomic.AtomicInteger(0);
        java.util.concurrent.atomic.AtomicInteger nbInsert = new java.util.concurrent.atomic.AtomicInteger(0);
        List<java.util.concurrent.Future<?>> futures = new ArrayList<>();
        for(String symbol : listeDbSymbols){
            futures.add(executor.submit(() -> {
                boolean isCalcul = true;
                try{
                    if(INSERT_ONLY){
                        String checkSql = "SELECT COUNT(*) FROM best_in_out_single_strategy WHERE symbol = ?";
                        int count = jdbcTemplate.queryForObject(checkSql, Integer.class, symbol);
                        if(count > 0){
                            isCalcul = false;
                            TradeUtils.log("calculCroisedStrategies: symbole "+symbol+" déjà en base, on passe");
                        }
                    }
                    if(isCalcul){
                        BestInOutStrategy result = optimseBestInOutByWalkForward(symbol);
                        this.saveBestInOutStrategy(symbol, result);
                        nbInsert.incrementAndGet();
                        try { Thread.sleep(200); } catch(Exception ignored) {}
                    }
                }catch(Exception e){
                    error.incrementAndGet();
                    TradeUtils.log("Erreur calcul("+symbol+") : " + e.getMessage());
                }
            }));
        }
        // Attendre la fin de tous les threads
        for (java.util.concurrent.Future<?> f : futures) {
            try { f.get(); } catch(Exception ignored) {}
        }
        executor.shutdown();
        TradeUtils.log("calculCroisedStrategies: total: "+listeDbSymbols.size()+", nbInsert: "+nbInsert.get()+", error: " + error.get());
    }

    /**
     * Optimise la meilleure combinaison IN/OUT par walk-forward pour un symbole.
     * @param symbol symbole
     * @return BestInOutStrategy
     */
    public BestInOutStrategy optimseBestInOutByWalkForward(String symbol) {
        List<DailyValue> listeValus = this.getDailyValuesFromDb(symbol, TradeConstant.NOMBRE_TOTAL_BOUGIES_OPTIM);
        BarSeries series = TradeUtils.mapping(listeValus);
        BarSeries[] split = TradeUtils.splitSeriesForWalkForward(series);
        BarSeries optimSeries = split[0];
        BarSeries testSeries = split[1];
        // Optimisation des paramètres sur la partie optimisation
        StrategieBackTest.ImprovedTrendFollowingParams bestImprovedTrend = strategieBackTest.optimiseImprovedTrendFollowingParameters(optimSeries, 10, 30, 5, 15, 15, 25, 0.001, 0.01, 0.002);
        StrategieBackTest.SmaCrossoverParams bestSmaCrossover = strategieBackTest.optimiseSmaCrossoverParameters(optimSeries, 5, 20, 10, 50);
        StrategieBackTest.RsiParams bestRsi = strategieBackTest.optimiseRsiParameters(optimSeries, 10, 20, 20, 40, 5, 60, 80, 5);
        StrategieBackTest.BreakoutParams bestBreakout = strategieBackTest.optimiseBreakoutParameters(optimSeries, 5, 50);
        StrategieBackTest.MacdParams bestMacd = strategieBackTest.optimiseMacdParameters(optimSeries, 8, 16, 20, 30, 6, 12);
        StrategieBackTest.MeanReversionParams bestMeanReversion = strategieBackTest.optimiseMeanReversionParameters(optimSeries, 10, 30, 1.0, 5.0, 0.5);
        // Liste des stratégies et paramètres
        java.util.List<Object[]> strategies = java.util.Arrays.asList(
            new Object[]{"Improved Trend", bestImprovedTrend},
            new Object[]{"SMA Crossover", bestSmaCrossover},
            new Object[]{"RSI", bestRsi},
            new Object[]{"Breakout", bestBreakout},
            new Object[]{"MACD", bestMacd},
            new Object[]{"Mean Reversion", bestMeanReversion}
        );
        double bestPerf = Double.NEGATIVE_INFINITY;
        BestInOutStrategy bestCombo = null;
        System.out.println("=== TESTS CROISÉS IN/OUT ===");
        com.google.gson.Gson gson = new com.google.gson.GsonBuilder().setPrettyPrinting().create();
        for (Object[] entry : strategies) {
            for (Object[] exit : strategies) {
                String entryName = (String) entry[0];
                Object entryParams = entry[1];
                String exitName = (String) exit[0];
                Object exitParams = exit[1];
                // Instancier les stratégies
                com.app.backend.trade.strategy.TradeStrategy entryStrategy = createStrategy(entryName, entryParams);
                com.app.backend.trade.strategy.TradeStrategy exitStrategy = createStrategy(exitName, exitParams);
                com.app.backend.trade.strategy.StrategieBackTest.CombinedTradeStrategy combined = new com.app.backend.trade.strategy.StrategieBackTest.CombinedTradeStrategy(entryStrategy, exitStrategy);
                // Backtest sur la partie test uniquement
                RiskResult result = strategieBackTest.backtestStrategyRisk(combined, testSeries);
                double scoreST =  TradeUtils.calculerScoreSwingTrade(result);
                result.setScoreSwingTrade(scoreST);
                System.out.println("Symbol: " + symbol +
                                   " | IN: " + entryName + " " + gson.toJson(entryParams) +
                                   " | OUT: " + exitName + " " + gson.toJson(exitParams) +
                                   " | Rendement: " + String.format("%.4f", result.rendement * 100) + "%"
                                   + " | Trades: " + result.tradeCount
                                   + " | WinRate: " + String.format("%.2f", result.winRate * 100) + "%"
                        + " | Drawdown: " + String.format("%.2f", result.maxDrawdown * 100) + "%"
                        + " | Score ST: " + String.format("%.2f", scoreST * 100) + "%");
                if (result.rendement > bestPerf) {
                    bestPerf = result.rendement;
                    bestCombo = BestInOutStrategy.builder()
                            .symbol(symbol)
                            .entryName(entryName)
                            .exitName(exitName)
                            .entryParams(entryParams)
                            .exitParams(exitParams)
                            .contextOptim(ContextOptim.builder()
                                    .initialCapital(StrategieBackTest.INITIAL_CAPITAL)
                                    .riskPerTrade(StrategieBackTest.RISK_PER_TRADE)
                                    .stopLossPct(StrategieBackTest.STOP_LOSS_PCT)
                                    .takeProfitPct(StrategieBackTest.TAKE_PROFIL_PCT)
                                    .nbSimples(listeValus.size())
                                    .build())
                            .result(result)
                            .build();
                }
            }
        }
        System.out.println("=== MEILLEUR COUPLE IN/OUT ===");
        if (bestCombo != null) {
            System.out.println("IN: " + bestCombo.entryName + " | OUT: " + bestCombo.exitName + " | Rendement: " + String.format("%.4f", bestCombo.result.rendement * 100) + "% | Trades: " + bestCombo.result.tradeCount);
            System.out.println("Paramètres IN: " + gson.toJson(bestCombo.entryParams));
            System.out.println("Paramètres OUT: " + gson.toJson(bestCombo.exitParams));
        }
        return bestCombo;
    }

    /**
     * Instancie une stratégie selon son nom et ses paramètres.
     * @param name nom de la stratégie
     * @param params paramètres
     * @return TradeStrategy
     */
    private com.app.backend.trade.strategy.TradeStrategy createStrategy(String name, Object params) {
        switch (name) {
            case "Improved Trend":
                StrategieBackTest.ImprovedTrendFollowingParams p = (StrategieBackTest.ImprovedTrendFollowingParams) params;
                return new com.app.backend.trade.strategy.ImprovedTrendFollowingStrategy(p.trendPeriod, p.shortMaPeriod, p.longMaPeriod, p.breakoutThreshold, p.useRsiFilter, p.rsiPeriod);
            case "SMA Crossover":
                StrategieBackTest.SmaCrossoverParams s = (StrategieBackTest.SmaCrossoverParams) params;
                return new com.app.backend.trade.strategy.SmaCrossoverStrategy(s.shortPeriod, s.longPeriod);
            case "RSI":
                StrategieBackTest.RsiParams r = (StrategieBackTest.RsiParams) params;
                return new com.app.backend.trade.strategy.RsiStrategy(r.rsiPeriod, r.oversold, r.overbought);
            case "Breakout":
                StrategieBackTest.BreakoutParams b = (StrategieBackTest.BreakoutParams) params;
                return new com.app.backend.trade.strategy.BreakoutStrategy(b.lookbackPeriod);
            case "MACD":
                StrategieBackTest.MacdParams m = (StrategieBackTest.MacdParams) params;
                return new com.app.backend.trade.strategy.MacdStrategy(m.shortPeriod, m.longPeriod, m.signalPeriod);
            case "Mean Reversion":
                StrategieBackTest.MeanReversionParams mr = (StrategieBackTest.MeanReversionParams) params;
                return new com.app.backend.trade.strategy.MeanReversionStrategy(mr.smaPeriod, mr.threshold);
            default:
                throw new IllegalArgumentException("Stratégie inconnue: " + name);
        }
    }

    /**
     * Sauvegarde la meilleure stratégie IN/OUT en base.
     * @param symbol symbole
     * @param best stratégie à sauvegarder
     */
    public void saveBestInOutStrategy(String symbol, BestInOutStrategy best) {
        String checkSql = "SELECT COUNT(*) FROM best_in_out_single_strategy WHERE symbol = ?";
        int count = jdbcTemplate.queryForObject(checkSql, Integer.class, symbol);
        String entryParamsJson = new com.google.gson.Gson().toJson(best.entryParams);
        String exitParamsJson = new com.google.gson.Gson().toJson(best.exitParams);
        if (count > 0) {
            // Mise à jour
            String updateSql = """
                UPDATE best_in_out_single_strategy SET
                    entry_strategy_name = ?,
                    entry_strategy_params = ?,
                    exit_strategy_name = ?,
                    exit_strategy_params = ?,
                    rendement = ?,
                    trade_count = ?,
                    win_rate = ?,
                    max_drawdown = ?,
                    avg_pnl = ?,
                    profit_factor = ?,
                    avg_trade_bars = ?,
                    max_trade_gain = ?,
                    max_trade_loss = ?,
                    score_swing_trade = ?,
                    initial_capital = ?,
                    risk_per_trade = ?,
                    stop_loss_pct = ?,
                    take_profit_pct = ?,
                    nb_simples = ?,
                    updated_date = CURRENT_TIMESTAMP
                WHERE symbol = ?
            """;
            jdbcTemplate.update(updateSql,
                best.entryName,
                entryParamsJson,
                best.exitName,
                exitParamsJson,
                best.result.rendement,
                best.result.tradeCount,
                best.result.winRate,
                best.result.maxDrawdown,
                best.result.avgPnL,
                best.result.profitFactor,
                best.result.avgTradeBars,
                best.result.maxTradeGain,
                best.result.maxTradeLoss,
                best.result.scoreSwingTrade,
                best.contextOptim.initialCapital,
                best.contextOptim.riskPerTrade,
                best.contextOptim.stopLossPct,
                best.contextOptim.takeProfitPct,
                best.contextOptim.nbSimples,
                symbol
            );
        } else {
            // Insertion
            String insertSql = """
                INSERT INTO best_in_out_single_strategy (
                    symbol, entry_strategy_name, entry_strategy_params,
                    exit_strategy_name, exit_strategy_params,
                    rendement, trade_count, win_rate, max_drawdown, avg_pnl, profit_factor, avg_trade_bars, max_trade_gain, max_trade_loss,
                    score_swing_trade, initial_capital, risk_per_trade, stop_loss_pct, take_profit_pct, nb_simples,
                    created_date, updated_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """;
            jdbcTemplate.update(insertSql,
                symbol,
                best.entryName,
                entryParamsJson,
                best.exitName,
                exitParamsJson,
                best.result.rendement,
                best.result.tradeCount,
                best.result.winRate,
                best.result.maxDrawdown,
                best.result.avgPnL,
                best.result.profitFactor,
                best.result.avgTradeBars,
                best.result.maxTradeGain,
                best.result.maxTradeLoss,
                best.result.scoreSwingTrade,
                best.contextOptim.initialCapital,
                best.contextOptim.riskPerTrade,
                best.contextOptim.stopLossPct,
                best.contextOptim.takeProfitPct,
                best.contextOptim.nbSimples,
                java.sql.Date.valueOf(java.time.LocalDate.now())
            );
        }
    }

    /**
     * Récupère la meilleure stratégie IN/OUT pour un symbole.
     * @param symbol symbole
     * @return BestInOutStrategy
     */
    public BestInOutStrategy getBestInOutStrategy(String symbol) {
        String sql = "SELECT * FROM best_in_out_single_strategy WHERE symbol = ?";
        try {
            return jdbcTemplate.queryForObject(sql, (rs, rowNum) -> {
                String entryName = rs.getString("entry_strategy_name");
                String entryParamsJson = rs.getString("entry_strategy_params");
                String exitName = rs.getString("exit_strategy_name");
                String exitParamsJson = rs.getString("exit_strategy_params");
                Object entryParams = TradeUtils.parseStrategyParams(entryName, entryParamsJson);
                Object exitParams = TradeUtils.parseStrategyParams(exitName, exitParamsJson);
                return BestInOutStrategy.builder()
                        .symbol(symbol)
                        .entryName(entryName)
                        .exitName(exitName)
                        .entryParams(entryParams)
                        .exitParams(exitParams)
                        .contextOptim(ContextOptim.builder()
                                .initialCapital(rs.getDouble("initial_capital"))
                                .riskPerTrade(rs.getDouble("risk_per_trade"))
                                .stopLossPct(rs.getDouble("stop_loss_pct"))
                                .takeProfitPct(rs.getDouble("take_profit_pct"))
                                .nbSimples(rs.getInt("nb_simples"))
                                .build())
                        .result(RiskResult.builder()
                                .rendement(rs.getDouble("rendement"))
                                .tradeCount(rs.getInt("trade_count"))
                                .winRate(rs.getDouble("win_rate"))
                                .maxDrawdown(rs.getDouble("max_drawdown"))
                                .avgPnL(rs.getDouble("avg_pnl"))
                                .profitFactor(rs.getDouble("profit_factor"))
                                .avgTradeBars(rs.getDouble("avg_trade_bars"))
                                .maxTradeGain(rs.getDouble("max_trade_gain"))
                                .maxTradeLoss(rs.getDouble("max_trade_loss"))
                                .scoreSwingTrade(rs.getDouble("score_swing_trade"))
                        .build()).build();
            }, symbol);
        } catch (org.springframework.dao.EmptyResultDataAccessException e) {
            logger.warn("Aucun BestInOutStrategy trouvé pour le symbole: {}", symbol);
            return null;
        }
    }

    /**
     * Récupère la liste des meilleures performances d'actions selon le tri et la limite.
     * @param limit nombre maximum d'actions à retourner (optionnel)
     * @param sort critère de tri (par défaut rendement)
     * @return liste des meilleures stratégies BestInOutStrategy
     */
    public List<BestInOutStrategy> getBestPerfActions(Integer limit, String sort){
        String orderBy = "rendement";
        if ("score_swing_trade".equalsIgnoreCase(sort)) {
            orderBy = "score_swing_trade";
        }
        String sql = "SELECT * FROM best_in_out_single_strategy WHERE profit_factor <> 0 AND max_drawdown <> 0 AND win_rate < 1 ORDER BY " + orderBy + " DESC";
        if (limit != null && limit > 0) {
            sql += " LIMIT " + limit;
        }
        List<BestInOutStrategy> results = jdbcTemplate.query(sql, (rs, rowNum) -> {
            String entryName = rs.getString("entry_strategy_name");
            String entryParamsJson = rs.getString("entry_strategy_params");
            String exitName = rs.getString("exit_strategy_name");
            String exitParamsJson = rs.getString("exit_strategy_params");
            Object entryParams = TradeUtils.parseStrategyParams(entryName, entryParamsJson);
            Object exitParams = TradeUtils.parseStrategyParams(exitName, exitParamsJson);
            return BestInOutStrategy.builder()
                    .symbol(rs.getString("symbol"))
                    .entryName(entryName)
                    .exitName(exitName)
                    .entryParams(entryParams)
                    .exitParams(exitParams)
                    .contextOptim(ContextOptim.builder()
                            .initialCapital(rs.getDouble("initial_capital"))
                            .riskPerTrade(rs.getDouble("risk_per_trade"))
                            .stopLossPct(rs.getDouble("stop_loss_pct"))
                            .takeProfitPct(rs.getDouble("take_profit_pct"))
                            .nbSimples(rs.getInt("nb_simples"))
                            .build())
                    .result(RiskResult.builder()
                            .rendement(rs.getDouble("rendement"))
                            .tradeCount(rs.getInt("trade_count"))
                            .winRate(rs.getDouble("win_rate"))
                            .maxDrawdown(rs.getDouble("max_drawdown"))
                            .avgPnL(rs.getDouble("avg_pnl"))
                            .profitFactor(rs.getDouble("profit_factor"))
                            .avgTradeBars(rs.getDouble("avg_trade_bars"))
                            .maxTradeGain(rs.getDouble("max_trade_gain"))
                            .maxTradeLoss(rs.getDouble("max_trade_loss"))
                            .scoreSwingTrade(rs.getDouble("score_swing_trade"))
                            .build()).build();
        });
        return results;
    }

    /**
     * Récupère le signal d'indice pour un symbole donné.
     * @param symbol symbole à analyser (optionnel)
     * @return type de signal (SignalType)
     */
    public SignalType getBestInOutSignal(String symbol) {
        BestInOutStrategy best = getBestInOutStrategy(symbol);
        if (best == null) return SignalType.NONE;
        BarSeries series = getAndUpdateDBDailyValu(symbol, TradeConstant.NOMBRE_TOTAL_BOUGIES);
        int lastIndex = series.getEndIndex();
        // Instancie les stratégies IN/OUT
        com.app.backend.trade.strategy.TradeStrategy entryStrategy = createStrategy(best.entryName, best.entryParams);
        com.app.backend.trade.strategy.TradeStrategy exitStrategy = createStrategy(best.exitName, best.exitParams);
        boolean entrySignal = entryStrategy.getEntryRule(series).isSatisfied(lastIndex);
        boolean exitSignal = exitStrategy.getExitRule(series).isSatisfied(lastIndex);
        if (entrySignal) return SignalType.BUY;
        if (exitSignal) return SignalType.SELL;
        return SignalType.NONE;
    }
}
