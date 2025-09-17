package com.app.backend.trade.controller;

import com.app.backend.trade.model.RiskResult;
import com.app.backend.trade.model.*;
import com.app.backend.trade.model.alpaca.AlpacaAsset;
import com.app.backend.trade.service.*;
import com.app.backend.trade.strategy.BestInOutStrategy;
import com.app.backend.trade.strategy.ParamsOptim;
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

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static com.app.backend.trade.util.TradeConstant.NOMBRE_TOTAL_BOUGIES_FOR_SIGNAL;

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
    private final SwingTradeOptimParams swingParams = new SwingTradeOptimParams();


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
        int compteur = 0;
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
            compteur++;
            TradeUtils.log("updateDBDailyValuAllSymbols: compteur "+compteur);
        }
        TradeUtils.log("updateDBDailyValuAllSymbols: total "+listeDbSymbols.size()+", error" + error);
    }


    /**
     * Récupère et met à jour les valeurs journalières en base, puis retourne la série correspondante.
     * @param symbol symbole à traiter
     * @return BarSeries
     */
    public void updateDBDailyValu(String symbol){
        String sql = "SELECT MAX(date) FROM daily_value WHERE symbol = ?";
        java.sql.Date lastDate = null;
        try {
            lastDate = jdbcTemplate.query(sql, ps -> ps.setString(1, symbol), rs -> {
                if (rs.next()) {
                    return rs.getDate(1);
                }
                return null;
            });
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
                }catch(Exception e){
                    logger.warn("Erreur lors du sleep: {}", e.getMessage());
                }
            }
        }
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
        String sql = "SELECT symbol FROM trade_ai.alpaca_asset WHERE status = 'active' and eligible = true ORDER BY symbol ASC;";
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
            lastDate = jdbcTemplate.query(sql, ps -> ps.setString(1, symbol), rs -> {
                if (rs.next()) {
                    return rs.getDate(1);
                }
                return null;
            });
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


    // Structure de suivi du calcul croisé
    public static class CroisedProgress {
        public String status = "";
        public int testedConfigs = 0;
        public int totalConfigs = 0;
        public long startTime = 0;
        public long endTime = 0;
        public long lastUpdate = 0;
    }

    private CroisedProgress croisedProgress = new CroisedProgress();

    public CroisedProgress getCroisedProgress() {
        return croisedProgress;
    }

    /**
     * Calcule les stratégies croisées pour tous les symboles éligibles.
     */
    public void calculCroisedStrategies(){
        croisedProgress = new CroisedProgress();
        croisedProgress.status = "en_cours";
        croisedProgress.startTime = System.currentTimeMillis();
        croisedProgress.lastUpdate = croisedProgress.startTime;
        java.util.concurrent.atomic.AtomicInteger error = new java.util.concurrent.atomic.AtomicInteger(0);
        java.util.concurrent.atomic.AtomicInteger nbInsert = new java.util.concurrent.atomic.AtomicInteger(0);
        try {
            List<String> listeDbSymbols = this.getAllAssetSymbolsEligibleFromDb();
            croisedProgress.totalConfigs = listeDbSymbols.size();
            int nbThreads = Math.max(2, Runtime.getRuntime().availableProcessors());
            java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(nbThreads);
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
                            BestInOutStrategy result = optimseStrategy(symbol);
                            this.saveBestInOutStrategy(symbol, result);
                            nbInsert.incrementAndGet();
                        }
                        // Incrémenter la progression à chaque symbole, calculé ou ignoré
                        synchronized (croisedProgress) {
                            croisedProgress.testedConfigs++;
                            croisedProgress.lastUpdate = System.currentTimeMillis();
                        }
                        try { Thread.sleep(200); } catch(Exception ignored) {}
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
            croisedProgress.status = "termine";
            croisedProgress.endTime = System.currentTimeMillis();
        } catch (Exception e) {
            croisedProgress.status = "erreur";
            croisedProgress.endTime = System.currentTimeMillis();
        }
        TradeUtils.log("calculCroisedStrategies: total: "+getCroisedProgress().totalConfigs+", nbInsert: "+nbInsert.get()+", error: " + error.get());
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
                System.out.println("Symbol: " + symbol +
                                   " | IN: " + entryName + " " + gson.toJson(entryParams) +
                                   " | OUT: " + exitName + " " + gson.toJson(exitParams) +
                                   " | Rendement: " + String.format("%.4f", result.rendement * 100) + "%"
                                   + " | Trades: " + result.tradeCount
                                   + " | WinRate: " + String.format("%.2f", result.winRate * 100) + "%"
                        + " | Drawdown: " + String.format("%.2f", result.maxDrawdown * 100) + "%"
                        + " | Score ST: " + String.format("%.2f", result.scoreSwingTrade));
                if (result.rendement > bestPerf) {
                    bestPerf = result.rendement;
                    bestCombo = BestInOutStrategy.builder()
                            .symbol(symbol)
                            .entryName(entryName)
                            .exitName(exitName)
                            .entryParams(entryParams)
                            .exitParams(exitParams)
                            .paramsOptim(ParamsOptim.builder()
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
        String check = new com.google.gson.Gson().toJson(best.check);
        if (count > 0) {
            // Mise à jour
            String updateSql = """
                UPDATE best_in_out_single_strategy SET
                    entry_strategy_name = ?,
                    entry_strategy_params = ?,
                    exit_strategy_name = ?,
                    exit_strategy_params = ?,
                    rendement = ?,
                    rendement_sum = ?,
                    rendement_diff = ?,
                    rendement_score = ?,
                    trade_count = ?,
                    win_rate = ?,
                    max_drawdown = ?,
                    avg_pnl = ?,
                    profit_factor = ?,
                    avg_trade_bars = ?,
                    max_trade_gain = ?,
                    max_trade_loss = ?,
                    score_swing_trade = ?,
                    fltred_out = ?,
                    initial_capital = ?,
                    risk_per_trade = ?,
                    stop_loss_pct = ?,
                    take_profit_pct = ?,
                    nb_simples = ?,
                    check_result = ?
                    updated_date = CURRENT_TIMESTAMP
                WHERE symbol = ?
            """;
            jdbcTemplate.update(updateSql,
                best.entryName,
                entryParamsJson,
                best.exitName,
                exitParamsJson,
                best.result.rendement,
                    best.rendementSum,
                    best.rendementDiff,
                    best.rendementScore,
                best.result.tradeCount,
                best.result.winRate,
                best.result.maxDrawdown,
                best.result.avgPnL,
                best.result.profitFactor,
                best.result.avgTradeBars,
                best.result.maxTradeGain,
                best.result.maxTradeLoss,
                best.result.scoreSwingTrade,
                best.result.fltredOut,
                best.paramsOptim.initialCapital,
                best.paramsOptim.riskPerTrade,
                best.paramsOptim.stopLossPct,
                best.paramsOptim.takeProfitPct,
                best.paramsOptim.nbSimples,
                    check,
                symbol
            );
        } else {
            // Insertion
            String insertSql = """
                INSERT INTO best_in_out_single_strategy (
                    symbol, entry_strategy_name, entry_strategy_params,
                    exit_strategy_name, exit_strategy_params,
                    rendement, rendement_sum, rendement_diff, rendement_score, trade_count, win_rate, max_drawdown, avg_pnl, profit_factor, avg_trade_bars, max_trade_gain, max_trade_loss,
                    score_swing_trade, fltred_out, initial_capital, risk_per_trade, stop_loss_pct, 
                    take_profit_pct, nb_simples, check_result, created_date, updated_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """;
            jdbcTemplate.update(insertSql,
                symbol,
                best.entryName,
                entryParamsJson,
                best.exitName,
                exitParamsJson,
                best.result.rendement,
                best.rendementSum,
                best.rendementDiff,
                    best.rendementScore,
                best.result.tradeCount,
                best.result.winRate,
                best.result.maxDrawdown,
                best.result.avgPnL,
                best.result.profitFactor,
                best.result.avgTradeBars,
                best.result.maxTradeGain,
                best.result.maxTradeLoss,
                best.result.scoreSwingTrade,
                best.result.fltredOut,
                best.paramsOptim.initialCapital,
                best.paramsOptim.riskPerTrade,
                best.paramsOptim.stopLossPct,
                best.paramsOptim.takeProfitPct,
                best.paramsOptim.nbSimples,
                    check,
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
                        .check(new com.google.gson.Gson().fromJson(rs.getString("check_result"), RiskResult.class))
                        .paramsOptim(ParamsOptim.builder()
                                .initialCapital(rs.getDouble("initial_capital"))
                                .riskPerTrade(rs.getDouble("risk_per_trade"))
                                .stopLossPct(rs.getDouble("stop_loss_pct"))
                                .takeProfitPct(rs.getDouble("take_profit_pct"))
                                .nbSimples(rs.getInt("nb_simples"))
                                .build())
                        .rendementSum(rs.getDouble("rendement_sum"))
                        .rendementDiff(rs.getDouble("rendement_diff"))
                        .rendementScore(rs.getDouble("rendement_score"))
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
                                .fltredOut(rs.getBoolean("fltred_out")).build()).build();
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
    public List<BestInOutStrategy> getBestPerfActions(Integer limit, String sort, String search, Boolean filtered){
        String orderBy = "rendement_score";
        if ("score_swing_trade".equalsIgnoreCase(sort)) {
            orderBy = "score_swing_trade";
        }else if ("rendement_sum".equalsIgnoreCase(sort)) {
            orderBy = "rendement_sum";
        }else if ("rendement".equalsIgnoreCase(sort)) {
            orderBy = "rendement";
        }
        String searchSQL = "";
        if(search != null && !search.isEmpty()){
            searchSQL = "symbol in ("+"'"+search.replaceAll(" ", "").replaceAll(",", "','")+"'"+") and";
        }
        String sql = "SELECT * FROM best_in_out_single_strategy WHERE "+ searchSQL +" profit_factor <> 0 AND win_rate < 1";
        if (filtered != null && filtered) {
            sql += " AND fltred_out = false";
        }
        sql += " ORDER BY " + orderBy + " DESC";
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
                    .check(new com.google.gson.Gson().fromJson(rs.getString("check_result"), RiskResult.class))
                    .paramsOptim(ParamsOptim.builder()
                            .initialCapital(rs.getDouble("initial_capital"))
                            .riskPerTrade(rs.getDouble("risk_per_trade"))
                            .stopLossPct(rs.getDouble("stop_loss_pct"))
                            .takeProfitPct(rs.getDouble("take_profit_pct"))
                            .nbSimples(rs.getInt("nb_simples"))
                            .build())
                    .rendementSum(rs.getDouble("rendement_sum"))
                    .rendementDiff(rs.getDouble("rendement_diff"))
                    .rendementScore(rs.getDouble("rendement_score"))
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
                            .fltredOut(rs.getBoolean("fltred_out"))
                            .build()).build();
        });
        return results;
    }

    /**
     * Récupère le signal d'indice pour un symbole donné.
     * @param symbol symbole à analyser (optionnel)
     * @return type de signal (SignalType)
     */
    public SignalInfo getBestInOutSignal(String symbol) {

        SignalInfo singleDB = this.getSingalTypeFromDB(symbol);
        if(singleDB != null){
            String dateStr = singleDB.getDate().toLocalDate().format(java.time.format.DateTimeFormatter.ofPattern("dd/MM"));
            return SignalInfo.builder().symbol(symbol).type(singleDB.getType()).dateStr(dateStr).build();
        }
        java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(java.time.LocalDate.now());


        BestInOutStrategy best = getBestInOutStrategy(symbol);
        if (best == null) return SignalInfo.builder().symbol(symbol).type(SignalType.NONE)
                .dateStr(lastTradingDay.format(java.time.format.DateTimeFormatter.ofPattern("dd/MM"))).build();
        updateDBDailyValu(symbol);
        List<DailyValue> listeValus = this.getDailyValuesFromDb(symbol, NOMBRE_TOTAL_BOUGIES_FOR_SIGNAL);
        BarSeries series = TradeUtils.mapping(listeValus);
        int lastIndex = series.getEndIndex();
        // Instancie les stratégies IN/OUT
        com.app.backend.trade.strategy.TradeStrategy entryStrategy = createStrategy(best.entryName, best.entryParams);
        com.app.backend.trade.strategy.TradeStrategy exitStrategy = createStrategy(best.exitName, best.exitParams);
        boolean entrySignal = entryStrategy.getEntryRule(series).isSatisfied(lastIndex);
        boolean exitSignal = exitStrategy.getExitRule(series).isSatisfied(lastIndex);
        SignalType signal = SignalType.HOLD;
        if (entrySignal) signal =  SignalType.BUY;
        if (exitSignal) signal = SignalType.SELL;
        LocalDate dateSaved = saveSignalHistory(symbol, signal);
        String dateSavedStr = dateSaved.format(java.time.format.DateTimeFormatter.ofPattern("dd/MM"));
        return SignalInfo.builder().symbol(symbol).type(signal).dateStr(dateSavedStr).build();
    }

    /**
     * Récupère le dernier signal enregistré en base pour un symbole donné, avec sa date.
     * @param symbol symbole à analyser
     * @return SignalInfo (type + date)
     */
    public SignalInfo getSingalTypeFromDB(String symbol) {
        String sql = "SELECT signal_single, single_created_at FROM signal_single WHERE symbol = ? ORDER BY single_created_at DESC LIMIT 1";
        try {
            return jdbcTemplate.query(sql, ps -> ps.setString(1, symbol), rs -> {
                if (rs.next()) {
                    String signalStr = rs.getString("signal_single");
                    java.sql.Date lastDate = rs.getDate("single_created_at");
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
                        return SignalInfo.builder().symbol(symbol).type(type).date(lastDate).dateStr(dateSavedStr).build();
                    }else{
                        return null;
                    }
                }
                return null;
            });
        } catch (Exception e) {
            logger.warn("Erreur SQL getSingalTypeFromDB pour {}: {}", symbol, e.getMessage());
            return null;
        }
    }

    public LocalDate saveSignalHistory(String symbol, SignalType signal) {
        java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(java.time.LocalDate.now());
        String insertSql = "INSERT INTO signal_single (symbol, signal_single, single_created_at) VALUES (?, ?, ?)";
        jdbcTemplate.update(insertSql,
                symbol,
                signal.name(),
                java.sql.Date.valueOf(lastTradingDay));
        return lastTradingDay;

    }


    public BestInOutStrategy optimseStrategy(String symbol) {
        List<DailyValue> listeValus = this.getDailyValuesFromDb(symbol, TradeConstant.NOMBRE_TOTAL_BOUGIES_OPTIM);
        BarSeries series = TradeUtils.mapping(listeValus);
        // Création d'une config de filtrage par défaut (modifiable si besoin)
        StrategyFilterConfig filterConfig = new StrategyFilterConfig();
        // Utilisation du swingParams de la classe (modifiable si besoin)
        WalkForwardResultPro walkForwardResultPro =  this.optimseStrategy(series, 0.2, 0.1, 0.1, filterConfig, swingParams);
        double rendementSum = walkForwardResultPro.getBestCombo().getResult().getRendement() + walkForwardResultPro.getCheck().getRendement();
        double rendementDiff = walkForwardResultPro.getBestCombo().getResult().getRendement() - walkForwardResultPro.getCheck().getRendement();
        return BestInOutStrategy.builder()
                .symbol(symbol)
                .rendementSum(rendementSum)
                .rendementDiff(rendementDiff)
                .rendementScore(rendementSum - (rendementDiff > 0 ? rendementDiff : -rendementDiff))
                .entryName(walkForwardResultPro.getBestCombo().getEntryName())
                .entryParams(walkForwardResultPro.getBestCombo().getEntryParams())
                .exitName(walkForwardResultPro.getBestCombo().getExitName())
                .exitParams(walkForwardResultPro.getBestCombo().getExitParams())
                .paramsOptim(ParamsOptim.builder()
                        .initialCapital(StrategieBackTest.INITIAL_CAPITAL)
                        .riskPerTrade(StrategieBackTest.RISK_PER_TRADE)
                        .stopLossPct(StrategieBackTest.STOP_LOSS_PCT)
                        .takeProfitPct(StrategieBackTest.TAKE_PROFIL_PCT)
                        .nbSimples(listeValus.size())
                        .build())
                .result(walkForwardResultPro.getBestCombo().getResult())
                .check(walkForwardResultPro.getCheck()).build();
    }


    /**
     * Optimisation walk-forward professionnelle pour le swing trade.
     * Les paramètres sont des pourcentages du nombre total de bougies.
     * @param series série de prix
     * @param optimWindowPct pourcentage de la fenêtre d'optimisation (ex: 0.2 pour 20%)
     * @param testWindowPct pourcentage de la fenêtre de test
     * @param stepWindowPct pourcentage du pas de glissement
     * @param filterConfig configuration des critères de filtrage
     * @param swingParams paramètres d'optimisation swing trade
     * @return WalkForwardResultPro
     */
    public WalkForwardResultPro optimseStrategy(BarSeries series, double optimWindowPct, double testWindowPct, double stepWindowPct, StrategyFilterConfig filterConfig, SwingTradeOptimParams swingParams) {
        logger.info("[optimseStrategy] Démarrage de l'optimisation walk-forward avec validation croisée : optimWindowPct={}, testWindowPct={}, stepWindowPct={}", optimWindowPct, testWindowPct, stepWindowPct);
        int totalBars = series.getBarCount();
        int optimWindow = Math.max(1, (int) Math.round(totalBars * optimWindowPct));
        int testWindow = Math.max(1, (int) Math.round(totalBars * testWindowPct));
        int stepWindow = Math.max(1, (int) Math.round(totalBars * stepWindowPct));
        int kFolds = 5; // Nombre de folds pour la validation croisée (modifiable)
        List<List<ComboResult>> foldsResults = new ArrayList<>();
        int foldSize = (totalBars - (optimWindow + testWindow)) / kFolds;
        if (foldSize < 1) foldSize = 1;
        List<Double> trainPerformances = new ArrayList<>();
        List<Double> testPerformances = new ArrayList<>();
        for (int fold = 0; fold < kFolds; fold++) {
            int start = fold * foldSize;
            if (start + optimWindow + testWindow > totalBars) break;
            BarSeries optimSeries = series.getSubSeries(start, start + optimWindow);
            BarSeries testSeries = series.getSubSeries(start + optimWindow, start + optimWindow + testWindow);
            // --- Optimisation des paramètres sur le train ---
            StrategieBackTest.ImprovedTrendFollowingParams bestImprovedTrend = strategieBackTest.optimiseImprovedTrendFollowingParameters(
                optimSeries,
                swingParams.trendMaMin, swingParams.trendMaMax,
                swingParams.trendShortMaMin, swingParams.trendShortMaMax,
                swingParams.trendLongMaMin, swingParams.trendLongMaMax,
                swingParams.trendBreakoutMin, swingParams.trendBreakoutMax, swingParams.trendBreakoutStep
            );
            StrategieBackTest.SmaCrossoverParams bestSmaCrossover = strategieBackTest.optimiseSmaCrossoverParameters(
                optimSeries,
                swingParams.smaShortMin, swingParams.smaShortMax,
                swingParams.smaLongMin, swingParams.smaLongMax
            );
            StrategieBackTest.RsiParams bestRsi = strategieBackTest.optimiseRsiParameters(
                optimSeries,
                swingParams.rsiPeriodMin, swingParams.rsiPeriodMax,
                swingParams.rsiOversoldMin, swingParams.rsiOversoldMax,
                swingParams.rsiStep,
                swingParams.rsiOverboughtMin, swingParams.rsiOverboughtMax,
                swingParams.rsiStep
            );
            StrategieBackTest.BreakoutParams bestBreakout = strategieBackTest.optimiseBreakoutParameters(
                optimSeries,
                swingParams.breakoutLookbackMin, swingParams.breakoutLookbackMax
            );
            StrategieBackTest.MacdParams bestMacd = strategieBackTest.optimiseMacdParameters(
                optimSeries,
                swingParams.macdShortMin, swingParams.macdShortMax,
                swingParams.macdLongMin, swingParams.macdLongMax,
                swingParams.macdSignalMin, swingParams.macdSignalMax
            );
            StrategieBackTest.MeanReversionParams bestMeanReversion = strategieBackTest.optimiseMeanReversionParameters(
                optimSeries,
                swingParams.meanRevSmaMin, swingParams.meanRevSmaMax,
                swingParams.meanRevThresholdMin, swingParams.meanRevThresholdMax,
                swingParams.meanRevThresholdStep
            );
            java.util.List<Object[]> strategies = java.util.Arrays.asList(
                new Object[]{"Improved Trend", bestImprovedTrend},
                new Object[]{"SMA Crossover", bestSmaCrossover},
                new Object[]{"RSI", bestRsi},
                new Object[]{"Breakout", bestBreakout},
                new Object[]{"MACD", bestMacd},
                new Object[]{"Mean Reversion", bestMeanReversion}
            );
            List<ComboResult> foldResults = new ArrayList<>();
            double bestPerf = Double.NEGATIVE_INFINITY;
            ComboResult bestCombo = null;
            double bestTrainPerf = Double.NEGATIVE_INFINITY;
            for (Object[] entry : strategies) {
                for (Object[] exit : strategies) {
                    String entryName = (String) entry[0];
                    Object entryParams = entry[1];
                    String exitName = (String) exit[0];
                    Object exitParams = exit[1];
                    com.app.backend.trade.strategy.TradeStrategy entryStrategy = createStrategy(entryName, entryParams);
                    com.app.backend.trade.strategy.TradeStrategy exitStrategy = createStrategy(exitName, exitParams);
                    com.app.backend.trade.strategy.StrategieBackTest.CombinedTradeStrategy combined = new com.app.backend.trade.strategy.StrategieBackTest.CombinedTradeStrategy(entryStrategy, exitStrategy);
                    // Backtest sur train
                    RiskResult trainResult = strategieBackTest.backtestStrategy(combined, optimSeries);
                    if (trainResult.getRendement() > bestTrainPerf) {
                        bestTrainPerf = trainResult.getRendement();
                    }
                    // Backtest sur test
                    RiskResult result = strategieBackTest.backtestStrategy(combined, testSeries);
                    // Calcul du ratio overfit pour ce combo
                    double overfitRatioCombo = result.getRendement() / (trainResult.getRendement() == 0.0 ? 1.0 : trainResult.getRendement());
                    boolean isOverfitCombo = (overfitRatioCombo < 0.7 || overfitRatioCombo > 1.3);
                    boolean stable = result != null && TradeUtils.isStableAndSimple(result, filterConfig);
                    result.setFltredOut(!stable || isOverfitCombo);
                    ComboResult combo = ComboResult.builder()
                        .entryName(entryName)
                        .entryParams(entryParams)
                        .exitName(exitName)
                        .exitParams(exitParams)
                        .result(result)
                        .build();
                    foldResults.add(combo);
                    if (result.getRendement() > bestPerf) {
                        bestPerf = result.getRendement();
                    }
                }
            }
            trainPerformances.add(bestTrainPerf);
            // Backtest sur test (déjà fait dans foldResults)
            double bestTestPerf = Double.NEGATIVE_INFINITY;
            for (ComboResult combo : foldResults) {
                double perf = combo.getResult().getRendement();
                if (perf > bestTestPerf) bestTestPerf = perf;
            }
            testPerformances.add(bestTestPerf);
            foldsResults.add(foldResults);
        }
        // Agrégation des résultats de tous les folds
        List<ComboResult> allResults = new ArrayList<>();
        for (List<ComboResult> fold : foldsResults) allResults.addAll(fold);
        // Filtrage des combos non overfit
        List<ComboResult> nonOverfitResults = new ArrayList<>();
        for (ComboResult r : allResults) {
            if (!r.getResult().isFltredOut()) {
                nonOverfitResults.add(r);
            }
        }
        // Sélection du meilleur combo parmi les non-overfit
        ComboResult bestComboNonOverfit = null;
        double bestPerfNonOverfit = Double.NEGATIVE_INFINITY;
        for (ComboResult r : nonOverfitResults) {
            RiskResult res = r.getResult();
            if (res.getRendement() > bestPerfNonOverfit) {
                bestPerfNonOverfit = res.getRendement();
                bestComboNonOverfit = r;
            }
        }

        double sumRendement = 0.0, sumDrawdown = 0.0, sumWinRate = 0.0, sumProfitFactor = 0.0, sumTradeDuration = 0.0;
        int totalTrades = 0;
        ComboResult bestCombo = null;
        // Suppression de la variable bestPerf inutile
        for (ComboResult r : allResults) {
            RiskResult res = r.getResult();
            sumRendement += res.getRendement();
            sumDrawdown += res.getMaxDrawdown();
            sumWinRate += res.getWinRate();
            sumProfitFactor += res.getProfitFactor();
            sumTradeDuration += res.getAvgTradeBars();
            totalTrades += res.getTradeCount();
            // Mise à jour du meilleur combo global (fallback)
            if (bestCombo == null || res.getRendement() > bestCombo.getResult().getRendement()) {
                bestCombo = r;
            }
        }
        int n = allResults.size();
        double avgRendement = n > 0 ? sumRendement / n : 0.0;
        double avgDrawdown = n > 0 ? sumDrawdown / n : 0.0;
        double avgWinRate = n > 0 ? sumWinRate / n : 0.0;
        double avgProfitFactor = n > 0 ? sumProfitFactor / n : 0.0;
        double avgTradeDuration = n > 0 ? sumTradeDuration / n : 0.0;
        double avgTrainPerf = trainPerformances.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double avgTestPerf = testPerformances.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double overfitRatio = avgTestPerf / (avgTrainPerf == 0.0 ? 1.0 : avgTrainPerf);
        boolean isOverfit = (overfitRatio < 0.7 || overfitRatio > 1.3);

        // Si aucun combo non-overfit, fallback sur le meilleur combo global
        ComboResult finalBestCombo = bestComboNonOverfit != null ? bestComboNonOverfit : bestCombo;

        //fait un check final
        com.app.backend.trade.strategy.TradeStrategy entryStrategyCheck = createStrategy(finalBestCombo.getEntryName(), finalBestCombo.getEntryParams());
        com.app.backend.trade.strategy.TradeStrategy exitStrategyCheck = createStrategy(finalBestCombo.getExitName(), finalBestCombo.getExitParams());
        com.app.backend.trade.strategy.StrategieBackTest.CombinedTradeStrategy combined = new com.app.backend.trade.strategy.StrategieBackTest.CombinedTradeStrategy(entryStrategyCheck, exitStrategyCheck);
        BarSeries checkSeries = series.getSubSeries(series.getBarCount() - testWindow, series.getBarCount());
        RiskResult checkR = strategieBackTest.backtestStrategy(combined, checkSeries);

        return WalkForwardResultPro.builder()
                .segmentResults(allResults)
                .avgRendement(avgRendement)
                .avgDrawdown(avgDrawdown)
                .avgWinRate(avgWinRate)
                .avgProfitFactor(avgProfitFactor)
                .avgTradeDuration(avgTradeDuration)
                .totalTrades(totalTrades)
                .bestCombo(finalBestCombo)
                .avgTrainRendement(avgTrainPerf)
                .avgTestRendement(avgTestPerf)
                .overfitRatio(overfitRatio)
                .isOverfit(isOverfit)
                .check(checkR)
                .build();
    }


}
