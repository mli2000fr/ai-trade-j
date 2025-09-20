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


    // Suivi du progrès de la mise à jour Daily Value
    public static class DailyValueUpdateProgress {
        public String status = ""; // en_cours, termine, erreur
        public int updatedItems = 0;
        public int totalItems = 0;
        public long startTime = 0;
        public long endTime = 0;
        public long lastUpdate = 0;
        public String name = "Update Daily Value";
        public String symbol = "";
    }
    private DailyValueUpdateProgress dailyValueProgress = new DailyValueUpdateProgress();

    public DailyValueUpdateProgress getDailyValueProgress() {
        return dailyValueProgress;
    }

    /**
     * Met à jour les valeurs journalières pour tous les symboles actifs en base.
     */
    public void updateDBDailyValuAllSymbols(){
        List<String> listeDbSymbols = this.getAllAssetSymbolsFromDb();
        int error = 0;
        int compteur = 0;
        dailyValueProgress.status = "en_cours";
        dailyValueProgress.updatedItems = 0;
        dailyValueProgress.totalItems = listeDbSymbols.size();
        dailyValueProgress.startTime = System.currentTimeMillis();
        dailyValueProgress.endTime = 0;
        dailyValueProgress.lastUpdate = dailyValueProgress.startTime;
        dailyValueProgress.name = "Update Daily Value";
        dailyValueProgress.symbol = "";
        for(String symbol : listeDbSymbols){
            try{
                int nbInsertion = this.updateDailyValue(symbol);
                TradeUtils.log("updateDailyValue("+symbol+") : " + nbInsertion);
                Thread.sleep(5000);
            }catch(Exception e){
                error++;
                TradeUtils.log("Erreur updateDailyValue("+symbol+") : " + e.getMessage());
                dailyValueProgress.status = "erreur";
            }
            compteur++;
            dailyValueProgress.updatedItems = compteur;
            dailyValueProgress.lastUpdate = System.currentTimeMillis();
            dailyValueProgress.symbol = symbol;
            TradeUtils.log("updateDBDailyValuAllSymbols: compteur "+compteur);
        }
        dailyValueProgress.status = error > 0 ? "erreur" : "termine";
        dailyValueProgress.endTime = System.currentTimeMillis();
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
        String sql = "SELECT symbol FROM trade_ai.alpaca_asset WHERE status = 'active' and eligible = true and filtre_out = false ORDER BY symbol ASC;";
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

    public List<DailyValue> getBougiesTodayBySymbol(String symbol){
        return this.alpacaService.getHistoricalBarsJsonDaysMin(symbol, TradeUtils.getDateToDayStart());
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
    public int updateDailyValue(String symbol) {

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
            java.time.LocalDate start = java.time.LocalDate.parse(dateStart);
            java.time.LocalDate end = TradeUtils.getLastTradingDayBefore(today);
            int compteur = 0;
            while (!start.isAfter(end) && compteur < 3000) {
                java.time.LocalDate trancheEnd = start.plusDays(999);
                if (trancheEnd.isAfter(end)) {
                    trancheEnd = end;
                }
                List<DailyValue> values = this.alpacaService.getHistoricalBars(symbol, start.toString(), trancheEnd.toString());
                if (values != null && !values.isEmpty()) {
                    for(DailyValue dv : values){
                        this.insertDailyValue(symbol, dv);
                        compteur++;
                    }
                }
                start = trancheEnd.plusDays(1);
            }
            return compteur;
        }
        return 0;
    }

    public void updateDBDailyValuAllSymbolsPre(){
        List<String> listeDbSymbols = this.getAllAssetSymbolsFromDb();

        for(String symbol : listeDbSymbols){
            try{
                int compteur = updateDailyValuePre(symbol);
                Thread.sleep(5000);
                logger.info("updateDBDailyValuAllSymbolsPre {} - {}", symbol, compteur);
            }catch(Exception e){
                TradeUtils.log("Erreur updateDBDailyValuAllSymbolsPre("+symbol+") : " + e.getMessage());
            }
        }
    }
    public int updateDailyValuePre(String symbol) {

        // 1. Chercher la date la plus récente pour ce symbol dans la table daily_value
        String sql = "SELECT MIN(date) FROM daily_value WHERE symbol = ?";
        java.sql.Date minDate = null;
        try {
            minDate = jdbcTemplate.query(sql, ps -> ps.setString(1, symbol), rs -> {
                if (rs.next()) {
                    return rs.getDate(1);
                }
                return null;
            });
        } catch (Exception e) {
            logger.warn("Aucune date trouvée pour le symbole {} dans daily_value ou erreur SQL: {}", symbol, e.getMessage());
        }
        //date de début
        LocalDate dateStart = LocalDate.now().minusDays(TradeConstant.HISTORIQUE_DAILY_VALUE);
        java.time.LocalDate minDateKnown = TradeUtils.getLastTradingDayBefore(java.time.LocalDate.now());
        if (minDate != null) {
            minDateKnown = minDate.toLocalDate();
        }
        java.time.LocalDate minDateTradingDay = TradeUtils.getLastTradingDayBefore(minDateKnown);

        logger.info("updateDailyValuePre start {} - {}", TradeUtils.getDateString(dateStart), TradeUtils.getDateString(minDateTradingDay));
        if (dateStart.isBefore(minDateTradingDay)) {
            int compteur = 0;
            LocalDate currentStart = dateStart;
            while (currentStart.isBefore(minDateTradingDay)) {
                // Calculer la date de fin (max 1000 jours ou minDateTradingDay)
                LocalDate currentEnd = currentStart.plusDays(999);
                if (currentEnd.isAfter(minDateTradingDay)) {
                    currentEnd = minDateTradingDay;
                }
                logger.info("updateDailyValuePre {} - {}", TradeUtils.getDateString(currentStart), TradeUtils.getDateString(currentEnd));
                List<DailyValue> values = this.alpacaService.getHistoricalBars(symbol, TradeUtils.getDateString(currentStart), TradeUtils.getDateString(currentEnd));
                if (values != null && !values.isEmpty()) {
                    for(DailyValue dv : values){
                        this.insertDailyValue(symbol, dv);
                        compteur++;
                    }
                }
                // Passer au prochain intervalle
                currentStart = currentEnd.plusDays(1);
            }
            logger.info("updateDailyValuePre fin {} - {}", TradeUtils.getDateString(dateStart), TradeUtils.getDateString(currentStart));
            return compteur;
        }else{
            return  0;
        }
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
                            if(result == null) {
                                error.incrementAndGet();
                                //update assert filtre_out false
                                String updateSql = "UPDATE alpaca_asset SET filtre_out = TRUE WHERE symbol = ?";
                                jdbcTemplate.update(updateSql, symbol);
                            }else{
                                this.saveBestInOutStrategy(symbol, result);
                            }
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
                    rendement_check = ?,
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
                    sharpe_ratio = ?
                    stability_score = ?,
                    score_swing_trade = ?,
                    score_swing_trade_check = ?,
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
                    best.check.rendement,
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
                best.result.sharpeRatio,
                best.result.stabilityScore,
                best.result.scoreSwingTrade,
                best.check.scoreSwingTrade,
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
                    rendement, rendement_check, rendement_sum, rendement_diff, rendement_score, trade_count, win_rate, max_drawdown, avg_pnl, profit_factor, avg_trade_bars, max_trade_gain, max_trade_loss,
                    sharpe_ratio, stability_score, score_swing_trade, score_swing_trade_check, fltred_out, initial_capital, risk_per_trade, stop_loss_pct, 
                    take_profit_pct, nb_simples, check_result, created_date, updated_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """;
            jdbcTemplate.update(insertSql,
                symbol,
                best.entryName,
                entryParamsJson,
                best.exitName,
                exitParamsJson,
                best.result.rendement,
                best.check.rendement,
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
                    best.result.sharpeRatio,
                    best.result.stabilityScore,
                best.result.scoreSwingTrade,
                best.check.scoreSwingTrade,
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
                                .fltredOut(rs.getBoolean("fltred_out"))
                                .sharpeRatio(rs.getDouble("sharpe_ratio"))
                                .stabilityScore(rs.getDouble("stability_score"))
                                .build())
                        .build();
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
        String orderBy = (sort == null || sort.isBlank()) ? "rendement_score" : sort;
        String searchSQL = "";
        if(search != null && !search.isEmpty()){
            orderBy = "s.symbol ASC"; // en cas de recherche, on trie par symbole
            searchSQL = "s.symbol in ("+"'"+search.replaceAll(" ", "").replaceAll(",", "','")+"'"+") and";
        }else{
            orderBy = "s." + orderBy + " DESC";
        }
        String sql = "SELECT s.*, a.name FROM best_in_out_single_strategy s JOIN alpaca_asset a ON s.symbol = a.symbol WHERE "+ searchSQL +" s.profit_factor <> 0 AND s.win_rate < 1 and filtre_out = false";
        if (filtered != null && filtered) {
            sql += " AND fltred_out = false";
        }
        sql += " ORDER BY " + orderBy;
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
                    .name(rs.getString("name"))
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
                            .sharpeRatio(rs.getDouble("sharpe_ratio"))
                            .stabilityScore(rs.getDouble("stability_score"))
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
        if(listeValus.size() < TradeConstant.NOMBRE_TOTAL_BOUGIES_MIN_OPTIM){
            throw new IllegalArgumentException("Pas assez de données pour le symbole " + symbol + " (" + listeValus.size() + " bougies, minimum " + TradeConstant.NOMBRE_TOTAL_BOUGIES_MIN_OPTIM + ")");
        }
        BarSeries series = TradeUtils.mapping(listeValus);
        // Utilisation du swingParams de la classe (modifiable si besoin)
        WalkForwardResultPro walkForwardResultPro =  this.optimseStrategy(series, swingParams);
        if (walkForwardResultPro == null || walkForwardResultPro.getBestCombo() == null) {
            return null;
        }
        double rendementSum = walkForwardResultPro.getBestCombo().getResult().getRendement() + walkForwardResultPro.getBestCombo().getCheckResult().getRendement();
        double rendementDiff = walkForwardResultPro.getBestCombo().getResult().getRendement() - walkForwardResultPro.getBestCombo().getCheckResult().getRendement();
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
                .check(walkForwardResultPro.getBestCombo().getCheckResult()).build();
    }


    /**
     * Optimisation walk-forward professionnelle pour le swing trade.
     * Les paramètres sont des pourcentages du nombre total de bougies.
     * @param series série de prix
     * @param swingParams paramètres d'optimisation swing trade
     * @return WalkForwardResultPro
     */
    public WalkForwardResultPro optimseStrategy(BarSeries series, SwingTradeOptimParams swingParams) {
        int totalBars = series.getBarCount();
        int kFolds = 3; // 3 folds fixes
        List<List<ComboResult>> foldsResults = new ArrayList<>();
        List<List<ComboResult>> foldsAllCombos = new ArrayList<>();
        List<Double> trainPerformances = new ArrayList<>();
        List<Double> testPerformances = new ArrayList<>();
        // Définition des indices pour chaque fold
        int[][] foldIndices = new int[][] {
            // Fold 0 : optim 0-35%, test 35-50%
            {0, (int)Math.round(totalBars*0.35), (int)Math.round(totalBars*0.35), (int)Math.round(totalBars*0.5)},
            // Fold 1 : optim 15-50%, test 50-65%
            {(int)Math.round(totalBars*0.15), (int)Math.round(totalBars*0.5), (int)Math.round(totalBars*0.5), (int)Math.round(totalBars*0.65)},
            // Fold 2 : optim 30-65%, test 65-80%
            {(int)Math.round(totalBars*0.3), (int)Math.round(totalBars*0.65), (int)Math.round(totalBars*0.65), (int)Math.round(totalBars*0.8)}
        };
        for (int fold = 0; fold < kFolds; fold++) {
            int optimStart = foldIndices[fold][0];
            int optimEnd = foldIndices[fold][1];
            int testStart = foldIndices[fold][2];
            int testEnd = foldIndices[fold][3];
            BarSeries optimSeries = series.getSubSeries(optimStart, optimEnd);
            BarSeries testSeries = series.getSubSeries(testStart, testEnd);
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
            List<ComboResult> foldAllCombos = new ArrayList<>();
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
                    result.setFltredOut(isOverfitCombo);
                    ComboResult combo = ComboResult.builder()
                            .entryName(entryName)
                            .entryParams(entryParams)
                            .exitName(exitName)
                            .exitParams(exitParams)
                            .result(result)
                            .trainRendement(trainResult.getRendement())
                            .build();
                    //fait un check final
                    BarSeries checkSeries = series.getSubSeries(series.getBarCount() - (int)Math.round(totalBars*0.2), series.getBarCount()); // dernier 20% pour le check
                    RiskResult checkR = (combined != null) ? strategieBackTest.backtestStrategy(combined, checkSeries) : null;
                    combo.setCheckResult(checkR);
                    foldAllCombos.add(combo);
                    if(!isOverfitCombo){
                        foldResults.add(combo);
                    }
                }
            }
            trainPerformances.add(bestTrainPerf);
            foldsResults.add(foldResults);
            foldsAllCombos.add(foldAllCombos);
        }
        // Agrégation des résultats de tous les folds
        List<ComboResult> allResults = new ArrayList<>();
        for (List<ComboResult> fold : foldsAllCombos) allResults.addAll(fold);
        List<ComboResult> filteredResults = new ArrayList<>();
        for (List<ComboResult> fold : foldsResults) filteredResults.addAll(fold);
        // Calcul des moyennes/statistiques pour le builder
        double sumRendement = 0.0, sumDrawdown = 0.0, sumWinRate = 0.0, sumProfitFactor = 0.0, sumTradeDuration = 0.0;
        int totalTrades = 0;
        for (ComboResult r : allResults) {
            RiskResult res = r.getResult();
            sumRendement += res.getRendement();
            sumDrawdown += res.getMaxDrawdown();
            sumWinRate += res.getWinRate();
            sumProfitFactor += res.getProfitFactor();
            sumTradeDuration += res.getAvgTradeBars();
            totalTrades += res.getTradeCount();
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
        // Sélection du meilleur combo swing trade
        List<ComboResult> scoredCombos = strategieBackTest.computeSwingTradeScores(filteredResults);
        ComboResult bestScoreResult = null;
        double maxScore = Double.NEGATIVE_INFINITY;
        // Sélection hybride/fallback
        if (!filteredResults.isEmpty()) {
            // Sélectionne le meilleur combo non-overfit
            for (ComboResult scoreResult : scoredCombos) {
                if (filteredResults.contains(scoreResult)) {
                    if (scoreResult.getResult().getScoreSwingTrade() > maxScore) {
                        maxScore = scoreResult.getResult().getScoreSwingTrade();
                        bestScoreResult = scoreResult;
                    }
                }
            }
        } else {
            // Fallback: sélectionne le combo dont le ratio overfit est le plus proche de 1
            double minOverfitDist = Double.POSITIVE_INFINITY;
            for (ComboResult scoreResult : scoredCombos) {
                RiskResult res = scoreResult.getResult();
                double trainRendement = scoreResult.getTrainRendement();
                double overfitRatioCombo = trainRendement == 0.0 ? 1.0 : res.getRendement() / trainRendement;
                double dist = Math.abs(overfitRatioCombo - 1.0);
                if (dist < minOverfitDist) {
                    minOverfitDist = dist;
                    bestScoreResult = scoreResult;
                }
            }
        }


        return WalkForwardResultPro.builder()
                .segmentResults(allResults)
                .avgRendement(avgRendement)
                .avgDrawdown(avgDrawdown)
                .avgWinRate(avgWinRate)
                .avgProfitFactor(avgProfitFactor)
                .avgTradeDuration(avgTradeDuration)
                .totalTrades(totalTrades)
                .bestCombo(bestScoreResult)
                .avgTrainRendement(avgTrainPerf)
                .avgTestRendement(avgTestPerf)
                .overfitRatio(overfitRatio)
                .isOverfit(isOverfit)
                .check(bestScoreResult.getCheckResult())
                .build();
    }




    /**
     * Structure pour les pondérations du scoring swing trade.
     */
    public static class SwingTradeScoreWeights {
        public double rendement = 0.35;
        public double sharpe = 0.25;
        public double drawdown = 0.25;
        public double stability = 0.15;
        public SwingTradeScoreWeights() {}
        public SwingTradeScoreWeights(double r, double s, double d, double st) {
            rendement = r; sharpe = s; drawdown = d; stability = st;
        }
    }
}
