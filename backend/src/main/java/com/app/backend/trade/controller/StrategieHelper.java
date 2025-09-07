package com.app.backend.trade.controller;

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
import org.ta4j.core.BaseBarSeries;
import org.ta4j.core.Rule;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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
     * Retourne un signal d'achat/vente combiné selon les stratégies actives et le mode choisi.
     * @param series série de prix (BarSeries)
     * @param isEntry true pour entrée (achat), false pour sortie (vente)
     * @return true si le signal est validé
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


    public boolean testCombinedSignalOnClosePrices(String symbol, boolean isEntry) {

        List<DailyValue> listeValues = alpacaService.getHistoricalBars(symbol, TradeUtils.getStartDate(700));
        BarSeries series = toBarSeries(listeValues);
        int lastIndex = series.getEndIndex();
        return getCombinedSignal(series, lastIndex, isEntry);
    }

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
     * Liste des jours fériés boursiers (à adapter selon le marché, ici exemple NYSE 2025)
     */
    private static final java.util.Set<java.time.LocalDate> MARKET_HOLIDAYS = java.util.Set.of(
        java.time.LocalDate.of(2025, 1, 1),   // New Year's Day
        java.time.LocalDate.of(2025, 1, 20),  // Martin Luther King Jr. Day
        java.time.LocalDate.of(2025, 2, 17),  // Presidents' Day
        java.time.LocalDate.of(2025, 4, 18),  // Good Friday
        java.time.LocalDate.of(2025, 5, 26),  // Memorial Day
        java.time.LocalDate.of(2025, 7, 4),   // Independence Day
        java.time.LocalDate.of(2025, 9, 1),   // Labor Day
        java.time.LocalDate.of(2025, 11, 27), // Thanksgiving Day
        java.time.LocalDate.of(2025, 12, 25)  // Christmas Day
    );

    /**
     * Retourne le dernier jour de cotation avant la date passée (week-end et jours fériés inclus).
     */
    private java.time.LocalDate getLastTradingDayBefore(java.time.LocalDate date) {
        java.time.LocalDate d = date.minusDays(1);
        while (d.getDayOfWeek() == java.time.DayOfWeek.SATURDAY ||
               d.getDayOfWeek() == java.time.DayOfWeek.SUNDAY ||
               MARKET_HOLIDAYS.contains(d)) {
            d = d.minusDays(1);
        }
        return d;
    }

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
            java.time.LocalDate lastTradingDay = getLastTradingDayBefore(today);
            java.time.LocalDate lastKnown = lastDate.toLocalDate();
            // Si la dernière date connue est le dernier jour de cotation, la base est à jour
            if (lastKnown.isEqual(lastTradingDay) || lastKnown.isAfter(lastTradingDay)) {
                isUpToDate = true;
            }
            // Sinon, on ajoute un jour à la date la plus récente
            dateStart = lastKnown.plusDays(1).toString(); // format YYYY-MM-DD
        }
        if (!isUpToDate) {
            List<DailyValue> listeValues = this.alpacaService.getHistoricalBars(symbol, dateStart);
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
        return this.mapping(getDailyValuesFromDb(symbol, limit));
    }


    public void updateDBDailyValuAllSymbolsComplement(){
        List<String> listeDbSymbols = this.getAllAssetSymbolsComplementFromDb();
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

    public void updateDBAssets(){
        List<AlpacaAsset> listeSymbols = this.alpacaService.getIexSymbolsFromAlpaca();
        this.saveSymbolsToDatabase(listeSymbols);
    }

    /**
     * Récupère la liste des DailyValue pour un symbole donné depuis la table daily_value
     * @param symbol le symbole de l'action
     * @return Liste des DailyValue triées par date croissante
     */
    public List<DailyValue> getDailyValuesFromDatabase(String symbol) {
        return this.getDailyValuesFromDb(symbol, TradeConstant.NOMBRE_TOTAL_BOUGIES);
    }


    /**
     * Convertit une liste de DailyValue en BarSeries (ta4j).
     */
    private BarSeries toBarSeries(List<DailyValue> values) {
        BarSeries series = new BaseBarSeries();
        for (DailyValue v : values) {
            try {
                series.addBar(
                        ZonedDateTime.parse(v.getDate()),
                        Double.parseDouble(v.getOpen()),
                        Double.parseDouble(v.getHigh()),
                        Double.parseDouble(v.getLow()),
                        Double.parseDouble(v.getClose()),
                        Double.parseDouble(v.getVolume())
                );
            } catch (Exception e) {
                TradeUtils.log("Erreur conversion DailyValue en BarSeries: " + e.getMessage());
            }
        }
        return series;
    }


    /**
     * Insère une ligne dans la table daily_value avec symbol et une instance de DailyValue.
     * La colonne date est stockée en type DATE (MySQL), donc conversion si nécessaire.
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
     * Récupère tous les symboles depuis la table alpaca_asset
     */
    public List<String> getAllAssetSymbolsFromDb() {
        String sql = "SELECT symbol FROM alpaca_asset WHERE status = 'active'";
        return jdbcTemplate.queryForList(sql, String.class);
    }


    /**
     * Récupère tous les symboles depuis la table alpaca_asset
     */
    public List<String> getAllAssetSymbolsEligibleFromDb() {
        String sql = "SELECT symbol FROM trade_ai.alpaca_asset WHERE status = 'active' and eligible = true;";
        return jdbcTemplate.queryForList(sql, String.class);
    }

    /**
     * Récupère tous les symboles depuis la table alpaca_asset
     */
    public List<String> getAllAssetSymbolsComplementFromDb() {
        String sql = "SELECT symbol FROM trade_ai.alpaca_asset WHERE symbol NOT IN (SELECT symbol FROM trade_ai.daily_value);";
        return jdbcTemplate.queryForList(sql, String.class);
    }

    /**
     * Récupère la liste des DailyValue pour un symbole donné depuis la table daily_value
     * @param symbol le symbole de l'action
     * @return Liste des DailyValue triées par date croissante
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
            dateStart = TradeUtils.getStartDate(800);
        } else {
            java.time.LocalDate lastTradingDay = getLastTradingDayBefore(today);
            java.time.LocalDate lastKnown = lastDate.toLocalDate();
            // Si la dernière date connue est le dernier jour de cotation, la base est à jour
            if (lastKnown.isEqual(lastTradingDay) || lastKnown.isAfter(lastTradingDay)) {
                isUpToDate = true;
            }
            // Sinon, on ajoute un jour à la date la plus récente
            dateStart = lastKnown.plusDays(1).toString(); // format YYYY-MM-DD
        }
        if (!isUpToDate) {
            return this.alpacaService.getHistoricalBars(symbol, dateStart);
        }
        return  new ArrayList<>();
    }

    /**
     * Convertit une liste de DailyValue en BarSeries (ta4j).
     */

    public BarSeries mapping(List<DailyValue> listeValues) {

        BarSeries series = new BaseBarSeries();

        for (DailyValue dailyValue : listeValues) {
            try {
                // Convertir la date du format "2025-09-03" vers ZonedDateTime
                ZonedDateTime dateTime;
                if (dailyValue.getDate().length() == 10) {
                    // Format "YYYY-MM-DD" -> ajouter l'heure par défaut
                    dateTime = java.time.LocalDate.parse(dailyValue.getDate())
                            .atStartOfDay(java.time.ZoneId.systemDefault());
                } else {
                    // Format ISO complet
                    dateTime = ZonedDateTime.parse(dailyValue.getDate());
                }

                series.addBar(
                        dateTime,
                        Double.parseDouble(dailyValue.getOpen()),
                        Double.parseDouble(dailyValue.getHigh()),
                        Double.parseDouble(dailyValue.getLow()),
                        Double.parseDouble(dailyValue.getClose()),
                        Double.parseDouble(dailyValue.getVolume())
                );
            } catch (Exception e) {
                logger.warn("Erreur conversion DailyValue en BarSeries pour la date {}: {}",
                        dailyValue.getDate(), e.getMessage());
            }
        }

        return series;
    }




    /**
     * Méthodes utilitaires pour la conversion JSON
     */
    private String convertMapToJson(java.util.Map<String, Double> map) {
        if (map == null) return null;
        com.google.gson.JsonObject jsonObj = new com.google.gson.JsonObject();
        map.forEach(jsonObj::addProperty);
        return new com.google.gson.Gson().toJson(jsonObj);
    }

    private String convertDetailedResults (java.util.Map<String, StrategieBackTest.RiskResult> detailedResults) {
        if (detailedResults == null) return null;
        com.google.gson.JsonObject jsonObj = new com.google.gson.JsonObject();
        detailedResults.forEach((key, result) -> {
            com.google.gson.JsonObject resultObj = new com.google.gson.JsonObject();
            resultObj.addProperty("rendement", result.rendement);
            resultObj.addProperty("maxDrawdown", result.maxDrawdown);
            resultObj.addProperty("tradeCount", result.tradeCount);
            resultObj.addProperty("winRate", result.winRate);
            resultObj.addProperty("avgPnL", result.avgPnL);
            resultObj.addProperty("profitFactor", result.profitFactor);
            resultObj.addProperty("avgTradeBars", result.avgTradeBars);
            resultObj.addProperty("maxTradeGain", result.maxTradeGain);
            resultObj.addProperty("maxTradeLoss", result.maxTradeLoss);
            jsonObj.add(key, resultObj);
        });
        return new com.google.gson.Gson().toJson(jsonObj);
    }

    private java.util.Map<String, Double> convertJsonToPerformanceMap(String json) {
        if (json == null) return new java.util.HashMap<>();
        try {
            com.google.gson.JsonObject jsonObj = new com.google.gson.JsonParser().parse(json).getAsJsonObject();
            java.util.Map<String, Double> map = new java.util.HashMap<>();
            jsonObj.entrySet().forEach(entry ->
                map.put(entry.getKey(), entry.getValue().getAsDouble()));
            return map;
        } catch (Exception e) {
            logger.warn("Erreur conversion JSON vers Map<String, Double>: {}", e.getMessage());
            return new java.util.HashMap<>();
        }
    }

    public void testAllCrossedStrategies(String symbol){
        BestInOutStrategy bestCombo = optimseBestInOutByWalkForward(symbol);
        //this.saveBestInOutStrategy(symbol, result);

        //BestInOutStrategy bestCombo = this.getBestInOutStrategy(symbol);
        System.out.println("=== RESTITUTION CROISÉS IN/OUT ===" + symbol);
        if (bestCombo != null) {
            System.out.println("IN: " + bestCombo.entryName + " | OUT: " + bestCombo.exitName + " | Rendement: " + String.format("%.4f", bestCombo.result.rendement * 100) + "% | Trades: " + bestCombo.result.tradeCount);
            com.google.gson.Gson gson = new com.google.gson.GsonBuilder().setPrettyPrinting().create();
            System.out.println("Paramètres IN: " + gson.toJson(bestCombo.entryParams));
            System.out.println("Paramètres OUT: " + gson.toJson(bestCombo.exitParams));
        }
    }

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
                        nbInsert.incrementAndGet();
                        BestInOutStrategy result = optimseBestInOutByWalkForward(symbol);
                        this.saveBestInOutStrategy(symbol, result);
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
     * Retourne la série de prix découpée en deux parties selon les pourcentages définis.
     * @param series la série complète
     * @return tableau [BarSeries optimisation, BarSeries test]
     */
    public BarSeries[] splitSeriesForWalkForward(BarSeries series) {
        int total = series.getBarCount();
        int nOptim = (int) Math.round(total * TradeConstant.PC_OPTIM);
        int nTest = total - nOptim;
        if (nOptim < 1 || nTest < 1) throw new IllegalArgumentException("Découpage walk-forward impossible : pas assez de données");
        BarSeries optimSeries = new BaseBarSeries();
        BarSeries testSeries = new BaseBarSeries();
        for (int i = 0; i < nOptim; i++) {
            optimSeries.addBar(series.getBar(i));
        }
        for (int i = nOptim; i < total; i++) {
            testSeries.addBar(series.getBar(i));
        }
        return new BarSeries[] {optimSeries, testSeries};
    }

    /**
     * Teste automatiquement toutes les combinaisons croisées de stratégies pour in (entrée) et out (sortie).
     * Utilise le découpage walk-forward défini par les constantes.
     */
    public BestInOutStrategy optimseBestInOutByWalkForward(String symbol) {
        BarSeries series = this.mapping(this.getDailyValuesFromDb(symbol, TradeConstant.NOMBRE_TOTAL_BOUGIES));
        BarSeries[] split = splitSeriesForWalkForward(series);
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
                StrategieBackTest.RiskResult result = strategieBackTest.backtestStrategyRisk(combined, testSeries);
                System.out.println("IN: " + entryName + " " + gson.toJson(entryParams) +
                                   " | OUT: " + exitName + " " + gson.toJson(exitParams) +
                                   " | Rendement: " + String.format("%.4f", result.rendement * 100) + "%"
                                   + " | Trades: " + result.tradeCount
                                   + " | WinRate: " + String.format("%.2f", result.winRate * 100) + "%"
                                   + " | Drawdown: " + String.format("%.2f", result.maxDrawdown * 100) + "%");
                if (result.rendement > bestPerf) {
                    bestPerf = result.rendement;
                    bestCombo = new BestInOutStrategy(symbol, entryName, entryParams, exitName, exitParams, result, StrategieBackTest.INITIAL_CAPITAL, StrategieBackTest.RISK_PER_TRADE, StrategieBackTest.STOP_LOSS_PCT, StrategieBackTest.TAKE_PROFIL_PCT);
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
     * Sauvegarde la meilleure combinaison in/out pour un symbole
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
                    initial_capital = ?,
                    risk_per_trade = ?,
                    stop_loss_pct = ?,
                    take_profit_pct = ?,
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
                best.initialCapital,
                best.riskPerTrade,
                best.stopLossPct,
                best.takeProfitPct,
                symbol
            );
        } else {
            // Insertion
            String insertSql = """
                INSERT INTO best_in_out_single_strategy (
                    symbol, entry_strategy_name, entry_strategy_params,
                    exit_strategy_name, exit_strategy_params,
                    rendement, trade_count, win_rate, max_drawdown, avg_pnl, profit_factor, avg_trade_bars, max_trade_gain, max_trade_loss,
                    initial_capital, risk_per_trade, stop_loss_pct, take_profit_pct,
                    created_date, updated_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
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
                best.initialCapital,
                best.riskPerTrade,
                best.stopLossPct,
                best.takeProfitPct,
                java.sql.Date.valueOf(java.time.LocalDate.now())
            );
        }
    }

    /**
     * Récupère la meilleure combinaison in/out pour un symbole
     */
    public BestInOutStrategy getBestInOutStrategy(String symbol) {
        String sql = "SELECT * FROM best_in_out_single_strategy WHERE symbol = ?";
        try {
            return jdbcTemplate.queryForObject(sql, (rs, rowNum) -> {
                String entryName = rs.getString("entry_strategy_name");
                String entryParamsJson = rs.getString("entry_strategy_params");
                String exitName = rs.getString("exit_strategy_name");
                String exitParamsJson = rs.getString("exit_strategy_params");
                StrategieBackTest.RiskResult result = new StrategieBackTest.RiskResult(
                    rs.getDouble("rendement"),
                    rs.getDouble("max_drawdown"),
                    rs.getInt("trade_count"),
                    rs.getDouble("win_rate"),
                    rs.getDouble("avg_pnl"),
                    rs.getDouble("profit_factor"),
                    rs.getDouble("avg_trade_bars"),
                    rs.getDouble("max_trade_gain"),
                    rs.getDouble("max_trade_loss")
                );
                Object entryParams = parseStrategyParams(entryName, entryParamsJson);
                Object exitParams = parseStrategyParams(exitName, exitParamsJson);
                double initialCapital = rs.getDouble("initial_capital");
                double riskPerTrade = rs.getDouble("risk_per_trade");
                double stopLossPct = rs.getDouble("stop_loss_pct");
                double takeProfitPct = rs.getDouble("take_profit_pct");
                return new BestInOutStrategy(symbol, entryName, entryParams, exitName, exitParams, result, initialCapital, riskPerTrade, stopLossPct, takeProfitPct);
            }, symbol);
        } catch (org.springframework.dao.EmptyResultDataAccessException e) {
            logger.warn("Aucun BestInOutStrategy trouvé pour le symbole: {}", symbol);
            return null;
        }
    }

    /**
     * Utilitaire pour parser les paramètres JSON selon le type de stratégie
     */
    private Object parseStrategyParams(String name, String json) {
        com.google.gson.Gson gson = new com.google.gson.Gson();
        switch (name) {
            case "Improved Trend":
                return gson.fromJson(json, StrategieBackTest.ImprovedTrendFollowingParams.class);
            case "SMA Crossover":
                return gson.fromJson(json, StrategieBackTest.SmaCrossoverParams.class);
            case "RSI":
                return gson.fromJson(json, StrategieBackTest.RsiParams.class);
            case "Breakout":
                return gson.fromJson(json, StrategieBackTest.BreakoutParams.class);
            case "MACD":
                return gson.fromJson(json, StrategieBackTest.MacdParams.class);
            case "Mean Reversion":
                return gson.fromJson(json, StrategieBackTest.MeanReversionParams.class);
            default:
                return null;
        }
    }



    public List<BestInOutStrategy> getBestPerfActions(Integer limit){
        String sql = "SELECT * FROM best_in_out_single_strategy ORDER BY rendement DESC";
        if (limit != null && limit > 0) {
            sql += " LIMIT " + limit;
        }
        List<BestInOutStrategy> results = jdbcTemplate.query(sql, (rs, rowNum) -> {
            String symbol = rs.getString("symbol");
            String entryName = rs.getString("entry_strategy_name");
            String entryParamsJson = rs.getString("entry_strategy_params");
            String exitName = rs.getString("exit_strategy_name");
            String exitParamsJson = rs.getString("exit_strategy_params");
            StrategieBackTest.RiskResult result = new StrategieBackTest.RiskResult(
                rs.getDouble("rendement"),
                rs.getDouble("max_drawdown"),
                rs.getInt("trade_count"),
                rs.getDouble("win_rate"),
                rs.getDouble("avg_pnl"),
                rs.getDouble("profit_factor"),
                rs.getDouble("avg_trade_bars"),
                rs.getDouble("max_trade_gain"),
                rs.getDouble("max_trade_loss")
            );
            Object entryParams = parseStrategyParams(entryName, entryParamsJson);
            Object exitParams = parseStrategyParams(exitName, exitParamsJson);
            double initialCapital = rs.getDouble("initial_capital");
            double riskPerTrade = rs.getDouble("risk_per_trade");
            double stopLossPct = rs.getDouble("stop_loss_pct");
            double takeProfitPct = rs.getDouble("take_profit_pct");
            return new BestInOutStrategy(symbol, entryName, entryParams, exitName, exitParams, result, initialCapital, riskPerTrade, stopLossPct, takeProfitPct);
        });
        return results;
    }

    public enum SignalType { BUY, SELL, NONE; }

    /**
     * Retourne le signal d'achat/vente pour un symbole selon la meilleure stratégie IN/OUT.
     * @param symbol le symbole à analyser
     * @return SignalType (BUY, SELL, NONE)
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
