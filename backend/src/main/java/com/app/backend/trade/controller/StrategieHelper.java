package com.app.backend.trade.controller;

import com.app.backend.trade.model.*;
import com.app.backend.trade.model.alpaca.AlpacaAsset;
import com.app.backend.trade.service.*;
import com.app.backend.trade.strategy.StrategieBackTest;
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
import java.util.Collections;
import java.util.List;


@Controller
public class StrategieHelper {

    private static final Logger logger = LoggerFactory.getLogger(StrategieHelper.class);
    private final AlpacaService alpacaService;
    private final StrategyService strategyService;
    private final JdbcTemplate jdbcTemplate;
    private final StrategieBackTest strategieBackTest;

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
        return this.getDailyValuesFromDb(symbol);
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
    public List<String> getAllAssetSymbolsComplementFromDb() {
        String sql = "SELECT symbol FROM trade_ai.alpaca_asset WHERE symbol NOT IN (SELECT symbol FROM trade_ai.daily_value);";
        return jdbcTemplate.queryForList(sql, String.class);
    }

    /**
     * Récupère la liste des DailyValue pour un symbole donné depuis la table daily_value
     * @param symbol le symbole de l'action
     * @return Liste des DailyValue triées par date croissante
     */
    public List<DailyValue> getDailyValuesFromDb(String symbol) {
        String sql = "SELECT date, open, high, low, close, volume, number_of_trades, volume_weighted_average_price " +
                "FROM daily_value WHERE symbol = ? ORDER BY date DESC LIMIT 500";

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
        if (lastDate == null) {
            // Si aucune ligne trouvée, on prend la date de start par défaut
            dateStart = TradeUtils.getStartDate(800);
        } else {
            // Sinon, on ajoute un jour à la date la plus récente
            java.time.LocalDate nextDay = lastDate.toLocalDate().plusDays(1);
            dateStart = nextDay.toString(); // format YYYY-MM-DD
        }
        // 2. Appeler getHistoricalBars avec la date de start calculée
        return this.alpacaService.getHistoricalBars(symbol, dateStart);
    }

    public void testAnalyse() {
        BarSeries series = this.mapping(this.getDailyValuesFromDb("AAPL"));

        System.out.println("=== DIAGNOSTIC DE LA SÉRIE ===");
        System.out.println("Nombre de bougies: " + series.getBarCount());
        System.out.println("Date début: " + series.getFirstBar().getEndTime());
        System.out.println("Date fin: " + series.getLastBar().getEndTime());
        System.out.println("Prix min: " + series.getBar(series.getBeginIndex()).getLowPrice());
        System.out.println("Prix max: " + series.getBar(series.getBeginIndex()).getHighPrice());

        System.out.println("\n=== TEST TREND FOLLOWING (original) ===");
        List<StrategieBackTest.WalkForwardResult> walkResults = strategieBackTest.runWalkForwardBacktestTrendFollowing(series, 300, 200, 5, 50);
        StrategieBackTest.printWalkForwardResults(walkResults);

        System.out.println("\n=== TEST IMPROVED TREND FOLLOWING (nouvelle version améliorée) ===");
        List<StrategieBackTest.WalkForwardResult> improvedTrendResults = strategieBackTest.runWalkForwardBacktestImprovedTrendFollowing(
            series, 300, 200,
            10, 30,    // trendMin, trendMax
            5, 15,     // shortMaMin, shortMaMax
            15, 25,    // longMaMin, longMaMax
            0.001, 0.01, 0.002  // thresholdMin, thresholdMax, thresholdStep
        );
        StrategieBackTest.printWalkForwardResultsGeneric(improvedTrendResults);

        System.out.println("\n=== TEST SMA CROSSOVER ===");
        List<StrategieBackTest.WalkForwardResult> smaCrossResults = strategieBackTest.runWalkForwardBacktestSmaCrossover(series, 300, 200, 5, 20, 10, 50);
        StrategieBackTest.printWalkForwardResultsSmaCrossover(smaCrossResults);

        System.out.println("\n=== TEST RSI ===");
        List<StrategieBackTest.WalkForwardResult> rsiResults = strategieBackTest.runWalkForwardBacktestRsi(series, 300, 200, 10, 20, 20, 40, 5, 60, 80, 5);
        StrategieBackTest.printWalkForwardResultsRsi(rsiResults);

        System.out.println("\n=== TEST BREAKOUT ===");
        List<StrategieBackTest.WalkForwardResult> breakoutResults = strategieBackTest.runWalkForwardBacktestBreakout(series, 300, 200, 5, 50);
        StrategieBackTest.printWalkForwardResultsGeneric(breakoutResults);

        System.out.println("\n=== TEST MACD ===");
        List<StrategieBackTest.WalkForwardResult> macdResults = strategieBackTest.runWalkForwardBacktestMacd(series, 300, 200, 8, 16, 20, 30, 6, 12);
        StrategieBackTest.printWalkForwardResultsMacd(macdResults);

        System.out.println("\n=== TEST MEAN REVERSION ===");
        List<StrategieBackTest.WalkForwardResult> meanRevResults = strategieBackTest.runWalkForwardBacktestMeanReversion(series, 300, 200, 10, 30, 1.0, 5.0, 0.5);
        StrategieBackTest.printWalkForwardResultsGeneric(meanRevResults);

        System.out.println("\n=== COMPARAISON DES PERFORMANCES ===");
        System.out.println("Trend Following original: " + (walkResults.isEmpty() ? "Aucun résultat" : walkResults.get(0).result.tradeCount + " trades"));
        System.out.println("Improved Trend Following: " + (improvedTrendResults.isEmpty() ? "Aucun résultat" : improvedTrendResults.get(0).result.tradeCount + " trades"));
        System.out.println("SMA Crossover: " + (smaCrossResults.isEmpty() ? "Aucun résultat" : smaCrossResults.get(0).result.tradeCount + " trades"));
        System.out.println("RSI: " + (rsiResults.isEmpty() ? "Aucun résultat" : rsiResults.get(0).result.tradeCount + " trades"));
        System.out.println("Breakout: " + (breakoutResults.isEmpty() ? "Aucun résultat" : breakoutResults.get(0).result.tradeCount + " trades"));
        System.out.println("MACD: " + (macdResults.isEmpty() ? "Aucun résultat" : macdResults.get(0).result.tradeCount + " trades"));
        System.out.println("Mean Reversion: " + (meanRevResults.isEmpty() ? "Aucun résultat" : meanRevResults.get(0).result.tradeCount + " trades"));

        System.out.println("\n=== CLASSEMENT PAR RENDEMENT ===");
        java.util.Map<String, Double> performances = new java.util.HashMap<>();
        if (!walkResults.isEmpty()) performances.put("Trend Following", walkResults.get(0).result.rendement);
        if (!improvedTrendResults.isEmpty()) performances.put("Improved Trend", improvedTrendResults.get(0).result.rendement);
        if (!smaCrossResults.isEmpty()) performances.put("SMA Crossover", smaCrossResults.get(0).result.rendement);
        if (!rsiResults.isEmpty()) performances.put("RSI", rsiResults.get(0).result.rendement);
        if (!breakoutResults.isEmpty()) performances.put("Breakout", breakoutResults.get(0).result.rendement);
        if (!macdResults.isEmpty()) performances.put("MACD", macdResults.get(0).result.rendement);
        if (!meanRevResults.isEmpty()) performances.put("Mean Reversion", meanRevResults.get(0).result.rendement);

        performances.entrySet().stream()
            .sorted(java.util.Map.Entry.<String, Double>comparingByValue().reversed())
            .forEach(entry -> System.out.println(entry.getKey() + ": " + String.format("%.4f", entry.getValue() * 100) + "%"));

        System.out.println("\n=== EXPORT JSON TOUTES LES STRATEGIES ===");
        String json = StrategieBackTest.exportWalkForwardResultsToJson(improvedTrendResults);
        System.out.println("JSON Improved Trend Following exported: " + json.length() + " caractères");
    }

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


}
