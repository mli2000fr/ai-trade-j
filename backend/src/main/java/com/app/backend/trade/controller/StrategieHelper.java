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

    public void optimseParamForAllSymbol(){
        List<String> listeDbSymbols = this.getAllAssetSymbolsEligibleFromDb();
        int error = 0;
        for(String symbol : listeDbSymbols){
            try{
                StrategieBackTest.AllBestParams params = this.optimseParamByWalkForward(symbol);
                this.saveBestParams(symbol, params);
                Thread.sleep(200);
            }catch(Exception e){
                error++;
                TradeUtils.log("Erreur optimseParam("+symbol+") : " + e.getMessage());
            }
        }
        TradeUtils.log("optimseParamForAllSymbol: total "+listeDbSymbols.size()+", error" + error);

    }

    public void test_analyse_ByWalkForward(String symbol){
        optimseParamByWalkForward(symbol);
    }

    public StrategieBackTest.AllBestParams optimseParamByWalkForward(String symbol) {
        BarSeries series = this.mapping(this.getDailyValuesFromDb(symbol));

        System.out.println("=== DIAGNOSTIC DE LA SÉRIE ===");
        System.out.println("Nombre de bougies: " + series.getBarCount());
        System.out.println("Date début: " + series.getFirstBar().getEndTime());
        System.out.println("Date fin: " + series.getLastBar().getEndTime());
        System.out.println("Prix min: " + series.getBar(series.getBeginIndex()).getLowPrice());
        System.out.println("Prix max: " + series.getBar(series.getBeginIndex()).getHighPrice());


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
        System.out.println("Improved Trend Following: " + (improvedTrendResults.isEmpty() ? "Aucun résultat" : improvedTrendResults.get(0).result.tradeCount + " trades"));
        System.out.println("SMA Crossover: " + (smaCrossResults.isEmpty() ? "Aucun résultat" : smaCrossResults.get(0).result.tradeCount + " trades"));
        System.out.println("RSI: " + (rsiResults.isEmpty() ? "Aucun résultat" : rsiResults.get(0).result.tradeCount + " trades"));
        System.out.println("Breakout: " + (breakoutResults.isEmpty() ? "Aucun résultat" : breakoutResults.get(0).result.tradeCount + " trades"));
        System.out.println("MACD: " + (macdResults.isEmpty() ? "Aucun résultat" : macdResults.get(0).result.tradeCount + " trades"));
        System.out.println("Mean Reversion: " + (meanRevResults.isEmpty() ? "Aucun résultat" : meanRevResults.get(0).result.tradeCount + " trades"));

        System.out.println("\n=== CLASSEMENT PAR RENDEMENT ===");
        java.util.Map<String, Double> performances = new java.util.HashMap<>();
        java.util.Map<String, StrategieBackTest.RiskResult> detailedResults = new java.util.HashMap<>();

        if (!improvedTrendResults.isEmpty()) {
            performances.put("Improved Trend", improvedTrendResults.get(0).result.rendement);
            detailedResults.put("Improved Trend", improvedTrendResults.get(0).result);
        }
        if (!smaCrossResults.isEmpty()) {
            performances.put("SMA Crossover", smaCrossResults.get(0).result.rendement);
            detailedResults.put("SMA Crossover", smaCrossResults.get(0).result);
        }
        if (!rsiResults.isEmpty()) {
            performances.put("RSI", rsiResults.get(0).result.rendement);
            detailedResults.put("RSI", rsiResults.get(0).result);
        }
        if (!breakoutResults.isEmpty()) {
            performances.put("Breakout", breakoutResults.get(0).result.rendement);
            detailedResults.put("Breakout", breakoutResults.get(0).result);
        }
        if (!macdResults.isEmpty()) {
            performances.put("MACD", macdResults.get(0).result.rendement);
            detailedResults.put("MACD", macdResults.get(0).result);
        }
        if (!meanRevResults.isEmpty()) {
            performances.put("Mean Reversion", meanRevResults.get(0).result.rendement);
            detailedResults.put("Mean Reversion", meanRevResults.get(0).result);
        }

        performances.entrySet().stream()
            .sorted(java.util.Map.Entry.<String, Double>comparingByValue().reversed())
            .forEach(entry -> System.out.println(entry.getKey() + ": " + String.format("%.4f", entry.getValue() * 100) + "%"));

        // Extraction des meilleurs paramètres
        StrategieBackTest.ImprovedTrendFollowingParams bestImprovedTrend = !improvedTrendResults.isEmpty() ?
            (StrategieBackTest.ImprovedTrendFollowingParams) improvedTrendResults.get(0).params : null;
        StrategieBackTest.SmaCrossoverParams bestSmaCrossover = !smaCrossResults.isEmpty() ?
            (StrategieBackTest.SmaCrossoverParams) smaCrossResults.get(0).params : null;
        StrategieBackTest.RsiParams bestRsi = !rsiResults.isEmpty() ?
            (StrategieBackTest.RsiParams) rsiResults.get(0).params : null;
        StrategieBackTest.BreakoutParams bestBreakout = !breakoutResults.isEmpty() ?
            (StrategieBackTest.BreakoutParams) breakoutResults.get(0).params : null;
        StrategieBackTest.MacdParams bestMacd = !macdResults.isEmpty() ?
            (StrategieBackTest.MacdParams) macdResults.get(0).params : null;
        StrategieBackTest.MeanReversionParams bestMeanReversion = !meanRevResults.isEmpty() ?
            (StrategieBackTest.MeanReversionParams) meanRevResults.get(0).params : null;

        // Création de l'objet de retour avec tous les meilleurs paramètres
        StrategieBackTest.AllBestParams allBestParams = new StrategieBackTest.AllBestParams(
            bestImprovedTrend,
            bestSmaCrossover,
            bestRsi,
            bestBreakout,
            bestMacd,
            bestMeanReversion,
            performances,
            detailedResults
        );

        System.out.println("\n=== RÉSUMÉ DES MEILLEURS PARAMÈTRES ===");
        System.out.println("Meilleure stratégie: " + allBestParams.getBestStrategyName() +
                          " (Performance: " + String.format("%.4f", allBestParams.getBestPerformance() * 100) + "%)");

        if (bestImprovedTrend != null) {
            System.out.println("Improved Trend - Period: " + bestImprovedTrend.trendPeriod +
                              ", Short MA: " + bestImprovedTrend.shortMaPeriod +
                              ", Long MA: " + bestImprovedTrend.longMaPeriod +
                              ", Threshold: " + String.format("%.3f", bestImprovedTrend.breakoutThreshold) +
                              ", RSI Filter: " + bestImprovedTrend.useRsiFilter);
        }
        if (bestSmaCrossover != null) {
            System.out.println("SMA Crossover - Short: " + bestSmaCrossover.shortPeriod +
                              ", Long: " + bestSmaCrossover.longPeriod);
        }
        if (bestRsi != null) {
            System.out.println("RSI - Period: " + bestRsi.rsiPeriod +
                              ", Oversold: " + bestRsi.oversold +
                              ", Overbought: " + bestRsi.overbought);
        }
        if (bestBreakout != null) {
            System.out.println("Breakout - Lookback: " + bestBreakout.lookbackPeriod);
        }
        if (bestMacd != null) {
            System.out.println("MACD - Short: " + bestMacd.shortPeriod +
                              ", Long: " + bestMacd.longPeriod +
                              ", Signal: " + bestMacd.signalPeriod);
        }
        if (bestMeanReversion != null) {
            System.out.println("Mean Reversion - SMA: " + bestMeanReversion.smaPeriod +
                              ", Threshold: " + String.format("%.2f", bestMeanReversion.threshold));
        }

        System.out.println("\n=== EXPORT JSON TOUS LES BEST PARAMS ===");
        String jsonAllParams = allBestParams.toJson();
        System.out.println("JSON complet exporté: " + jsonAllParams.length() + " caractères");

        this.saveBestParams(symbol, allBestParams);

        return allBestParams;
    }

    public void test_analyse_RollingWindow(){
        optimseParamByRollingWindow("NVDA");
    }

    public StrategieBackTest.AllBestParams optimseParamByRollingWindow(String symbol) {
        BarSeries series = this.mapping(this.getDailyValuesFromDb(symbol));

        System.out.println("=== DIAGNOSTIC DE LA SÉRIE (ROLLING WINDOW) ===");
        System.out.println("Nombre de bougies: " + series.getBarCount());
        System.out.println("Date début: " + series.getFirstBar().getEndTime());
        System.out.println("Date fin: " + series.getLastBar().getEndTime());
        System.out.println("Prix min: " + series.getBar(series.getBeginIndex()).getLowPrice());
        System.out.println("Prix max: " + series.getBar(series.getBeginIndex()).getHighPrice());

        // Paramètres pour Rolling Window (stepSize = 50 pour glisser la fenêtre progressivement)
        int stepSize = 50;

        System.out.println("\n=== TEST IMPROVED TREND FOLLOWING (ROLLING WINDOW) ===");
        List<StrategieBackTest.RollingWindowResult> improvedTrendResults = strategieBackTest.runRollingWindowBacktestImprovedTrendFollowing(
            series, 300, 200, stepSize,
            10, 30,    // trendMin, trendMax
            5, 15,     // shortMaMin, shortMaMax
            15, 25,    // longMaMin, longMaMax
            0.001, 0.01, 0.002  // thresholdMin, thresholdMax, thresholdStep
        );
        StrategieBackTest.printRollingWindowResultsGeneric(improvedTrendResults);

        System.out.println("\n=== TEST SMA CROSSOVER (ROLLING WINDOW) ===");
        List<StrategieBackTest.RollingWindowResult> smaCrossResults = strategieBackTest.runRollingWindowBacktestSmaCrossover(series, 300, 200, stepSize, 5, 20, 10, 50);
        StrategieBackTest.printRollingWindowResultsSmaCrossover(smaCrossResults);

        System.out.println("\n=== TEST RSI (ROLLING WINDOW) ===");
        List<StrategieBackTest.RollingWindowResult> rsiResults = strategieBackTest.runRollingWindowBacktestRsi(series, 300, 200, stepSize, 10, 20, 20, 40, 5, 60, 80, 5);
        StrategieBackTest.printRollingWindowResultsRsi(rsiResults);

        System.out.println("\n=== TEST BREAKOUT (ROLLING WINDOW) ===");
        List<StrategieBackTest.RollingWindowResult> breakoutResults = strategieBackTest.runRollingWindowBacktestBreakout(series, 300, 200, stepSize, 5, 50);
        StrategieBackTest.printRollingWindowResultsGeneric(breakoutResults);

        System.out.println("\n=== TEST MACD (ROLLING WINDOW) ===");
        List<StrategieBackTest.RollingWindowResult> macdResults = strategieBackTest.runRollingWindowBacktestMacd(series, 300, 200, stepSize, 8, 16, 20, 30, 6, 12);
        StrategieBackTest.printRollingWindowResultsMacd(macdResults);

        System.out.println("\n=== TEST MEAN REVERSION (ROLLING WINDOW) ===");
        List<StrategieBackTest.RollingWindowResult> meanRevResults = strategieBackTest.runRollingWindowBacktestMeanReversion(series, 300, 200, stepSize, 10, 30, 1.0, 5.0, 0.5);
        StrategieBackTest.printRollingWindowResultsGeneric(meanRevResults);

        System.out.println("\n=== COMPARAISON DES PERFORMANCES (ROLLING WINDOW) ===");
        System.out.println("Improved Trend Following: " + (improvedTrendResults.isEmpty() ? "Aucun résultat" : improvedTrendResults.size() + " fenêtres testées"));
        System.out.println("SMA Crossover: " + (smaCrossResults.isEmpty() ? "Aucun résultat" : smaCrossResults.size() + " fenêtres testées"));
        System.out.println("RSI: " + (rsiResults.isEmpty() ? "Aucun résultat" : rsiResults.size() + " fenêtres testées"));
        System.out.println("Breakout: " + (breakoutResults.isEmpty() ? "Aucun résultat" : breakoutResults.size() + " fenêtres testées"));
        System.out.println("MACD: " + (macdResults.isEmpty() ? "Aucun résultat" : macdResults.size() + " fenêtres testées"));
        System.out.println("Mean Reversion: " + (meanRevResults.isEmpty() ? "Aucun résultat" : meanRevResults.size() + " fenêtres testées"));

        // Calculer la performance moyenne de chaque stratégie sur toutes les fenêtres
        System.out.println("\n=== CLASSEMENT PAR RENDEMENT MOYEN (ROLLING WINDOW) ===");
        java.util.Map<String, Double> performances = new java.util.HashMap<>();
        java.util.Map<String, StrategieBackTest.RiskResult> detailedResults = new java.util.HashMap<>();

        if (!improvedTrendResults.isEmpty()) {
            double avgReturn = improvedTrendResults.stream()
                .mapToDouble(r -> r.result.rendement)
                .average()
                .orElse(0.0);
            performances.put("Improved Trend", avgReturn);
            // Prendre le meilleur résultat pour les paramètres
            StrategieBackTest.RollingWindowResult bestResult = improvedTrendResults.stream()
                .max((r1, r2) -> Double.compare(r1.result.rendement, r2.result.rendement))
                .orElse(improvedTrendResults.get(0));
            detailedResults.put("Improved Trend", bestResult.result);
        }
        if (!smaCrossResults.isEmpty()) {
            double avgReturn = smaCrossResults.stream()
                .mapToDouble(r -> r.result.rendement)
                .average()
                .orElse(0.0);
            performances.put("SMA Crossover", avgReturn);
            StrategieBackTest.RollingWindowResult bestResult = smaCrossResults.stream()
                .max((r1, r2) -> Double.compare(r1.result.rendement, r2.result.rendement))
                .orElse(smaCrossResults.get(0));
            detailedResults.put("SMA Crossover", bestResult.result);
        }
        if (!rsiResults.isEmpty()) {
            double avgReturn = rsiResults.stream()
                .mapToDouble(r -> r.result.rendement)
                .average()
                .orElse(0.0);
            performances.put("RSI", avgReturn);
            StrategieBackTest.RollingWindowResult bestResult = rsiResults.stream()
                .max((r1, r2) -> Double.compare(r1.result.rendement, r2.result.rendement))
                .orElse(rsiResults.get(0));
            detailedResults.put("RSI", bestResult.result);
        }
        if (!breakoutResults.isEmpty()) {
            double avgReturn = breakoutResults.stream()
                .mapToDouble(r -> r.result.rendement)
                .average()
                .orElse(0.0);
            performances.put("Breakout", avgReturn);
            StrategieBackTest.RollingWindowResult bestResult = breakoutResults.stream()
                .max((r1, r2) -> Double.compare(r1.result.rendement, r2.result.rendement))
                .orElse(breakoutResults.get(0));
            detailedResults.put("Breakout", bestResult.result);
        }
        if (!macdResults.isEmpty()) {
            double avgReturn = macdResults.stream()
                .mapToDouble(r -> r.result.rendement)
                .average()
                .orElse(0.0);
            performances.put("MACD", avgReturn);
            StrategieBackTest.RollingWindowResult bestResult = macdResults.stream()
                .max((r1, r2) -> Double.compare(r1.result.rendement, r2.result.rendement))
                .orElse(macdResults.get(0));
            detailedResults.put("MACD", bestResult.result);
        }
        if (!meanRevResults.isEmpty()) {
            double avgReturn = meanRevResults.stream()
                .mapToDouble(r -> r.result.rendement)
                .average()
                .orElse(0.0);
            performances.put("Mean Reversion", avgReturn);
            StrategieBackTest.RollingWindowResult bestResult = meanRevResults.stream()
                .max((r1, r2) -> Double.compare(r1.result.rendement, r2.result.rendement))
                .orElse(meanRevResults.get(0));
            detailedResults.put("Mean Reversion", bestResult.result);
        }

        performances.entrySet().stream()
            .sorted(java.util.Map.Entry.<String, Double>comparingByValue().reversed())
            .forEach(entry -> System.out.println(entry.getKey() + " (moyenne): " + String.format("%.4f", entry.getValue() * 100) + "%"));

        // Extraction des meilleurs paramètres (ceux qui ont donné le meilleur résultat unique)
        StrategieBackTest.ImprovedTrendFollowingParams bestImprovedTrend = !improvedTrendResults.isEmpty() ?
            (StrategieBackTest.ImprovedTrendFollowingParams) improvedTrendResults.stream()
                .max((r1, r2) -> Double.compare(r1.result.rendement, r2.result.rendement))
                .get().params : null;
        StrategieBackTest.SmaCrossoverParams bestSmaCrossover = !smaCrossResults.isEmpty() ?
            (StrategieBackTest.SmaCrossoverParams) smaCrossResults.stream()
                .max((r1, r2) -> Double.compare(r1.result.rendement, r2.result.rendement))
                .get().params : null;
        StrategieBackTest.RsiParams bestRsi = !rsiResults.isEmpty() ?
            (StrategieBackTest.RsiParams) rsiResults.stream()
                .max((r1, r2) -> Double.compare(r1.result.rendement, r2.result.rendement))
                .get().params : null;
        StrategieBackTest.BreakoutParams bestBreakout = !breakoutResults.isEmpty() ?
            (StrategieBackTest.BreakoutParams) breakoutResults.stream()
                .max((r1, r2) -> Double.compare(r1.result.rendement, r2.result.rendement))
                .get().params : null;
        StrategieBackTest.MacdParams bestMacd = !macdResults.isEmpty() ?
            (StrategieBackTest.MacdParams) macdResults.stream()
                .max((r1, r2) -> Double.compare(r1.result.rendement, r2.result.rendement))
                .get().params : null;
        StrategieBackTest.MeanReversionParams bestMeanReversion = !meanRevResults.isEmpty() ?
            (StrategieBackTest.MeanReversionParams) meanRevResults.stream()
                .max((r1, r2) -> Double.compare(r1.result.rendement, r2.result.rendement))
                .get().params : null;

        // Création de l'objet de retour avec tous les meilleurs paramètres
        StrategieBackTest.AllBestParams allBestParams = new StrategieBackTest.AllBestParams(
            bestImprovedTrend,
            bestSmaCrossover,
            bestRsi,
            bestBreakout,
            bestMacd,
            bestMeanReversion,
            performances,
            detailedResults
        );

        System.out.println("\n=== RÉSUMÉ DES MEILLEURS PARAMÈTRES (ROLLING WINDOW) ===");
        System.out.println("Meilleure stratégie (moyenne): " + allBestParams.getBestStrategyName() +
                          " (Performance moyenne: " + String.format("%.4f", allBestParams.getBestPerformance() * 100) + "%)");

        if (bestImprovedTrend != null) {
            System.out.println("Improved Trend - Period: " + bestImprovedTrend.trendPeriod +
                              ", Short MA: " + bestImprovedTrend.shortMaPeriod +
                              ", Long MA: " + bestImprovedTrend.longMaPeriod +
                              ", Threshold: " + String.format("%.3f", bestImprovedTrend.breakoutThreshold) +
                              ", RSI Filter: " + bestImprovedTrend.useRsiFilter);
        }
        if (bestSmaCrossover != null) {
            System.out.println("SMA Crossover - Short: " + bestSmaCrossover.shortPeriod +
                              ", Long: " + bestSmaCrossover.longPeriod);
        }
        if (bestRsi != null) {
            System.out.println("RSI - Period: " + bestRsi.rsiPeriod +
                              ", Oversold: " + bestRsi.oversold +
                              ", Overbought: " + bestRsi.overbought);
        }
        if (bestBreakout != null) {
            System.out.println("Breakout - Lookback: " + bestBreakout.lookbackPeriod);
        }
        if (bestMacd != null) {
            System.out.println("MACD - Short: " + bestMacd.shortPeriod +
                              ", Long: " + bestMacd.longPeriod +
                              ", Signal: " + bestMacd.signalPeriod);
        }
        if (bestMeanReversion != null) {
            System.out.println("Mean Reversion - SMA: " + bestMeanReversion.smaPeriod +
                              ", Threshold: " + String.format("%.2f", bestMeanReversion.threshold));
        }

        System.out.println("\n=== EXPORT JSON TOUS LES BEST PARAMS (ROLLING WINDOW) ===");
        String jsonAllParams = allBestParams.toJson();
        System.out.println("JSON complet exporté: " + jsonAllParams.length() + " caractères");

        this.saveBestParams(symbol + "_rolling", allBestParams);

        return allBestParams;
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


    /**
     * Sauvegarde les meilleurs paramètres pour un symbole en base de données
     * @param symbol le symbole de l'action
     * @param bestParams les meilleurs paramètres à sauvegarder
     */
    public void saveBestParams(String symbol, StrategieBackTest.AllBestParams bestParams) {
        String checkSql = "SELECT COUNT(*) FROM best_param WHERE symbol = ?";
        int count = jdbcTemplate.queryForObject(checkSql, Integer.class, symbol);

        if (count > 0) {
            // Mise à jour
            updateBestParams(symbol, bestParams);
        } else {
            // Insertion
            insertBestParams(symbol, bestParams);
        }
    }

    /**
     * Insère de nouveaux meilleurs paramètres
     */
    private void insertBestParams(String symbol, StrategieBackTest.AllBestParams bestParams) {
        String sql = """
            INSERT INTO best_param (
                symbol, created_date, performance_ranking,
                itf_trend_period, itf_short_ma_period, itf_long_ma_period, itf_breakout_threshold, 
                itf_use_rsi_filter, itf_rsi_period, itf_performance,
                sma_short_period, sma_long_period, sma_performance,
                rsi_period, rsi_oversold, rsi_overbought, rsi_performance,
                breakout_lookback_period, breakout_performance,
                macd_short_period, macd_long_period, macd_signal_period, macd_performance,
                mr_sma_period, mr_threshold, mr_performance,
                best_strategy_name, best_strategy_performance, detailed_results,
                initial_capital, risk_per_trade, stop_loss_pct, take_profit_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """;

        jdbcTemplate.update(sql,
            symbol,
            java.sql.Date.valueOf(java.time.LocalDate.now()),
            convertMapToJson(bestParams.performanceRanking),
            // Improved Trend Following
            bestParams.improvedTrendFollowing != null ? bestParams.improvedTrendFollowing.trendPeriod : null,
            bestParams.improvedTrendFollowing != null ? bestParams.improvedTrendFollowing.shortMaPeriod : null,
            bestParams.improvedTrendFollowing != null ? bestParams.improvedTrendFollowing.longMaPeriod : null,
            bestParams.improvedTrendFollowing != null ? bestParams.improvedTrendFollowing.breakoutThreshold : null,
            bestParams.improvedTrendFollowing != null ? bestParams.improvedTrendFollowing.useRsiFilter : null,
            bestParams.improvedTrendFollowing != null ? bestParams.improvedTrendFollowing.rsiPeriod : null,
            bestParams.improvedTrendFollowing != null ? bestParams.improvedTrendFollowing.performance : null,
            // SMA Crossover
            bestParams.smaCrossover != null ? bestParams.smaCrossover.shortPeriod : null,
            bestParams.smaCrossover != null ? bestParams.smaCrossover.longPeriod : null,
            bestParams.smaCrossover != null ? bestParams.smaCrossover.performance : null,
            // RSI
            bestParams.rsi != null ? bestParams.rsi.rsiPeriod : null,
            bestParams.rsi != null ? bestParams.rsi.oversold : null,
            bestParams.rsi != null ? bestParams.rsi.overbought : null,
            bestParams.rsi != null ? bestParams.rsi.performance : null,
            // Breakout
            bestParams.breakout != null ? bestParams.breakout.lookbackPeriod : null,
            bestParams.breakout != null ? bestParams.breakout.performance : null,
            // MACD
            bestParams.macd != null ? bestParams.macd.shortPeriod : null,
            bestParams.macd != null ? bestParams.macd.longPeriod : null,
            bestParams.macd != null ? bestParams.macd.signalPeriod : null,
            bestParams.macd != null ? bestParams.macd.performance : null,
            // Mean Reversion
            bestParams.meanReversion != null ? bestParams.meanReversion.smaPeriod : null,
            bestParams.meanReversion != null ? bestParams.meanReversion.threshold : null,
            bestParams.meanReversion != null ? bestParams.meanReversion.performance : null,
            // Best strategy
            bestParams.getBestStrategyName(),
            bestParams.getBestPerformance(),
            convertDetailedResultsToJson(bestParams.detailedResults),
            // Paramètres de référence du backtest
            StrategieBackTest.INITIAL_CAPITAL,  // INITIAL_CAPITAL
            StrategieBackTest.RISK_PER_TRADE,     // RISK_PER_TRADE
            StrategieBackTest.STOP_LOSS_PCT_STOP,     // STOP_LOSS_PCT_STOP
            StrategieBackTest.TAKE_PROFIL_PCT      // TAKE_PROFIT_PCT
        );

        logger.info("Nouveaux meilleurs paramètres insérés pour le symbole: {}", symbol);
    }

    /**
     * Met à jour les meilleurs paramètres existants
     */
    private void updateBestParams(String symbol, StrategieBackTest.AllBestParams bestParams) {
        String sql = """
            UPDATE best_param SET
                updated_date = CURRENT_TIMESTAMP, performance_ranking = ?,
                itf_trend_period = ?, itf_short_ma_period = ?, itf_long_ma_period = ?, itf_breakout_threshold = ?, 
                itf_use_rsi_filter = ?, itf_rsi_period = ?, itf_performance = ?,
                sma_short_period = ?, sma_long_period = ?, sma_performance = ?,
                rsi_period = ?, rsi_oversold = ?, rsi_overbought = ?, rsi_performance = ?,
                breakout_lookback_period = ?, breakout_performance = ?,
                macd_short_period = ?, macd_long_period = ?, macd_signal_period = ?, macd_performance = ?,
                mr_sma_period = ?, mr_threshold = ?, mr_performance = ?,
                best_strategy_name = ?, best_strategy_performance = ?, detailed_results = ?,
                initial_capital = ?, risk_per_trade = ?, stop_loss_pct = ?, take_profit_pct = ?
            WHERE symbol = ?
            """;

        jdbcTemplate.update(sql,
            convertMapToJson(bestParams.performanceRanking),
            // Improved Trend Following
            bestParams.improvedTrendFollowing != null ? bestParams.improvedTrendFollowing.trendPeriod : null,
            bestParams.improvedTrendFollowing != null ? bestParams.improvedTrendFollowing.shortMaPeriod : null,
            bestParams.improvedTrendFollowing != null ? bestParams.improvedTrendFollowing.longMaPeriod : null,
            bestParams.improvedTrendFollowing != null ? bestParams.improvedTrendFollowing.breakoutThreshold : null,
            bestParams.improvedTrendFollowing != null ? bestParams.improvedTrendFollowing.useRsiFilter : null,
            bestParams.improvedTrendFollowing != null ? bestParams.improvedTrendFollowing.rsiPeriod : null,
            bestParams.improvedTrendFollowing != null ? bestParams.improvedTrendFollowing.performance : null,
            // SMA Crossover
            bestParams.smaCrossover != null ? bestParams.smaCrossover.shortPeriod : null,
            bestParams.smaCrossover != null ? bestParams.smaCrossover.longPeriod : null,
            bestParams.smaCrossover != null ? bestParams.smaCrossover.performance : null,
            // RSI
            bestParams.rsi != null ? bestParams.rsi.rsiPeriod : null,
            bestParams.rsi != null ? bestParams.rsi.oversold : null,
            bestParams.rsi != null ? bestParams.rsi.overbought : null,
            bestParams.rsi != null ? bestParams.rsi.performance : null,
            // Breakout
            bestParams.breakout != null ? bestParams.breakout.lookbackPeriod : null,
            bestParams.breakout != null ? bestParams.breakout.performance : null,
            // MACD
            bestParams.macd != null ? bestParams.macd.shortPeriod : null,
            bestParams.macd != null ? bestParams.macd.longPeriod : null,
            bestParams.macd != null ? bestParams.macd.signalPeriod : null,
            bestParams.macd != null ? bestParams.macd.performance : null,
            // Mean Reversion
            bestParams.meanReversion != null ? bestParams.meanReversion.smaPeriod : null,
            bestParams.meanReversion != null ? bestParams.meanReversion.threshold : null,
            bestParams.meanReversion != null ? bestParams.meanReversion.performance : null,
            // Best strategy
            bestParams.getBestStrategyName(),
            bestParams.getBestPerformance(),
            convertDetailedResultsToJson(bestParams.detailedResults),
            // Paramètres de référence du backtest
            StrategieBackTest.INITIAL_CAPITAL,  // INITIAL_CAPITAL
            StrategieBackTest.RISK_PER_TRADE,     // RISK_PER_TRADE
            StrategieBackTest.STOP_LOSS_PCT_STOP,     // STOP_LOSS_PCT_STOP
            StrategieBackTest.TAKE_PROFIL_PCT, // TAKE_PROFIT_PCT
            symbol
        );

        logger.info("Meilleurs paramètres mis à jour pour le symbole: {}", symbol);
    }

    /**
     * Récupère les meilleurs paramètres pour un symbole depuis la base de données
     * @param symbol le symbole de l'action
     * @return AllBestParams reconstruit ou null si non trouvé
     */
    public StrategieBackTest.AllBestParams getBestParams(String symbol) {
        String sql = """
            SELECT * FROM best_param WHERE symbol = ?
            """;

        try {
            return jdbcTemplate.queryForObject(sql, (rs, rowNum) -> {
                // Plus de reconstruction de TrendFollowingParams car les colonnes n'existent plus
                StrategieBackTest.TrendFollowingParams tfParams = null;

                StrategieBackTest.ImprovedTrendFollowingParams itfParams = null;
                if (rs.getObject("itf_trend_period") != null) {
                    itfParams = new StrategieBackTest.ImprovedTrendFollowingParams(
                        rs.getInt("itf_trend_period"),
                        rs.getInt("itf_short_ma_period"),
                        rs.getInt("itf_long_ma_period"),
                        rs.getDouble("itf_breakout_threshold"),
                        rs.getBoolean("itf_use_rsi_filter"),
                        rs.getInt("itf_rsi_period"),
                        rs.getDouble("itf_performance")
                    );
                }

                StrategieBackTest.SmaCrossoverParams smaParams = null;
                if (rs.getObject("sma_short_period") != null) {
                    smaParams = new StrategieBackTest.SmaCrossoverParams(
                        rs.getInt("sma_short_period"),
                        rs.getInt("sma_long_period"),
                        rs.getDouble("sma_performance")
                    );
                }

                StrategieBackTest.RsiParams rsiParams = null;
                if (rs.getObject("rsi_period") != null) {
                    rsiParams = new StrategieBackTest.RsiParams(
                        rs.getInt("rsi_period"),
                        rs.getDouble("rsi_oversold"),
                        rs.getDouble("rsi_overbought"),
                        rs.getDouble("rsi_performance")
                    );
                }

                StrategieBackTest.BreakoutParams breakoutParams = null;
                if (rs.getObject("breakout_lookback_period") != null) {
                    breakoutParams = new StrategieBackTest.BreakoutParams(
                        rs.getInt("breakout_lookback_period"),
                        rs.getDouble("breakout_performance")
                    );
                }

                StrategieBackTest.MacdParams macdParams = null;
                if (rs.getObject("macd_short_period") != null) {
                    macdParams = new StrategieBackTest.MacdParams(
                        rs.getInt("macd_short_period"),
                        rs.getInt("macd_long_period"),
                        rs.getInt("macd_signal_period"),
                        rs.getDouble("macd_performance")
                    );
                }

                StrategieBackTest.MeanReversionParams mrParams = null;
                if (rs.getObject("mr_sma_period") != null) {
                    mrParams = new StrategieBackTest.MeanReversionParams(
                        rs.getInt("mr_sma_period"),
                        rs.getDouble("mr_threshold"),
                        rs.getDouble("mr_performance")
                    );
                }

                // Reconstruction des maps depuis JSON
                java.util.Map<String, Double> performanceRanking = convertJsonToPerformanceMap(rs.getString("performance_ranking"));
                java.util.Map<String, StrategieBackTest.RiskResult> detailedResults = convertJsonToDetailedResults(rs.getString("detailed_results"));

                return new StrategieBackTest.AllBestParams(
                    itfParams, smaParams, rsiParams, breakoutParams, macdParams, mrParams,
                    performanceRanking, detailedResults
                );
            }, symbol);
        } catch (org.springframework.dao.EmptyResultDataAccessException e) {
            logger.warn("Aucun meilleur paramètre trouvé pour le symbole: {}", symbol);
            return null;
        }
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

    private String convertDetailedResultsToJson(java.util.Map<String, StrategieBackTest.RiskResult> detailedResults) {
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

    private java.util.Map<String, StrategieBackTest.RiskResult> convertJsonToDetailedResults(String json) {
        if (json == null) return new java.util.HashMap<>();
        try {
            com.google.gson.JsonObject jsonObj = new com.google.gson.JsonParser().parse(json).getAsJsonObject();
            java.util.Map<String, StrategieBackTest.RiskResult> map = new java.util.HashMap<>();

            jsonObj.entrySet().forEach(entry -> {
                com.google.gson.JsonObject resultObj = entry.getValue().getAsJsonObject();
                StrategieBackTest.RiskResult riskResult = new StrategieBackTest.RiskResult(
                    resultObj.get("rendement").getAsDouble(),
                    resultObj.get("maxDrawdown").getAsDouble(),
                    resultObj.get("tradeCount").getAsInt(),
                    resultObj.get("winRate").getAsDouble(),
                    resultObj.get("avgPnL").getAsDouble(),
                    resultObj.get("profitFactor").getAsDouble(),
                    resultObj.get("avgTradeBars").getAsDouble(),
                    resultObj.get("maxTradeGain").getAsDouble(),
                    resultObj.get("maxTradeLoss").getAsDouble()
                );
                map.put(entry.getKey(), riskResult);
            });
            return map;
        } catch (Exception e) {
            logger.warn("Erreur conversion JSON vers Map<String, RiskResult>: {}", e.getMessage());
            return new java.util.HashMap<>();
        }
    }

    /**
     * Récupère tous les symboles ayant des meilleurs paramètres sauvegardés
     * @return Liste des symboles
     */
    public List<String> getAllSymbolsWithBestParams() {
        String sql = "SELECT symbol FROM best_param ORDER BY updated_date DESC";
        return jdbcTemplate.queryForList(sql, String.class);
    }

    /**
     * Supprime les meilleurs paramètres pour un symbole
     * @param symbol le symbole à supprimer
     */
    public void deleteBestParams(String symbol) {
        String sql = "DELETE FROM best_param WHERE symbol = ?";
        int rowsAffected = jdbcTemplate.update(sql, symbol);
        if (rowsAffected > 0) {
            logger.info("Meilleurs paramètres supprimés pour le symbole: {}", symbol);
        }
    }

    public static class BestInOutStrategy {
        public final String entryName;
        public final Object entryParams;
        public final String exitName;
        public final Object exitParams;
        public final StrategieBackTest.RiskResult result;
        public BestInOutStrategy(String entryName, Object entryParams, String exitName, Object exitParams, StrategieBackTest.RiskResult result) {
            this.entryName = entryName;
            this.entryParams = entryParams;
            this.exitName = exitName;
            this.exitParams = exitParams;
            this.result = result;
        }
    }

    public BestInOutStrategy optimseBestInOutByWalkForward(String symbol) {
        BarSeries series = this.mapping(this.getDailyValuesFromDb(symbol));
        // Générer les meilleurs paramètres pour chaque stratégie
        StrategieBackTest.ImprovedTrendFollowingParams bestImprovedTrend = strategieBackTest.optimiseImprovedTrendFollowingParameters(series, 10, 30, 5, 15, 15, 25, 0.001, 0.01, 0.002);
        StrategieBackTest.SmaCrossoverParams bestSmaCrossover = strategieBackTest.optimiseSmaCrossoverParameters(series, 5, 20, 10, 50);
        StrategieBackTest.RsiParams bestRsi = strategieBackTest.optimiseRsiParameters(series, 10, 20, 20, 40, 5, 60, 80, 5);
        StrategieBackTest.BreakoutParams bestBreakout = strategieBackTest.optimiseBreakoutParameters(series, 5, 50);
        StrategieBackTest.MacdParams bestMacd = strategieBackTest.optimiseMacdParameters(series, 8, 16, 20, 30, 6, 12);
        StrategieBackTest.MeanReversionParams bestMeanReversion = strategieBackTest.optimiseMeanReversionParameters(series, 10, 30, 1.0, 5.0, 0.5);

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
                StrategieBackTest.RiskResult result = strategieBackTest.backtestStrategyRisk(combined, series);
                System.out.println("IN: " + entryName + " | OUT: " + exitName + " | Rendement: " + String.format("%.4f", result.rendement * 100) + "% | Trades: " + result.tradeCount + " | WinRate: " + String.format("%.2f", result.winRate * 100) + "% | Drawdown: " + String.format("%.2f", result.maxDrawdown * 100) + "%");
                if (result.rendement > bestPerf) {
                    bestPerf = result.rendement;
                    bestCombo = new BestInOutStrategy(entryName, entryParams, exitName, exitParams, result);
                }
            }
        }
        System.out.println("=== MEILLEUR COUPLE IN/OUT ===");
        if (bestCombo != null) {
            System.out.println("IN: " + bestCombo.entryName + " | OUT: " + bestCombo.exitName + " | Rendement: " + String.format("%.4f", bestCombo.result.rendement * 100) + "% | Trades: " + bestCombo.result.tradeCount);
        }
        return bestCombo;
    }

    /**
     * Teste automatiquement toutes les combinaisons croisées de stratégies pour in (entrée) et out (sortie).
     * Affiche les résultats pour chaque couple et retourne le meilleur couple.
     */
    public BestInOutStrategy testAllCrossedStrategies(String symbol) {
        BarSeries series = this.mapping(this.getDailyValuesFromDb(symbol));
        // Optimisation des paramètres pour chaque stratégie
        StrategieBackTest.ImprovedTrendFollowingParams bestImprovedTrend = strategieBackTest.optimiseImprovedTrendFollowingParameters(series, 10, 30, 5, 15, 15, 25, 0.001, 0.01, 0.002);
        StrategieBackTest.SmaCrossoverParams bestSmaCrossover = strategieBackTest.optimiseSmaCrossoverParameters(series, 5, 20, 10, 50);
        StrategieBackTest.RsiParams bestRsi = strategieBackTest.optimiseRsiParameters(series, 10, 20, 20, 40, 5, 60, 80, 5);
        StrategieBackTest.BreakoutParams bestBreakout = strategieBackTest.optimiseBreakoutParameters(series, 5, 50);
        StrategieBackTest.MacdParams bestMacd = strategieBackTest.optimiseMacdParameters(series, 8, 16, 20, 30, 6, 12);
        StrategieBackTest.MeanReversionParams bestMeanReversion = strategieBackTest.optimiseMeanReversionParameters(series, 10, 30, 1.0, 5.0, 0.5);

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
                StrategieBackTest.RiskResult result = strategieBackTest.backtestStrategyRisk(combined, series);
                System.out.println("IN: " + entryName + " | OUT: " + exitName + " | Rendement: " + String.format("%.4f", result.rendement * 100) + "% | Trades: " + result.tradeCount + " | WinRate: " + String.format("%.2f", result.winRate * 100) + "% | Drawdown: " + String.format("%.2f", result.maxDrawdown * 100) + "%");
                if (result.rendement > bestPerf) {
                    bestPerf = result.rendement;
                    bestCombo = new BestInOutStrategy(entryName, entryParams, exitName, exitParams, result);
                }
            }
        }
        System.out.println("=== MEILLEUR COUPLE IN/OUT ===");
        if (bestCombo != null) {
            System.out.println("IN: " + bestCombo.entryName + " | OUT: " + bestCombo.exitName + " | Rendement: " + String.format("%.4f", bestCombo.result.rendement * 100) + "% | Trades: " + bestCombo.result.tradeCount);
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
}
