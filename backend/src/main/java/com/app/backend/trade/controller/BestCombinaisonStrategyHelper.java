package com.app.backend.trade.controller;

import com.app.backend.model.RiskResult;
import com.app.backend.trade.model.BestCombinationResult;
import com.app.backend.trade.model.SignalType;
import com.app.backend.trade.strategy.*;
import com.app.backend.trade.util.TradeConstant;
import com.app.backend.trade.util.TradeUtils;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Controller;
import org.ta4j.core.BarSeries;
import org.ta4j.core.Rule;
import java.util.*;

@Controller
public class BestCombinaisonStrategyHelper {


    private StrategieHelper strategieHelper;

    // Constantes pour la gestion des bougies et des pourcentages
    private static final int NB_IN = 2;
    private static final double NB_OUT = 2;
    private static final boolean INSERT_ONLY = true;

    @Autowired
    private JdbcTemplate jdbcTemplate;
    private final Gson gson = new Gson();

    @Autowired
    public BestCombinaisonStrategyHelper(StrategieHelper strategieHelper) {
        this.strategieHelper = strategieHelper;
    }

    public BestCombinationResult findBestCombination(String symbol, int in, int out) {
        // Récupérer les BarSeries depuis la base via StrategieHelper
        BarSeries series = strategieHelper.getAndUpdateDBDailyValu(symbol, TradeConstant.NOMBRE_TOTAL_BOUGIES);
        List<BarSeries> seriesList = new ArrayList<>();
        if (series != null) {
            seriesList.add(series);
        }
        return findBestCombination(symbol, seriesList, in, out);
    }

    /**
     * Cherche la meilleure combinaison de stratégies pour in et out.
     * @param symbol le symbole à trader
     * @param seriesList la liste des BarSeries
     * @param in le nombre de stratégies pour in
     * @param out le nombre de stratégies pour out
     * @return un objet contenant la meilleure combinaison et ses paramètres
     */
    public BestCombinationResult findBestCombination(String symbol, List<BarSeries> seriesList, int in, int out) {
        List<Class<? extends TradeStrategy>> strategies = Arrays.asList(
            ImprovedTrendFollowingStrategy.class,
            SmaCrossoverStrategy.class,
            RsiStrategy.class,
            BreakoutStrategy.class,
            MacdStrategy.class,
            MeanReversionStrategy.class
        );
        List<List<Class<? extends TradeStrategy>>> inCombinations = generateCombinations(strategies, in);
        List<List<Class<? extends TradeStrategy>>> outCombinations = generateCombinations(strategies, out);
        BestCombinationResult bestResult = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (List<Class<? extends TradeStrategy>> inCombo : inCombinations) {
            for (List<Class<? extends TradeStrategy>> outCombo : outCombinations) {
                BestCombinationResult result = evaluateCombination(symbol, seriesList, inCombo, outCombo);
                if (result.score > bestScore) {
                    bestScore = result.score;
                    bestResult = result;
                }
            }
        }
        return bestResult;
    }

    /**
     * Boucle sur toutes les valeurs possibles de in et out (de 1 à 6),
     * compare les scores et retourne le meilleur résultat global.
     */
    public BestCombinationResult findBestCombinationGlobal(String symbol) {
        Map<String, String> bestInOutStrategy = this.getInOutStrategiesForSymbol(symbol);
        BestCombinationResult bestGlobal = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        List<Class<? extends TradeStrategy>> strategies = Arrays.asList(
            ImprovedTrendFollowingStrategy.class,
            SmaCrossoverStrategy.class,
            RsiStrategy.class,
            BreakoutStrategy.class,
            MacdStrategy.class,
            MeanReversionStrategy.class
        );
        // Conversion nom -> classe
        Class<? extends TradeStrategy> inClass = null;
        Class<? extends TradeStrategy> outClass = null;
        if (bestInOutStrategy != null) {
            for (Class<? extends TradeStrategy> clazz : strategies) {
                if (clazz.getSimpleName().equals(TradeUtils.parseStrategyName(bestInOutStrategy.get("in")))) {
                    inClass = clazz;
                }
                if (clazz.getSimpleName().equals(TradeUtils.parseStrategyName(bestInOutStrategy.get("out")))) {
                    outClass = clazz;
                }
            }
        }
        BarSeries barSeries = strategieHelper.getAndUpdateDBDailyValu(symbol, TradeConstant.NOMBRE_TOTAL_BOUGIES_OPTIM);
        for (int in = 1; in <= NB_IN; in++) {
            for (int out = 1; out <= NB_OUT; out++) {
                // Générer les combinaisons qui incluent obligatoirement la best in/out
                List<List<Class<? extends TradeStrategy>>> inCombinations = (inClass != null) ? generateCombinationsWithMandatory(strategies, in, inClass) : generateCombinations(strategies, in);
                List<List<Class<? extends TradeStrategy>>> outCombinations = (outClass != null) ? generateCombinationsWithMandatory(strategies, out, outClass) : generateCombinations(strategies, out);
                for (List<Class<? extends TradeStrategy>> inCombo : inCombinations) {
                    for (List<Class<? extends TradeStrategy>> outCombo : outCombinations) {
                        BestCombinationResult result = evaluateCombination(symbol, Arrays.asList(barSeries), inCombo, outCombo);
                        // TradeUtils.log("Global search: inCombo=" + inCombo + ", outCombo=" + outCombo + " => score=" + result.score);
                        if (result != null && result.score > bestScore) {
                            bestScore = result.score;
                            bestGlobal = result;
                        }
                    }
                }
            }
        }
        TradeUtils.log("Best global combination for symbol=" + symbol + " : " + resultObjToString(bestGlobal));
        return bestGlobal;
    }

    // Génère toutes les combinaisons de n éléments parmi la liste
    private List<List<Class<? extends TradeStrategy>>> generateCombinations(List<Class<? extends TradeStrategy>> strategies, int n) {
        List<List<Class<? extends TradeStrategy>>> result = new ArrayList<>();
        generateCombinationsHelper(strategies, n, 0, new ArrayList<>(), result);
        return result;
    }
    private void generateCombinationsHelper(List<Class<? extends TradeStrategy>> strategies, int n, int start, List<Class<? extends TradeStrategy>> current, List<List<Class<? extends TradeStrategy>>> result) {
        if (current.size() == n) {
            result.add(new ArrayList<>(current));
            return;
        }
        for (int i = start; i < strategies.size(); i++) {
            current.add(strategies.get(i));
            generateCombinationsHelper(strategies, n, i + 1, current, result);
            current.remove(current.size() - 1);
        }
    }

    // Génère toutes les combinaisons de n éléments parmi la liste, incluant obligatoirement une stratégie spécifique
    private List<List<Class<? extends TradeStrategy>>> generateCombinationsWithMandatory(List<Class<? extends TradeStrategy>> strategies, int n, Class<? extends TradeStrategy> mandatory) {
        List<List<Class<? extends TradeStrategy>>> all = generateCombinations(strategies, n);
        List<List<Class<? extends TradeStrategy>>> filtered = new ArrayList<>();
        for (List<Class<? extends TradeStrategy>> combo : all) {
            if (combo.contains(mandatory)) {
                filtered.add(combo);
            }
        }
        return filtered;
    }

    // Évalue la combinaison selon la logique métier de StrategieHelper
    private BestCombinationResult evaluateCombination(String symbol, List<BarSeries> seriesList, List<Class<? extends TradeStrategy>> inCombo, List<Class<? extends TradeStrategy>> outCombo) {
        BestCombinationResult resultObj = new BestCombinationResult();
        if (seriesList == null || seriesList.isEmpty()) {
            resultObj.score = Double.NEGATIVE_INFINITY;
            return resultObj;
        }
        BarSeries fullSeries = seriesList.get(0);
        int totalCount = fullSeries.getBarCount();
        int optimCount = (int) Math.round(totalCount * TradeConstant.PC_OPTIM);
        int testCount = totalCount - optimCount;
        // Séparer les bougies pour optimisation et test
        BarSeries[] split = TradeUtils.splitSeriesForWalkForward(fullSeries);
        BarSeries optimSeries = split[0];
        BarSeries testSeries = split[1];
        StrategieBackTest backTest = new StrategieBackTest();
        List<TradeStrategy> inStrategies = new ArrayList<>();
        List<String> inStrategyNames = new ArrayList<>();
        for (Class<? extends TradeStrategy> clazz : inCombo) {
            inStrategyNames.add(clazz.getSimpleName());
            if (clazz.equals(ImprovedTrendFollowingStrategy.class)) {
                StrategieBackTest.ImprovedTrendFollowingParams params = backTest.optimiseImprovedTrendFollowingParameters(optimSeries, 10, 30, 5, 15, 15, 25, 0.001, 0.01, 0.002);
                inStrategies.add(new ImprovedTrendFollowingStrategy(params.trendPeriod, params.shortMaPeriod, params.longMaPeriod, params.breakoutThreshold, params.useRsiFilter, params.rsiPeriod));
                resultObj.inParams.put("ImprovedTrendFollowing", params);
            } else if (clazz.equals(SmaCrossoverStrategy.class)) {
                StrategieBackTest.SmaCrossoverParams params = backTest.optimiseSmaCrossoverParameters(optimSeries, 5, 20, 10, 50);
                inStrategies.add(new SmaCrossoverStrategy(params.shortPeriod, params.longPeriod));
                resultObj.inParams.put("SmaCrossover", params);
            } else if (clazz.equals(RsiStrategy.class)) {
                StrategieBackTest.RsiParams params = backTest.optimiseRsiParameters(optimSeries, 10, 20, 20, 40, 5, 60, 80, 5);
                inStrategies.add(new RsiStrategy(params.rsiPeriod, params.oversold, params.overbought));
                resultObj.inParams.put("Rsi", params);
            } else if (clazz.equals(BreakoutStrategy.class)) {
                StrategieBackTest.BreakoutParams params = backTest.optimiseBreakoutParameters(optimSeries, 5, 50);
                inStrategies.add(new BreakoutStrategy(params.lookbackPeriod));
                resultObj.inParams.put("Breakout", params);
            } else if (clazz.equals(MacdStrategy.class)) {
                StrategieBackTest.MacdParams params = backTest.optimiseMacdParameters(optimSeries, 8, 16, 20, 30, 6, 12);
                inStrategies.add(new MacdStrategy(params.shortPeriod, params.longPeriod, params.signalPeriod));
                resultObj.inParams.put("Macd", params);
            } else if (clazz.equals(MeanReversionStrategy.class)) {
                StrategieBackTest.MeanReversionParams params = backTest.optimiseMeanReversionParameters(optimSeries, 10, 30, 1.0, 5.0, 0.5);
                inStrategies.add(new MeanReversionStrategy(params.smaPeriod, params.threshold));
                resultObj.inParams.put("MeanReversion", params);
            }
        }
        List<TradeStrategy> outStrategies = new ArrayList<>();
        List<String> outStrategyNames = new ArrayList<>();
        for (Class<? extends TradeStrategy> clazz : outCombo) {
            outStrategyNames.add(clazz.getSimpleName());
            if (clazz.equals(ImprovedTrendFollowingStrategy.class)) {
                StrategieBackTest.ImprovedTrendFollowingParams params = backTest.optimiseImprovedTrendFollowingParameters(optimSeries, 10, 30, 5, 15, 15, 25, 0.001, 0.01, 0.002);
                outStrategies.add(new ImprovedTrendFollowingStrategy(params.trendPeriod, params.shortMaPeriod, params.longMaPeriod, params.breakoutThreshold, params.useRsiFilter, params.rsiPeriod));
                resultObj.outParams.put("ImprovedTrendFollowing", params);
            } else if (clazz.equals(SmaCrossoverStrategy.class)) {
                StrategieBackTest.SmaCrossoverParams params = backTest.optimiseSmaCrossoverParameters(optimSeries, 5, 20, 10, 50);
                outStrategies.add(new SmaCrossoverStrategy(params.shortPeriod, params.longPeriod));
                resultObj.outParams.put("SmaCrossover", params);
            } else if (clazz.equals(RsiStrategy.class)) {
                StrategieBackTest.RsiParams params = backTest.optimiseRsiParameters(optimSeries, 10, 20, 20, 40, 5, 60, 80, 5);
                outStrategies.add(new RsiStrategy(params.rsiPeriod, params.oversold, params.overbought));
                resultObj.outParams.put("Rsi", params);
            } else if (clazz.equals(BreakoutStrategy.class)) {
                StrategieBackTest.BreakoutParams params = backTest.optimiseBreakoutParameters(optimSeries, 5, 50);
                outStrategies.add(new BreakoutStrategy(params.lookbackPeriod));
                resultObj.outParams.put("Breakout", params);
            } else if (clazz.equals(MacdStrategy.class)) {
                StrategieBackTest.MacdParams params = backTest.optimiseMacdParameters(optimSeries, 8, 16, 20, 30, 6, 12);
                outStrategies.add(new MacdStrategy(params.shortPeriod, params.longPeriod, params.signalPeriod));
                resultObj.outParams.put("Macd", params);
            } else if (clazz.equals(MeanReversionStrategy.class)) {
                StrategieBackTest.MeanReversionParams params = backTest.optimiseMeanReversionParameters(optimSeries, 10, 30, 1.0, 5.0, 0.5);
                outStrategies.add(new MeanReversionStrategy(params.smaPeriod, params.threshold));
                resultObj.outParams.put("MeanReversion", params);
            }
        }
        final Rule finalEntryRule;
        final Rule finalExitRule;
        {
            Rule tempEntryRule = null;
            Rule tempExitRule = null;
            for (TradeStrategy strat : inStrategies) {
                if (tempEntryRule == null) tempEntryRule = strat.getEntryRule(testSeries);
                else tempEntryRule = tempEntryRule.or(strat.getEntryRule(testSeries));
            }
            for (TradeStrategy strat : outStrategies) {
                if (tempExitRule == null) tempExitRule = strat.getExitRule(testSeries);
                else tempExitRule = tempExitRule.or(strat.getExitRule(testSeries));
            }
            finalEntryRule = tempEntryRule;
            finalExitRule = tempExitRule;
        }
        TradeStrategy combinedStrategy = new TradeStrategy() {
            @Override
            public Rule getEntryRule(BarSeries s) { return finalEntryRule; }
            @Override
            public Rule getExitRule(BarSeries s) { return finalExitRule; }
            @Override
            public String getName() { return "CombinedStrategy"; }
        };
        RiskResult backtestResult = backTest.backtestStrategyRisk(combinedStrategy, testSeries);
        resultObj.score = backtestResult.rendement;
        resultObj.backtestResult = backtestResult;
        resultObj.inStrategyNames = inStrategyNames;
        resultObj.outStrategyNames = outStrategyNames;
        TradeUtils.log("BestCombinationResult : " + resultObjToString(resultObj));
        return resultObj;
    }

    // Méthode utilitaire pour afficher les détails du résultat
    private String resultObjToString(BestCombinationResult result) {
        StringBuilder sb = new StringBuilder();
        sb.append("inStrategyNames=").append(result.inStrategyNames).append(", ");
        sb.append("outStrategyNames=").append(result.outStrategyNames).append(", ");
        sb.append("score=").append(result.score).append(", ");
        sb.append("inParams=").append(result.inParams).append(", ");
        sb.append("outParams=").append(result.outParams).append(", ");
        if (result.backtestResult != null) {
            sb.append("backtestResult={");
            sb.append("rendement=").append(result.backtestResult.rendement).append(", ");
            sb.append("maxDrawdown=").append(result.backtestResult.maxDrawdown).append(", ");
            sb.append("tradeCount=").append(result.backtestResult.tradeCount).append(", ");
            sb.append("winRate=").append(result.backtestResult.winRate).append(", ");
            sb.append("avgPnL=").append(result.backtestResult.avgPnL).append(", ");
            sb.append("profitFactor=").append(result.backtestResult.profitFactor).append(", ");
            sb.append("avgTradeBars=").append(result.backtestResult.avgTradeBars).append(", ");
            sb.append("maxTradeGain=").append(result.backtestResult.maxTradeGain).append(", ");
            sb.append("maxTradeLoss=").append(result.backtestResult.maxTradeLoss);
            sb.append("}");
        }
        return sb.toString();
    }


    /**
     * Retourne les noms des stratégies in et out pour un symbole donné
     */
    public Map<String, String> getInOutStrategiesForSymbol(String symbol) {
        BestInOutStrategy best = strategieHelper.getBestInOutStrategy(symbol);
        Map<String, String> result = new HashMap<>();
        if (best != null) {
            result.put("in", best.getEntryName());
            result.put("out", best.getExitName());
        }
        return result;
    }

    /**
     * Sauvegarde un BestCombinationResult dans la table best_in_out_mix_strategy
     * Si le symbol existe déjà, fait un UPDATE, sinon fait un INSERT
     * Prend en compte initialCapital, riskPerTrade, stopLossPct, takeProfitPct
     */
    public void saveBestCombinationResult(String symbol, BestCombinationResult result) {
        String inStrategyNamesJson = gson.toJson(result.inStrategyNames);
        String outStrategyNamesJson = gson.toJson(result.outStrategyNames);
        String inParamsJson = gson.toJson(result.inParams);
        String outParamsJson = gson.toJson(result.outParams);
        // Vérifier si le symbol existe déjà
        String checkSql = "SELECT COUNT(*) FROM best_in_out_mix_strategy WHERE symbol = ?";
        int count = jdbcTemplate.queryForObject(checkSql, Integer.class, symbol);
        if (count > 0) {
            // UPDATE
            String updateSql = """
                    UPDATE best_in_out_mix_strategy SET 
                    in_strategy_names = ?, 
                    out_strategy_names = ?, 
                    in_params = ?, 
                    out_params = ?, 
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
                    update_date = CURRENT_TIMESTAMP WHERE symbol = ?""";
            jdbcTemplate.update(updateSql,
                    inStrategyNamesJson,
                    outStrategyNamesJson,
                    inParamsJson,
                    outParamsJson,
                    result.backtestResult.rendement,
                    result.backtestResult.tradeCount,
                    result.backtestResult.winRate,
                    result.backtestResult.maxDrawdown,
                    result.backtestResult.avgPnL,
                    result.backtestResult.profitFactor,
                    result.backtestResult.avgTradeBars,
                    result.backtestResult.maxTradeGain,
                    result.backtestResult.maxTradeLoss,
                    result.backtestResult.scoreSwingTrade,
                    StrategieBackTest.INITIAL_CAPITAL,
                    StrategieBackTest.RISK_PER_TRADE,
                    StrategieBackTest.STOP_LOSS_PCT,
                    StrategieBackTest.TAKE_PROFIL_PCT,
                    symbol);
        } else {
            // INSERT
            String insertSql = """
                    INSERT INTO best_in_out_mix_strategy (
                    symbol, 
                    in_strategy_names, 
                    out_strategy_names,
                    in_params, 
                    out_params, 
                    rendement,
                    trade_count,
                    win_rate,
                    max_drawdown,
                    avg_pnl,
                    profit_factor,
                    avg_trade_bars,
                    max_trade_gain,
                    max_trade_loss,
                    score_swing_trade, 
                    initial_capital, 
                    risk_per_trade, 
                    stop_loss_pct, 
                    take_profit_pct, 
                    update_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""";
            jdbcTemplate.update(insertSql,
                    symbol,
                    inStrategyNamesJson,
                    outStrategyNamesJson,
                    inParamsJson,
                    outParamsJson,
                    result.backtestResult.rendement,
                    result.backtestResult.tradeCount,
                    result.backtestResult.winRate,
                    result.backtestResult.maxDrawdown,
                    result.backtestResult.avgPnL,
                    result.backtestResult.profitFactor,
                    result.backtestResult.avgTradeBars,
                    result.backtestResult.maxTradeGain,
                    result.backtestResult.maxTradeLoss,
                    result.backtestResult.scoreSwingTrade,
                    StrategieBackTest.INITIAL_CAPITAL,
                    StrategieBackTest.RISK_PER_TRADE,
                    StrategieBackTest.STOP_LOSS_PCT,
                    StrategieBackTest.TAKE_PROFIL_PCT);
        }
    }

    /**
     * Récupère le BestCombinationResult le plus récent pour un symbole
     * Prend en compte initialCapital, riskPerTrade, stopLossPct, takeProfitPct
     */
    public BestCombinationResult getBestCombinationResult(String symbol) {
        String sql = "SELECT * FROM best_in_out_mix_strategy WHERE symbol = ? ORDER BY update_date DESC LIMIT 1";
        List<Map<String, Object>> rows = jdbcTemplate.queryForList(sql, symbol);
        if (rows.isEmpty()) return null;
        Map<String, Object> row = rows.get(0);
        BestCombinationResult result = new BestCombinationResult();
        result.inStrategyNames = gson.fromJson((String) row.get("in_strategy_names"), new TypeToken<List<String>>(){}.getType());
        result.outStrategyNames = gson.fromJson((String) row.get("out_strategy_names"), new TypeToken<List<String>>(){}.getType());
        result.inParams = gson.fromJson((String) row.get("in_params"), new TypeToken<Map<String, Object>>(){}.getType());
        result.outParams = gson.fromJson((String) row.get("out_params"), new TypeToken<Map<String, Object>>(){}.getType());
        result.initialCapital = row.get("initial_capital") != null ? ((Number) row.get("initial_capital")).doubleValue() : 0.0;
        result.riskPerTrade = row.get("risk_per_trade") != null ? ((Number) row.get("risk_per_trade")).doubleValue() : 0.0;
        result.stopLossPct = row.get("stop_loss_pct") != null ? ((Number) row.get("stop_loss_pct")).doubleValue() : 0.0;
        result.takeProfitPct = row.get("take_profit_pct") != null ? ((Number) row.get("take_profit_pct")).doubleValue() : 0.0;
        result.backtestResult = new RiskResult(
                row.get("rendement") != null ? ((Number) row.get("rendement")).doubleValue() : 0.0,
                row.get("max_drawdown") != null ? ((Number) row.get("max_drawdown")).doubleValue() : 0.0,
                row.get("trade_count") != null ? ((Number) row.get("trade_count")).intValue() : 0,
                row.get("win_rate") != null ? ((Number) row.get("win_rate")).doubleValue() : 0.0,
                row.get("avg_pnl") != null ? ((Number) row.get("avg_pnl")).doubleValue() : 0.0,
                row.get("profit_factor") != null ? ((Number) row.get("profit_factor")).doubleValue() : 0.0,
                row.get("avg_trade_bars") != null ? ((Number) row.get("avg_trade_bars")).doubleValue() : 0.0,
                row.get("max_trade_gain") != null ? ((Number) row.get("max_trade_gain")).doubleValue() : 0.0,
                row.get("max_trade_loss") != null ? ((Number) row.get("max_trade_loss")).doubleValue() : 0.0,
                row.get("score_swing_trade") != null ? ((Number) row.get("score_swing_trade")).doubleValue() : 0.0
        );
        result.score = result.backtestResult.rendement;
        return result;
    }

    public SignalType getSignal(String symbol) {
        BestCombinationResult bestCombinationResult = getBestCombinationResult(symbol);
        BarSeries barSeries = strategieHelper.getAndUpdateDBDailyValu(symbol, TradeConstant.NOMBRE_TOTAL_BOUGIES);
        if (bestCombinationResult == null || barSeries == null || barSeries.getBarCount() == 0) {
            return SignalType.NONE;
        }
        // Recréer les stratégies d'entrée
        List<TradeStrategy> inStrategies = new ArrayList<>();
        for (String name : bestCombinationResult.inStrategyNames) {
            Object params = bestCombinationResult.inParams.get(name.replace("Strategy", ""));
            if (name.equals("ImprovedTrendFollowingStrategy") && params instanceof StrategieBackTest.ImprovedTrendFollowingParams) {
                StrategieBackTest.ImprovedTrendFollowingParams p = (StrategieBackTest.ImprovedTrendFollowingParams) params;
                inStrategies.add(new ImprovedTrendFollowingStrategy(p.trendPeriod, p.shortMaPeriod, p.longMaPeriod, p.breakoutThreshold, p.useRsiFilter, p.rsiPeriod));
            } else if (name.equals("SmaCrossoverStrategy") && params instanceof StrategieBackTest.SmaCrossoverParams) {
                StrategieBackTest.SmaCrossoverParams p = (StrategieBackTest.SmaCrossoverParams) params;
                inStrategies.add(new SmaCrossoverStrategy(p.shortPeriod, p.longPeriod));
            } else if (name.equals("RsiStrategy") && params instanceof StrategieBackTest.RsiParams) {
                StrategieBackTest.RsiParams p = (StrategieBackTest.RsiParams) params;
                inStrategies.add(new RsiStrategy(p.rsiPeriod, p.oversold, p.overbought));
            } else if (name.equals("BreakoutStrategy") && params instanceof StrategieBackTest.BreakoutParams) {
                StrategieBackTest.BreakoutParams p = (StrategieBackTest.BreakoutParams) params;
                inStrategies.add(new BreakoutStrategy(p.lookbackPeriod));
            } else if (name.equals("MacdStrategy") && params instanceof StrategieBackTest.MacdParams) {
                StrategieBackTest.MacdParams p = (StrategieBackTest.MacdParams) params;
                inStrategies.add(new MacdStrategy(p.shortPeriod, p.longPeriod, p.signalPeriod));
            } else if (name.equals("MeanReversionStrategy") && params instanceof StrategieBackTest.MeanReversionParams) {
                StrategieBackTest.MeanReversionParams p = (StrategieBackTest.MeanReversionParams) params;
                inStrategies.add(new MeanReversionStrategy(p.smaPeriod, p.threshold));
            }
        }
        // Recréer les stratégies de sortie
        List<TradeStrategy> outStrategies = new ArrayList<>();
        for (String name : bestCombinationResult.outStrategyNames) {
            Object params = bestCombinationResult.outParams.get(name.replace("Strategy", ""));
            if (name.equals("ImprovedTrendFollowingStrategy") && params instanceof StrategieBackTest.ImprovedTrendFollowingParams) {
                StrategieBackTest.ImprovedTrendFollowingParams p = (StrategieBackTest.ImprovedTrendFollowingParams) params;
                outStrategies.add(new ImprovedTrendFollowingStrategy(p.trendPeriod, p.shortMaPeriod, p.longMaPeriod, p.breakoutThreshold, p.useRsiFilter, p.rsiPeriod));
            } else if (name.equals("SmaCrossoverStrategy") && params instanceof StrategieBackTest.SmaCrossoverParams) {
                StrategieBackTest.SmaCrossoverParams p = (StrategieBackTest.SmaCrossoverParams) params;
                outStrategies.add(new SmaCrossoverStrategy(p.shortPeriod, p.longPeriod));
            } else if (name.equals("RsiStrategy") && params instanceof StrategieBackTest.RsiParams) {
                StrategieBackTest.RsiParams p = (StrategieBackTest.RsiParams) params;
                outStrategies.add(new RsiStrategy(p.rsiPeriod, p.oversold, p.overbought));
            } else if (name.equals("BreakoutStrategy") && params instanceof StrategieBackTest.BreakoutParams) {
                StrategieBackTest.BreakoutParams p = (StrategieBackTest.BreakoutParams) params;
                outStrategies.add(new BreakoutStrategy(p.lookbackPeriod));
            } else if (name.equals("MacdStrategy") && params instanceof StrategieBackTest.MacdParams) {
                StrategieBackTest.MacdParams p = (StrategieBackTest.MacdParams) params;
                outStrategies.add(new MacdStrategy(p.shortPeriod, p.longPeriod, p.signalPeriod));
            } else if (name.equals("MeanReversionStrategy") && params instanceof StrategieBackTest.MeanReversionParams) {
                StrategieBackTest.MeanReversionParams p = (StrategieBackTest.MeanReversionParams) params;
                outStrategies.add(new MeanReversionStrategy(p.smaPeriod, p.threshold));
            }
        }
        // Combiner les règles d'entrée et de sortie
        Rule entryRule = null;
        Rule exitRule = null;
        for (TradeStrategy strat : inStrategies) {
            if (entryRule == null) entryRule = strat.getEntryRule(barSeries);
            else entryRule = entryRule.or(strat.getEntryRule(barSeries));
        }
        for (TradeStrategy strat : outStrategies) {
            if (exitRule == null) exitRule = strat.getExitRule(barSeries);
            else exitRule = exitRule.or(strat.getExitRule(barSeries));
        }
        int lastIndex = barSeries.getEndIndex();
        if (entryRule != null && entryRule.isSatisfied(lastIndex)) {
            return SignalType.BUY;
        } else if (exitRule != null && exitRule.isSatisfied(lastIndex)) {
            return SignalType.SELL;
        } else {
            return SignalType.NONE;
        }
    }

    /**
     * Récupère tous les symboles depuis la table alpaca_asset
     */
    public List<String> getAllAssetSymbolsEligibleFromDb() {
        String sql = "SELECT symbol FROM trade_ai.alpaca_asset WHERE status = 'active' and eligible = true;";
        return jdbcTemplate.queryForList(sql, String.class);
    }


    public void calculMixStrategies(){
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
                        String checkSql = "SELECT COUNT(*) FROM best_in_out_mix_strategy WHERE symbol = ?";
                        int count = jdbcTemplate.queryForObject(checkSql, Integer.class, symbol);
                        if(count > 0){
                            isCalcul = false;
                            TradeUtils.log("calculMixStrategies: symbole "+symbol+" déjà en base, on passe");
                        }
                    }
                    if(isCalcul){
                        BestCombinationResult result = findBestCombinationGlobal(symbol);
                        double scoreST =  TradeUtils.calculerScoreSwingTrade(result.getBacktestResult());
                        result.backtestResult.setScoreSwingTrade(scoreST);
                        this.saveBestCombinationResult(symbol, result);
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
        TradeUtils.log("calculMixStrategies: total: "+listeDbSymbols.size()+", nbInsert: "+nbInsert.get()+", error: " + error.get());
    }



    public SignalType getBestSignal(String symbol){
        BestCombinationResult bestCombinationResult = getBestCombinationResult(symbol);
        BestInOutStrategy best = strategieHelper.getBestInOutStrategy(symbol);
        if(bestCombinationResult == null && best == null){
            return SignalType.NONE;
        }else if(bestCombinationResult == null) {
            return strategieHelper.getBestInOutSignal(symbol);
        }else if(best == null) {
            return this.getSignal(symbol);
        }else{
            if(bestCombinationResult.score > best.getResult().getRendement()){
                return this.getSignal(symbol);
            }else {
                return strategieHelper.getBestInOutSignal(symbol);
            }
        }
    }


    public void calculScoreST(){
        String selectSql = "SELECT symbol, backtest_result FROM best_in_out_mix_strategy";
        List<Map<String, Object>> rows = jdbcTemplate.queryForList(selectSql);
        for (Map<String, Object> row : rows) {
            String symbol = (String) row.get("symbol");
            String backtestResultJson = (String) row.get("backtest_result");
            if (backtestResultJson == null || backtestResultJson.isEmpty()) continue;
            RiskResult result = null;
            try {
                result = gson.fromJson(backtestResultJson, RiskResult.class);
            } catch (Exception e) {
                continue;
            }
            if (result == null) continue;
            double scoreST = TradeUtils.calculerScoreSwingTrade(result);
            String updateSql = "UPDATE best_in_out_mix_strategy SET score_swing_trade = ? WHERE symbol = ?";
            jdbcTemplate.update(updateSql, scoreST, symbol);
        }
    }

}
