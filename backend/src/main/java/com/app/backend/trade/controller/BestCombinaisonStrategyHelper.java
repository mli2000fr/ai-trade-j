package com.app.backend.trade.controller;

import com.app.backend.trade.strategy.*;
import com.app.backend.trade.util.TradeUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.ta4j.core.BarSeries;
import org.ta4j.core.Rule;
import java.util.*;

@Controller
public class BestCombinaisonStrategyHelper {


    private StrategieHelper strategieHelper;


    @Autowired
    public BestCombinaisonStrategyHelper(StrategieHelper strategieHelper) {
        this.strategieHelper = strategieHelper;
    }

    public BestCombinationResult findBestCombination(String symbol, int in, int out) {
        // Récupérer les BarSeries depuis la base via StrategieHelper
        BarSeries series = strategieHelper.getAndUpdateDBDailyValu(symbol, 500);
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
        BestCombinationResult bestGlobal = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int in = 1; in <= 2; in++) {
            for (int out = 1; out <= 2; out++) {
                BestCombinationResult result = findBestCombination(symbol, in, out);
                TradeUtils.log("Global search: in=" + in + ", out=" + out + " => score=" + result.score);
                if (result != null && result.score > bestScore) {
                    bestScore = result.score;
                    bestGlobal = result;
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

    // Évalue la combinaison selon la logique métier de StrategieHelper
    private BestCombinationResult evaluateCombination(String symbol, List<BarSeries> seriesList, List<Class<? extends TradeStrategy>> inCombo, List<Class<? extends TradeStrategy>> outCombo) {
        BestCombinationResult resultObj = new BestCombinationResult();
        if (seriesList == null || seriesList.isEmpty()) {
            resultObj.score = Double.NEGATIVE_INFINITY;
            return resultObj;
        }
        BarSeries series = seriesList.get(0);
        StrategieBackTest backTest = new StrategieBackTest();
        List<TradeStrategy> inStrategies = new ArrayList<>();
        List<String> inStrategyNames = new ArrayList<>();
        for (Class<? extends TradeStrategy> clazz : inCombo) {
            inStrategyNames.add(clazz.getSimpleName());
            if (clazz.equals(ImprovedTrendFollowingStrategy.class)) {
                StrategieBackTest.ImprovedTrendFollowingParams params = backTest.optimiseImprovedTrendFollowingParameters(series, 10, 30, 5, 15, 15, 25, 0.001, 0.01, 0.002);
                inStrategies.add(new ImprovedTrendFollowingStrategy(params.trendPeriod, params.shortMaPeriod, params.longMaPeriod, params.breakoutThreshold, params.useRsiFilter, params.rsiPeriod));
                resultObj.inParams.put("ImprovedTrendFollowing", params);
            } else if (clazz.equals(SmaCrossoverStrategy.class)) {
                StrategieBackTest.SmaCrossoverParams params = backTest.optimiseSmaCrossoverParameters(series, 5, 20, 10, 50);
                inStrategies.add(new SmaCrossoverStrategy(params.shortPeriod, params.longPeriod));
                resultObj.inParams.put("SmaCrossover", params);
            } else if (clazz.equals(RsiStrategy.class)) {
                StrategieBackTest.RsiParams params = backTest.optimiseRsiParameters(series, 10, 20, 20, 40, 5, 60, 80, 5);
                inStrategies.add(new RsiStrategy(params.rsiPeriod, params.oversold, params.overbought));
                resultObj.inParams.put("Rsi", params);
            } else if (clazz.equals(BreakoutStrategy.class)) {
                StrategieBackTest.BreakoutParams params = backTest.optimiseBreakoutParameters(series, 5, 50);
                inStrategies.add(new BreakoutStrategy(params.lookbackPeriod));
                resultObj.inParams.put("Breakout", params);
            } else if (clazz.equals(MacdStrategy.class)) {
                StrategieBackTest.MacdParams params = backTest.optimiseMacdParameters(series, 8, 16, 20, 30, 6, 12);
                inStrategies.add(new MacdStrategy(params.shortPeriod, params.longPeriod, params.signalPeriod));
                resultObj.inParams.put("Macd", params);
            } else if (clazz.equals(MeanReversionStrategy.class)) {
                StrategieBackTest.MeanReversionParams params = backTest.optimiseMeanReversionParameters(series, 10, 30, 1.0, 5.0, 0.5);
                inStrategies.add(new MeanReversionStrategy(params.smaPeriod, params.threshold));
                resultObj.inParams.put("MeanReversion", params);
            }
        }
        List<TradeStrategy> outStrategies = new ArrayList<>();
        List<String> outStrategyNames = new ArrayList<>();
        for (Class<? extends TradeStrategy> clazz : outCombo) {
            outStrategyNames.add(clazz.getSimpleName());
            if (clazz.equals(ImprovedTrendFollowingStrategy.class)) {
                StrategieBackTest.ImprovedTrendFollowingParams params = backTest.optimiseImprovedTrendFollowingParameters(series, 10, 30, 5, 15, 15, 25, 0.001, 0.01, 0.002);
                outStrategies.add(new ImprovedTrendFollowingStrategy(params.trendPeriod, params.shortMaPeriod, params.longMaPeriod, params.breakoutThreshold, params.useRsiFilter, params.rsiPeriod));
                resultObj.outParams.put("ImprovedTrendFollowing", params);
            } else if (clazz.equals(SmaCrossoverStrategy.class)) {
                StrategieBackTest.SmaCrossoverParams params = backTest.optimiseSmaCrossoverParameters(series, 5, 20, 10, 50);
                outStrategies.add(new SmaCrossoverStrategy(params.shortPeriod, params.longPeriod));
                resultObj.outParams.put("SmaCrossover", params);
            } else if (clazz.equals(RsiStrategy.class)) {
                StrategieBackTest.RsiParams params = backTest.optimiseRsiParameters(series, 10, 20, 20, 40, 5, 60, 80, 5);
                outStrategies.add(new RsiStrategy(params.rsiPeriod, params.oversold, params.overbought));
                resultObj.outParams.put("Rsi", params);
            } else if (clazz.equals(BreakoutStrategy.class)) {
                StrategieBackTest.BreakoutParams params = backTest.optimiseBreakoutParameters(series, 5, 50);
                outStrategies.add(new BreakoutStrategy(params.lookbackPeriod));
                resultObj.outParams.put("Breakout", params);
            } else if (clazz.equals(MacdStrategy.class)) {
                StrategieBackTest.MacdParams params = backTest.optimiseMacdParameters(series, 8, 16, 20, 30, 6, 12);
                outStrategies.add(new MacdStrategy(params.shortPeriod, params.longPeriod, params.signalPeriod));
                resultObj.outParams.put("Macd", params);
            } else if (clazz.equals(MeanReversionStrategy.class)) {
                StrategieBackTest.MeanReversionParams params = backTest.optimiseMeanReversionParameters(series, 10, 30, 1.0, 5.0, 0.5);
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
                if (tempEntryRule == null) tempEntryRule = strat.getEntryRule(series);
                else tempEntryRule = tempEntryRule.and(strat.getEntryRule(series));
            }
            for (TradeStrategy strat : outStrategies) {
                if (tempExitRule == null) tempExitRule = strat.getExitRule(series);
                else tempExitRule = tempExitRule.and(strat.getExitRule(series));
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
        StrategieBackTest.RiskResult backtestResult = backTest.backtestStrategyRisk(combinedStrategy, series);
        resultObj.score = backtestResult.rendement;
        resultObj.backtestResult = backtestResult;
        resultObj.inStrategyNames = inStrategyNames;
        resultObj.outStrategyNames = outStrategyNames;
        // Optionnel : supprimer ou ignorer les champs inStrategies/outStrategies pour la sérialisation
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

    // Classe interne pour le résultat enrichi
    public static class BestCombinationResult {
        public List<String> inStrategyNames;
        public List<String> outStrategyNames;
        public double score;
        public Map<String, Object> inParams = new HashMap<>();
        public Map<String, Object> outParams = new HashMap<>();
        public StrategieBackTest.RiskResult backtestResult;
    }
}
