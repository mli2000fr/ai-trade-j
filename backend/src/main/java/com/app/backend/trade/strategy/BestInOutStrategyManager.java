package com.app.backend.trade.strategy;

import com.app.backend.trade.model.RiskResult;
import com.app.backend.trade.util.TradeUtils;
import com.app.backend.trade.model.DailyValue;
import org.ta4j.core.BarSeries;
import java.util.List;

/**
 * Gestion dédiée des stratégies IN/OUT croisées pour alléger StrategieHelper.
 */
public class BestInOutStrategyManager {
    private final StrategieBackTest strategieBackTest;

    public BestInOutStrategyManager(StrategieBackTest strategieBackTest) {
        this.strategieBackTest = strategieBackTest;
    }

    /**
     * Optimise toutes les combinaisons croisées de stratégies IN/OUT pour un symbole.
     * @param symbol le symbole à traiter
     * @param dailyValues les valeurs historiques
     * @return BestInOutStrategy (meilleure combinaison)
     */
    public BestInOutStrategy optimiseBestInOutByWalkForward(String symbol, List<DailyValue> dailyValues) {
        BarSeries series = TradeUtils.mapping(dailyValues);
        // Optimisation des paramètres pour chaque stratégie
        StrategieBackTest.ImprovedTrendFollowingParams bestImprovedTrend = strategieBackTest.optimiseImprovedTrendFollowingParameters(series, 10, 30, 5, 15, 15, 25, 0.001, 0.01, 0.002);
        StrategieBackTest.SmaCrossoverParams bestSmaCrossover = strategieBackTest.optimiseSmaCrossoverParameters(series, 5, 20, 10, 50);
        StrategieBackTest.RsiParams bestRsi = strategieBackTest.optimiseRsiParameters(series, 10, 20, 20, 40, 5, 60, 80, 5);
        StrategieBackTest.BreakoutParams bestBreakout = strategieBackTest.optimiseBreakoutParameters(series, 5, 50);
        StrategieBackTest.MacdParams bestMacd = strategieBackTest.optimiseMacdParameters(series, 8, 16, 20, 30, 6, 12);
        StrategieBackTest.MeanReversionParams bestMeanReversion = strategieBackTest.optimiseMeanReversionParameters(series, 10, 30, 1.0, 5.0, 0.5);

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
                com.app.backend.trade.strategy.TradeStrategy entryStrategy = createStrategy(entryName, entryParams);
                com.app.backend.trade.strategy.TradeStrategy exitStrategy = createStrategy(exitName, exitParams);
                StrategieBackTest.CombinedTradeStrategy combined = new StrategieBackTest.CombinedTradeStrategy(entryStrategy, exitStrategy);
                RiskResult result = strategieBackTest.backtestStrategyRisk(combined, series);
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
                                    .nbSimples(dailyValues.size())
                                    .build())
                            .result(result)
                            .build();
                }
            }
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

