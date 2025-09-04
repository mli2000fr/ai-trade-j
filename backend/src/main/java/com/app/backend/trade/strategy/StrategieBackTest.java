package com.app.backend.trade.strategy;

import org.ta4j.core.BarSeries;
import org.ta4j.core.Rule;

public class StrategieBackTest {
    // Backtest générique pour une stratégie TradeStrategy
    private double backtestStrategy(TradeStrategy strategy, BarSeries series) {
        Rule entryRule = strategy.getEntryRule(series);
        Rule exitRule = strategy.getExitRule(series);
        boolean inPosition = false;
        double entryPrice = 0.0;
        double totalReturn = 1.0;
        for (int i = 0; i < series.getBarCount(); i++) {
            if (!inPosition && entryRule.isSatisfied(i)) {
                entryPrice = series.getBar(i).getClosePrice().doubleValue();
                inPosition = true;
            } else if (inPosition && exitRule.isSatisfied(i)) {
                double exitPrice = series.getBar(i).getClosePrice().doubleValue();
                totalReturn *= (exitPrice / entryPrice);
                inPosition = false;
            }
        }
        // Si une position reste ouverte à la fin, on la clôture au dernier prix
        if (inPosition) {
            double exitPrice = series.getBar(series.getEndIndex()).getClosePrice().doubleValue();
            totalReturn *= (exitPrice / entryPrice);
        }
        return totalReturn - 1.0; // rendement total (ex: 0.25 = +25%)
    }

    // Backtest pour BreakoutStrategy
    public double backtestBreakoutStrategy(BarSeries series, int lookbackPeriod) {
        BreakoutStrategy strategy = new BreakoutStrategy(lookbackPeriod);
        return backtestStrategy(strategy, series);
    }

    // Backtest pour MacdStrategy
    public double backtestMacdStrategy(BarSeries series, int shortPeriod, int longPeriod, int signalPeriod) {
        MacdStrategy strategy = new MacdStrategy(shortPeriod, longPeriod, signalPeriod);
        return backtestStrategy(strategy, series);
    }

    // Backtest pour MeanReversionStrategy
    public double backtestMeanReversionStrategy(BarSeries series, int smaPeriod, double threshold) {
        MeanReversionStrategy strategy = new MeanReversionStrategy(smaPeriod, threshold);
        return backtestStrategy(strategy, series);
    }

    // Backtest pour RsiStrategy
    public double backtestRsiStrategy(BarSeries series, int rsiPeriod, double oversoldThreshold, double overboughtThreshold) {
        RsiStrategy strategy = new RsiStrategy(rsiPeriod, oversoldThreshold, overboughtThreshold);
        return backtestStrategy(strategy, series);
    }

    // Backtest pour SmaCrossoverStrategy
    public double backtestSmaCrossoverStrategy(BarSeries series, int shortPeriod, int longPeriod) {
        SmaCrossoverStrategy strategy = new SmaCrossoverStrategy(shortPeriod, longPeriod);
        return backtestStrategy(strategy, series);
    }

    // Backtest pour TrendFollowingStrategy
    public double backtestTrendFollowingStrategy(BarSeries series, int trendPeriod) {
        TrendFollowingStrategy strategy = new TrendFollowingStrategy(trendPeriod);
        return backtestStrategy(strategy, series);
    }

    /**
     * Optimisation des paramètres pour MacdStrategy
     */
    public MacdParams optimiseMacdParameters(BarSeries series, int shortMin, int shortMax, int longMin, int longMax, int signalMin, int signalMax) {
        double bestReturn = Double.NEGATIVE_INFINITY;
        int bestShort = shortMin, bestLong = longMin, bestSignal = signalMin;
        for (int shortPeriod = shortMin; shortPeriod <= shortMax; shortPeriod++) {
            for (int longPeriod = longMin; longPeriod <= longMax; longPeriod++) {
                for (int signalPeriod = signalMin; signalPeriod <= signalMax; signalPeriod++) {
                    double result = backtestMacdStrategy(series, shortPeriod, longPeriod, signalPeriod);
                    if (result > bestReturn) {
                        bestReturn = result;
                        bestShort = shortPeriod;
                        bestLong = longPeriod;
                        bestSignal = signalPeriod;
                    }
                }
            }
        }
        return new MacdParams(bestShort, bestLong, bestSignal, bestReturn);
    }

    /**
     * Optimisation des paramètres pour BreakoutStrategy
     */
    public BreakoutParams optimiseBreakoutParameters(BarSeries series, int lookbackMin, int lookbackMax) {
        double bestReturn = Double.NEGATIVE_INFINITY;
        int bestLookback = lookbackMin;
        for (int lookback = lookbackMin; lookback <= lookbackMax; lookback++) {
            double result = backtestBreakoutStrategy(series, lookback);
            if (result > bestReturn) {
                bestReturn = result;
                bestLookback = lookback;
            }
        }
        return new BreakoutParams(bestLookback, bestReturn);
    }

    /**
     * Optimisation des paramètres pour MeanReversionStrategy
     */
    public MeanReversionParams optimiseMeanReversionParameters(BarSeries series, int smaMin, int smaMax, double thresholdMin, double thresholdMax, double thresholdStep) {
        double bestReturn = Double.NEGATIVE_INFINITY;
        int bestSma = smaMin;
        double bestThreshold = thresholdMin;
        for (int sma = smaMin; sma <= smaMax; sma++) {
            for (double threshold = thresholdMin; threshold <= thresholdMax; threshold += thresholdStep) {
                double result = backtestMeanReversionStrategy(series, sma, threshold);
                if (result > bestReturn) {
                    bestReturn = result;
                    bestSma = sma;
                    bestThreshold = threshold;
                }
            }
        }
        return new MeanReversionParams(bestSma, bestThreshold, bestReturn);
    }

    /**
     * Optimisation des paramètres pour RsiStrategy
     */
    public RsiParams optimiseRsiParameters(BarSeries series, int rsiMin, int rsiMax, double oversoldMin, double oversoldMax, double oversoldStep, double overboughtMin, double overboughtMax, double overboughtStep) {
        double bestReturn = Double.NEGATIVE_INFINITY;
        int bestRsi = rsiMin;
        double bestOversold = oversoldMin;
        double bestOverbought = overboughtMin;
        for (int rsi = rsiMin; rsi <= rsiMax; rsi++) {
            for (double oversold = oversoldMin; oversold <= oversoldMax; oversold += oversoldStep) {
                for (double overbought = overboughtMin; overbought <= overboughtMax; overbought += overboughtStep) {
                    double result = backtestRsiStrategy(series, rsi, oversold, overbought);
                    if (result > bestReturn) {
                        bestReturn = result;
                        bestRsi = rsi;
                        bestOversold = oversold;
                        bestOverbought = overbought;
                    }
                }
            }
        }
        return new RsiParams(bestRsi, bestOversold, bestOverbought, bestReturn);
    }

    /**
     * Optimisation des paramètres pour SmaCrossoverStrategy
     */
    public SmaCrossoverParams optimiseSmaCrossoverParameters(BarSeries series, int shortMin, int shortMax, int longMin, int longMax) {
        double bestReturn = Double.NEGATIVE_INFINITY;
        int bestShort = shortMin;
        int bestLong = longMin;
        for (int shortPeriod = shortMin; shortPeriod <= shortMax; shortPeriod++) {
            for (int longPeriod = longMin; longPeriod <= longMax; longPeriod++) {
                double result = backtestSmaCrossoverStrategy(series, shortPeriod, longPeriod);
                if (result > bestReturn) {
                    bestReturn = result;
                    bestShort = shortPeriod;
                    bestLong = longPeriod;
                }
            }
        }
        return new SmaCrossoverParams(bestShort, bestLong, bestReturn);
    }

    /**
     * Optimisation des paramètres pour TrendFollowingStrategy
     */
    public TrendFollowingParams optimiseTrendFollowingParameters(BarSeries series, int trendMin, int trendMax) {
        double bestReturn = Double.NEGATIVE_INFINITY;
        int bestTrend = trendMin;
        for (int trendPeriod = trendMin; trendPeriod <= trendMax; trendPeriod++) {
            double result = backtestTrendFollowingStrategy(series, trendPeriod);
            if (result > bestReturn) {
                bestReturn = result;
                bestTrend = trendPeriod;
            }
        }
        return new TrendFollowingParams(bestTrend, bestReturn);
    }

    // Classes de paramètres pour le retour des optimisations
    public static class MacdParams {
        public final int shortPeriod, longPeriod, signalPeriod;
        public final double performance;
        public MacdParams(int shortPeriod, int longPeriod, int signalPeriod, double performance) {
            this.shortPeriod = shortPeriod;
            this.longPeriod = longPeriod;
            this.signalPeriod = signalPeriod;
            this.performance = performance;
        }
    }
    public static class BreakoutParams {
        public final int lookbackPeriod;
        public final double performance;
        public BreakoutParams(int lookbackPeriod, double performance) {
            this.lookbackPeriod = lookbackPeriod;
            this.performance = performance;
        }
    }
    public static class MeanReversionParams {
        public final int smaPeriod;
        public final double threshold, performance;
        public MeanReversionParams(int smaPeriod, double threshold, double performance) {
            this.smaPeriod = smaPeriod;
            this.threshold = threshold;
            this.performance = performance;
        }
    }
    public static class RsiParams {
        public final int rsiPeriod;
        public final double oversold, overbought, performance;
        public RsiParams(int rsiPeriod, double oversold, double overbought, double performance) {
            this.rsiPeriod = rsiPeriod;
            this.oversold = oversold;
            this.overbought = overbought;
            this.performance = performance;
        }
    }
    public static class SmaCrossoverParams {
        public final int shortPeriod, longPeriod;
        public final double performance;
        public SmaCrossoverParams(int shortPeriod, int longPeriod, double performance) {
            this.shortPeriod = shortPeriod;
            this.longPeriod = longPeriod;
            this.performance = performance;
        }
    }
    public static class TrendFollowingParams {
        public final int trendPeriod;
        public final double performance;
        public TrendFollowingParams(int trendPeriod, double performance) {
            this.trendPeriod = trendPeriod;
            this.performance = performance;
        }
    }
}
