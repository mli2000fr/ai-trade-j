package com.app.backend.trade.strategy;

import org.ta4j.core.BarSeries;
import org.ta4j.core.Rule;
// Importer les stratégies spécifiques
// import com.app.backend.trade.strategy.BreakoutStrategy;
// import com.app.backend.trade.strategy.MacdStrategy;
// import com.app.backend.trade.strategy.MeanReversionStrategy;
// import com.app.backend.trade.strategy.RsiStrategy;
// import com.app.backend.trade.strategy.SmaCrossoverStrategy;
// import com.app.backend.trade.strategy.TrendFollowingStrategy;

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
}
