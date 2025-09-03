package com.app.backend.trade.strategy;

import org.ta4j.core.BarSeries;
import org.ta4j.core.Rule;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.indicators.helpers.HighPriceIndicator;
import org.ta4j.core.indicators.helpers.LowPriceIndicator;
import org.ta4j.core.indicators.helpers.HighestValueIndicator;
import org.ta4j.core.indicators.helpers.LowestValueIndicator;
import org.ta4j.core.rules.CrossedUpIndicatorRule;
import org.ta4j.core.rules.CrossedDownIndicatorRule;

public class TrendFollowingStrategy implements TradeStrategy {
    private final int trendPeriod;

    public TrendFollowingStrategy() {
        this(50); // Valeur par défaut
    }

    public TrendFollowingStrategy(int trendPeriod) {
        this.trendPeriod = trendPeriod;
    }

    @Override
    public Rule getEntryRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        HighestValueIndicator highestHigh = new HighestValueIndicator(new HighPriceIndicator(series), trendPeriod);
        return new CrossedUpIndicatorRule(close, highestHigh);
    }

    @Override
    public Rule getExitRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        LowestValueIndicator lowestLow = new LowestValueIndicator(new LowPriceIndicator(series), trendPeriod);
        return new CrossedDownIndicatorRule(close, lowestLow);
    }

    @Override
    public String getName() {
        return "Trend Following";
    }
}
