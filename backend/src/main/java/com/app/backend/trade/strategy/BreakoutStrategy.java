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

public class BreakoutStrategy implements TradeStrategy {
    private final int lookbackPeriod;

    public BreakoutStrategy() {
        this(20); // Valeur par défaut
    }

    public BreakoutStrategy(int lookbackPeriod) {
        this.lookbackPeriod = lookbackPeriod;
    }

    @Override
    public Rule getEntryRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        HighestValueIndicator highestHigh = new HighestValueIndicator(new HighPriceIndicator(series), lookbackPeriod);
        return new CrossedUpIndicatorRule(close, highestHigh);
    }

    @Override
    public Rule getExitRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        LowestValueIndicator lowestLow = new LowestValueIndicator(new LowPriceIndicator(series), lookbackPeriod);
        return new CrossedDownIndicatorRule(close, lowestLow);
    }

    @Override
    public String getName() {
        return "Breakout";
    }
}
