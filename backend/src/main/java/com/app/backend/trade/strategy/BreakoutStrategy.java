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
import org.ta4j.core.rules.OverIndicatorRule;
import org.ta4j.core.rules.UnderIndicatorRule;
import org.ta4j.core.num.DecimalNum;
import org.ta4j.core.indicators.helpers.TransformIndicator;

public class BreakoutStrategy implements TradeStrategy {
    private final int lookbackPeriod;

    public BreakoutStrategy() {
        this(15); // Réduire la valeur par défaut
    }

    public BreakoutStrategy(int lookbackPeriod) {
        this.lookbackPeriod = lookbackPeriod;
    }

    @Override
    public Rule getEntryRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);

        // Breakout plus réaliste : prix au-dessus de X% du plus haut récent
        HighestValueIndicator highestHigh = new HighestValueIndicator(new HighPriceIndicator(series), lookbackPeriod);
        TransformIndicator breakoutLevel = new TransformIndicator(highestHigh,
            value -> value.multipliedBy(DecimalNum.valueOf(0.998))); // 0.2% en dessous du plus haut

        // Signal d'entrée : prix franchit le niveau de breakout
        return new OverIndicatorRule(close, breakoutLevel);
    }

    @Override
    public Rule getExitRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);

        // Sortie symétrique : prix en-dessous du plus bas récent + marge
        LowestValueIndicator lowestLow = new LowestValueIndicator(new LowPriceIndicator(series), lookbackPeriod);
        TransformIndicator breakdownLevel = new TransformIndicator(lowestLow,
            value -> value.multipliedBy(DecimalNum.valueOf(1.002))); // 0.2% au-dessus du plus bas

        return new UnderIndicatorRule(close, breakdownLevel);
    }

    @Override
    public String getName() {
        return "Breakout";
    }
}
