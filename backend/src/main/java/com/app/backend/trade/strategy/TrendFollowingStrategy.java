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
import org.ta4j.core.indicators.SMAIndicator;
import org.ta4j.core.num.DecimalNum;
import org.ta4j.core.indicators.helpers.TransformIndicator;

public class TrendFollowingStrategy implements TradeStrategy {
    private final int trendPeriod;

    public TrendFollowingStrategy() {
        this(20); // Réduire la valeur par défaut
    }

    public TrendFollowingStrategy(int trendPeriod) {
        this.trendPeriod = trendPeriod;
    }

    @Override
    public Rule getEntryRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);

        // Version plus réaliste : prix au-dessus du plus haut des N dernières périodes avec un seuil
        HighestValueIndicator highestHigh = new HighestValueIndicator(new HighPriceIndicator(series), trendPeriod);
        TransformIndicator highestWithThreshold = new TransformIndicator(highestHigh,
            value -> value.multipliedBy(DecimalNum.valueOf(0.995))); // 0.5% en dessous du plus haut

        // Alternative : utiliser une moyenne mobile pour plus de signaux
        SMAIndicator sma = new SMAIndicator(close, trendPeriod);

        // Condition : prix au-dessus de 99.5% du plus haut OU au-dessus de la SMA
        return new org.ta4j.core.rules.OrRule(
            new OverIndicatorRule(close, highestWithThreshold),
            new CrossedUpIndicatorRule(close, sma)
        );
    }

    @Override
    public Rule getExitRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);

        // Version plus réaliste : prix en-dessous du plus bas avec seuil
        LowestValueIndicator lowestLow = new LowestValueIndicator(new LowPriceIndicator(series), trendPeriod);
        TransformIndicator lowestWithThreshold = new TransformIndicator(lowestLow,
            value -> value.multipliedBy(DecimalNum.valueOf(1.005))); // 0.5% au-dessus du plus bas

        // Alternative : moyenne mobile pour sortie
        SMAIndicator sma = new SMAIndicator(close, trendPeriod);

        return new org.ta4j.core.rules.OrRule(
            new UnderIndicatorRule(close, lowestWithThreshold),
            new CrossedDownIndicatorRule(close, sma)
        );
    }

    @Override
    public String getName() {
        return "Trend Following";
    }
}
