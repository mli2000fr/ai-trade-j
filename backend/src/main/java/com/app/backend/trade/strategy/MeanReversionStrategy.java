package com.app.backend.trade.strategy;

import org.ta4j.core.BarSeries;
import org.ta4j.core.Rule;
import org.ta4j.core.indicators.SMAIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.rules.OverIndicatorRule;
import org.ta4j.core.rules.UnderIndicatorRule;

public class MeanReversionStrategy implements TradeStrategy {
    private final int smaPeriod;
    private final double thresholdPercent;

    public MeanReversionStrategy() {
        this(20, 2.0); // SMA 20, seuil 2%
    }

    public MeanReversionStrategy(int smaPeriod, double thresholdPercent) {
        this.smaPeriod = smaPeriod;
        this.thresholdPercent = thresholdPercent;
    }

    @Override
    public Rule getEntryRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        SMAIndicator sma = new SMAIndicator(close, smaPeriod);
        // Achat si le prix est sous la SMA de plus de thresholdPercent
        return new UnderIndicatorRule(close, sma.numOf(1 - thresholdPercent / 100).multipliedBy(sma));
    }

    @Override
    public Rule getExitRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        SMAIndicator sma = new SMAIndicator(close, smaPeriod);
        // Vente si le prix est au-dessus de la SMA de plus de thresholdPercent
        return new OverIndicatorRule(close, sma.numOf(1 + thresholdPercent / 100).multipliedBy(sma));
    }

    @Override
    public String getName() {
        return "Mean Reversion";
    }
}
