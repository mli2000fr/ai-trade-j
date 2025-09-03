package com.app.backend.trade.strategy;

import org.ta4j.core.BarSeries;
import org.ta4j.core.Rule;
import org.ta4j.core.indicators.MACDIndicator;
import org.ta4j.core.indicators.EMAIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.rules.CrossedUpIndicatorRule;
import org.ta4j.core.rules.CrossedDownIndicatorRule;

public class MacdStrategy implements TradeStrategy {
    private final int shortPeriod;
    private final int longPeriod;
    private final int signalPeriod;

    public MacdStrategy() {
        this(12, 26, 9); // Valeurs par défaut
    }

    public MacdStrategy(int shortPeriod, int longPeriod, int signalPeriod) {
        this.shortPeriod = shortPeriod;
        this.longPeriod = longPeriod;
        this.signalPeriod = signalPeriod;
    }

    @Override
    public Rule getEntryRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        MACDIndicator macd = new MACDIndicator(close, shortPeriod, longPeriod);
        EMAIndicator signal = new EMAIndicator(macd, signalPeriod);
        return new CrossedUpIndicatorRule(macd, signal);
    }

    @Override
    public Rule getExitRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        MACDIndicator macd = new MACDIndicator(close, shortPeriod, longPeriod);
        EMAIndicator signal = new EMAIndicator(macd, signalPeriod);
        return new CrossedDownIndicatorRule(macd, signal);
    }

    @Override
    public String getName() {
        return "MACD";
    }
}
