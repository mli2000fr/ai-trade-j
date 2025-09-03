package com.app.backend.trade.strategy;

import org.ta4j.core.BarSeries;
import org.ta4j.core.Rule;
import org.ta4j.core.indicators.SMAIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.rules.CrossedUpIndicatorRule;
import org.ta4j.core.rules.CrossedDownIndicatorRule;

public class SmaCrossoverStrategy implements TradeStrategy {
    private final int shortPeriod;
    private final int longPeriod;

    public SmaCrossoverStrategy() {
        this(5, 20); // Valeurs par défaut
    }

    public SmaCrossoverStrategy(int shortPeriod, int longPeriod) {
        this.shortPeriod = shortPeriod;
        this.longPeriod = longPeriod;
    }

    @Override
    public Rule getEntryRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        SMAIndicator shortSma = new SMAIndicator(close, shortPeriod);
        SMAIndicator longSma = new SMAIndicator(close, longPeriod);
        return new CrossedUpIndicatorRule(shortSma, longSma);
    }

    @Override
    public Rule getExitRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        SMAIndicator shortSma = new SMAIndicator(close, shortPeriod);
        SMAIndicator longSma = new SMAIndicator(close, longPeriod);
        return new CrossedDownIndicatorRule(shortSma, longSma);
    }

    @Override
    public String getName() {
        return "SMA Crossover";
    }
}

