package com.app.backend.trade.strategy;

import org.ta4j.core.BarSeries;
import org.ta4j.core.Rule;
import org.ta4j.core.indicators.RSIIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.rules.OverIndicatorRule;
import org.ta4j.core.rules.UnderIndicatorRule;

public class RsiStrategy implements TradeStrategy {
    private final int rsiPeriod;
    private final double oversoldThreshold;
    private final double overboughtThreshold;

    public RsiStrategy() {
        this(14, 30, 70); // Valeurs par défaut
    }

    public RsiStrategy(int rsiPeriod, double oversoldThreshold, double overboughtThreshold) {
        this.rsiPeriod = rsiPeriod;
        this.oversoldThreshold = oversoldThreshold;
        this.overboughtThreshold = overboughtThreshold;
    }

    @Override
    public Rule getEntryRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        RSIIndicator rsi = new RSIIndicator(close, rsiPeriod);
        return new UnderIndicatorRule(rsi, oversoldThreshold);
    }

    @Override
    public Rule getExitRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        RSIIndicator rsi = new RSIIndicator(close, rsiPeriod);
        return new OverIndicatorRule(rsi, overboughtThreshold);
    }

    @Override
    public String getName() {
        return "RSI";
    }
}
