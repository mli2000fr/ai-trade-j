package com.app.backend.trade.strategy;

import org.ta4j.core.BarSeries;
import org.ta4j.core.TradingRecord;
import org.ta4j.core.Rule;

public interface TradeStrategy {
    Rule getEntryRule(BarSeries series);
    Rule getExitRule(BarSeries series);
    String getName();
}

