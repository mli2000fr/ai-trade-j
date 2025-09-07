package com.app.backend.trade.strategy;

import com.app.backend.trade.strategy.StrategieBackTest;
import lombok.Getter;

/**
 * Structure pour stocker la meilleure combinaison IN/OUT et ses résultats.
 */
@Getter
public class BestInOutStrategy {
    public final String symbol;
    public final String entryName;
    public final Object entryParams;
    public final String exitName;
    public final Object exitParams;
    public final StrategieBackTest.RiskResult result;
    public final double initialCapital;
    public final double riskPerTrade;
    public final double stopLossPct;
    public final double takeProfitPct;

    public BestInOutStrategy(String symbol, String entryName, Object entryParams, String exitName, Object exitParams, StrategieBackTest.RiskResult result,
                             double initialCapital, double riskPerTrade, double stopLossPct, double takeProfitPct) {
        this.symbol = symbol;
        this.entryName = entryName;
        this.entryParams = entryParams;
        this.exitName = exitName;
        this.exitParams = exitParams;
        this.result = result;
        this.initialCapital = initialCapital;
        this.riskPerTrade = riskPerTrade;
        this.stopLossPct = stopLossPct;
        this.takeProfitPct = takeProfitPct;
    }
}

