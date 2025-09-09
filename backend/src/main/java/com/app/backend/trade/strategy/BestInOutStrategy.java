package com.app.backend.trade.strategy;

import com.app.backend.model.RiskResult;
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
    public final RiskResult result;
    public final double initialCapital;
    public final double riskPerTrade;
    public final double stopLossPct;
    public final double takeProfitPct;
    public final int nbSimples;


    public BestInOutStrategy(String symbol, String entryName, Object entryParams, String exitName, Object exitParams, RiskResult result,
                             double initialCapital, double riskPerTrade, double stopLossPct, double takeProfitPct, int nbSimples) {
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
        this.nbSimples = nbSimples;
    }
}

