package com.app.backend.trade.strategy;

import com.app.backend.trade.model.RiskResult;
import lombok.*;

@Builder
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class BestInOutStrategy {
    public String name;
    public String symbol;
    public String entryName;
    public Object entryParams;
    public String exitName;
    public Object exitParams;
    public RiskResult finalResult;
    public RiskResult testResult;
    public ParamsOptim paramsOptim;

    public static BestInOutStrategy empty(){
        return empty("");
    }
    public static BestInOutStrategy empty(String symbol) {
        BestInOutStrategy result = new BestInOutStrategy();
        result.symbol = symbol;
        result.entryName = "";
        result.entryParams = null;
        result.exitName = "";
        result.exitParams = null;
        result.finalResult = RiskResult.builder()
            .rendement(0.0)
            .maxDrawdown(0.0)
            .tradeCount(0)
            .winRate(0.0)
            .avgPnL(0.0)
            .profitFactor(0.0)
            .avgTradeBars(0.0)
            .maxTradeGain(0.0)
            .maxTradeLoss(0.0)
            .scoreSwingTrade(0.0)
            .build();
        result.testResult = RiskResult.builder()
            .rendement(0.0)
            .maxDrawdown(0.0)
            .tradeCount(0)
            .winRate(0.0)
            .avgPnL(0.0)
            .profitFactor(0.0)
            .avgTradeBars(0.0)
            .maxTradeGain(0.0)
            .maxTradeLoss(0.0)
            .scoreSwingTrade(0.0)
            .build();
        result.paramsOptim = null;
        return result;
    }
}

