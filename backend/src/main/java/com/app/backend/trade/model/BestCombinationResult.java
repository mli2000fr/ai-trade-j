package com.app.backend.trade.model;

import com.app.backend.trade.strategy.ParamsOptim;
import lombok.*;

import java.util.List;
import java.util.Map;
import java.util.HashMap;

@Builder
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class BestCombinationResult {
    public String name;
    public String symbol;
    public List<String> inStrategyNames;
    public List<String> outStrategyNames;
    public Map<String, Object> inParams = new HashMap<>();
    public Map<String, Object> outParams = new HashMap<>();
    public RiskResult finalResult;
    public RiskResult testResult;
    public double rendementSum;
    public double rendementDiff;
    public double rendementScore;
    public ParamsOptim contextOptim;

    public static BestCombinationResult empty() {
        BestCombinationResult result = new BestCombinationResult();
        result.symbol = "";
        result.inStrategyNames = List.of();
        result.outStrategyNames = List.of();
        result.inParams = new HashMap<>();
        result.outParams = new HashMap<>();
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
        result.rendementSum = 0.0;
        result.rendementDiff = 0.0;
        result.rendementScore = 0.0;
        result.contextOptim = new ParamsOptim();
        return result;
    }
}
