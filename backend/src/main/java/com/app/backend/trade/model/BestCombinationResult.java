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
    public String symbol;
    public List<String> inStrategyNames;
    public List<String> outStrategyNames;
    public Map<String, Object> inParams = new HashMap<>();
    public Map<String, Object> outParams = new HashMap<>();
    public RiskResult result;
    public RiskResult check;
    public double rendementSum;
    public double rendementDiff;
    public double rendementScore;
    public ParamsOptim contextOptim;
}

