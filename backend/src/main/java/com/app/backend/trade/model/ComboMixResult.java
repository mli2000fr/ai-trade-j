package com.app.backend.trade.model;

import com.app.backend.trade.strategy.ParamsOptim;
import lombok.Builder;
import lombok.Data;
import lombok.ToString;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Builder
@Data
@ToString
public class ComboMixResult {
    private String symbol;
    public List<String> inStrategyNames;
    public List<String> outStrategyNames;
    public Map<String, Object> inParams = new HashMap<>();
    public Map<String, Object> outParams = new HashMap<>();
    private RiskResult result;
    private ParamsOptim paramsOptim;
}
