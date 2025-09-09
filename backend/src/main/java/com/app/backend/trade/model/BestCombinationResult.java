package com.app.backend.trade.model;

import com.app.backend.model.RiskResult;
import com.app.backend.trade.strategy.StrategieBackTest;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;
import java.util.Map;
import java.util.HashMap;


@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class BestCombinationResult {
    public List<String> inStrategyNames;
    public List<String> outStrategyNames;
    public double score;
    public Map<String, Object> inParams = new HashMap<>();
    public Map<String, Object> outParams = new HashMap<>();
    public RiskResult backtestResult;

    public double initialCapital;
    public double riskPerTrade;
    public double stopLossPct;
    public double takeProfitPct;
    public int nbSimples;
}

