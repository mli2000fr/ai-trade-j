package com.app.backend.trade.model;
import java.util.List;

public class SetActiveStrategiesRequest {
    private List<String> strategyNames;
    public List<String> getStrategyNames() { return strategyNames; }
    public void setStrategyNames(List<String> strategyNames) { this.strategyNames = strategyNames; }
}

