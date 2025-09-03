package com.app.backend.trade.model;
import java.util.List;

public class StrategyListDto {
    private List<String> allStrategies;
    private List<String> activeStrategies;
    private String combinationMode;
    private List<String> logs;

    public StrategyListDto(List<String> allStrategies, List<String> activeStrategies, String combinationMode, List<String> logs) {
        this.allStrategies = allStrategies;
        this.activeStrategies = activeStrategies;
        this.combinationMode = combinationMode;
        this.logs = logs;
    }

    public List<String> getAllStrategies() { return allStrategies; }
    public List<String> getActiveStrategies() { return activeStrategies; }
    public String getCombinationMode() { return combinationMode; }
    public List<String> getLogs() { return logs; }

    public void setAllStrategies(List<String> allStrategies) { this.allStrategies = allStrategies; }
    public void setActiveStrategies(List<String> activeStrategies) { this.activeStrategies = activeStrategies; }
    public void setCombinationMode(String combinationMode) { this.combinationMode = combinationMode; }
    public void setLogs(List<String> logs) { this.logs = logs; }
}
