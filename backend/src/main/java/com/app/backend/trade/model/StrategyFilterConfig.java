package com.app.backend.trade.model;

public class StrategyFilterConfig {
    public double maxDrawdown = 0.5;
    public double minProfitFactor = 1.0;
    public double minWinRate = 0.2;
    public int minAvgTradeBars = 1;
    public int maxAvgTradeBars = 30;
    public double minGainLossRatio = 0.7;
    public int maxParamCount = 15;

    public StrategyFilterConfig() {}

    public StrategyFilterConfig(double maxDrawdown, double minProfitFactor, double minWinRate,
                                int minAvgTradeBars, int maxAvgTradeBars, double minGainLossRatio, int maxParamCount) {
        this.maxDrawdown = maxDrawdown;
        this.minProfitFactor = minProfitFactor;
        this.minWinRate = minWinRate;
        this.minAvgTradeBars = minAvgTradeBars;
        this.maxAvgTradeBars = maxAvgTradeBars;
        this.minGainLossRatio = minGainLossRatio;
        this.maxParamCount = maxParamCount;
    }
}

