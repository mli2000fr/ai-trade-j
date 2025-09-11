package com.app.backend.trade.model;

import java.util.List;

public class WalkForwardResultPro {
    private List<ComboResult> segmentResults;
    private double avgRendement;
    private double avgDrawdown;
    private double avgWinRate;
    private double avgProfitFactor;
    private double avgTradeDuration;
    private double avgGainLossRatio;
    private double scoreSwingTrade;
    private int totalTrades;

    public WalkForwardResultPro(List<ComboResult> segmentResults, double avgRendement, double avgDrawdown, double avgWinRate, double avgProfitFactor, double avgTradeDuration, double avgGainLossRatio, double scoreSwingTrade, int totalTrades) {
        this.segmentResults = segmentResults;
        this.avgRendement = avgRendement;
        this.avgDrawdown = avgDrawdown;
        this.avgWinRate = avgWinRate;
        this.avgProfitFactor = avgProfitFactor;
        this.avgTradeDuration = avgTradeDuration;
        this.avgGainLossRatio = avgGainLossRatio;
        this.scoreSwingTrade = scoreSwingTrade;
        this.totalTrades = totalTrades;
    }

    public List<ComboResult> getSegmentResults() { return segmentResults; }
    public double getAvgRendement() { return avgRendement; }
    public double getAvgDrawdown() { return avgDrawdown; }
    public double getAvgWinRate() { return avgWinRate; }
    public double getAvgProfitFactor() { return avgProfitFactor; }
    public double getAvgTradeDuration() { return avgTradeDuration; }
    public double getAvgGainLossRatio() { return avgGainLossRatio; }
    public double getScoreSwingTrade() { return scoreSwingTrade; }
    public int getTotalTrades() { return totalTrades; }
}

