package com.app.backend.trade.model;

import java.util.List;

public class WalkForwardResult {
    private List<ComboResult> segmentResults;
    private double avgRendement;
    private double avgDrawdown;
    private double avgWinRate;
    private int totalTrades;

    public WalkForwardResult(List<ComboResult> segmentResults, double avgRendement, double avgDrawdown, double avgWinRate, int totalTrades) {
        this.segmentResults = segmentResults;
        this.avgRendement = avgRendement;
        this.avgDrawdown = avgDrawdown;
        this.avgWinRate = avgWinRate;
        this.totalTrades = totalTrades;
    }

    public List<ComboResult> getSegmentResults() { return segmentResults; }
    public double getAvgRendement() { return avgRendement; }
    public double getAvgDrawdown() { return avgDrawdown; }
    public double getAvgWinRate() { return avgWinRate; }
    public int getTotalTrades() { return totalTrades; }
}

