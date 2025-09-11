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
    private double avgTrainRendement;
    private double avgTestRendement;
    private double overfitRatio;
    private boolean isOverfit;
    private double sharpeRatio;
    private double rendementStdDev;
    private double sortinoRatio;

    public WalkForwardResultPro(List<ComboResult> segmentResults, double avgRendement, double avgDrawdown, double avgWinRate, double avgProfitFactor, double avgTradeDuration, double avgGainLossRatio, double scoreSwingTrade, int totalTrades, double avgTrainRendement, double avgTestRendement, double overfitRatio, boolean isOverfit, double sharpeRatio, double rendementStdDev, double sortinoRatio) {
        this.segmentResults = segmentResults;
        this.avgRendement = avgRendement;
        this.avgDrawdown = avgDrawdown;
        this.avgWinRate = avgWinRate;
        this.avgProfitFactor = avgProfitFactor;
        this.avgTradeDuration = avgTradeDuration;
        this.avgGainLossRatio = avgGainLossRatio;
        this.scoreSwingTrade = scoreSwingTrade;
        this.totalTrades = totalTrades;
        this.avgTrainRendement = avgTrainRendement;
        this.avgTestRendement = avgTestRendement;
        this.overfitRatio = overfitRatio;
        this.isOverfit = isOverfit;
        this.sharpeRatio = sharpeRatio;
        this.rendementStdDev = rendementStdDev;
        this.sortinoRatio = sortinoRatio;
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
    public double getAvgTrainRendement() { return avgTrainRendement; }
    public double getAvgTestRendement() { return avgTestRendement; }
    public double getOverfitRatio() { return overfitRatio; }
    public boolean isOverfit() { return isOverfit; }
    public double getSharpeRatio() { return sharpeRatio; }
    public double getRendementStdDev() { return rendementStdDev; }
    public double getSortinoRatio() { return sortinoRatio; }
}
