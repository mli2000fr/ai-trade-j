package com.app.backend.trade.model;

import com.app.backend.trade.lstm.LstmTradePredictor;

public class TradeStylePrediction implements java.io.Serializable {
    public String symbol;
    public double lastClose;
    public double predictedClose;
    public double deltaPct;
    public String tendance; // UP/DOWN/STABLE
    public String action;   // BUY/HOLD/SELL
    public double atrPct;
    public double rsi;
    public double volumeRatio;
    public double thresholdAtrAdaptive;
    public double percentileThreshold;
    public double signalStrength;
    public boolean contrarianAdjusted;
    public String contrarianReason;
    public boolean rsiFiltered;
    public boolean volumeFiltered;
    public boolean entryLogicOr;
    public double aggressivenessBoost;
    public int windowSize;
    public String comment;
    public LstmTradePredictor.LoadedModel loadedModel;
}
