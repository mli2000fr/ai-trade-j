package com.app.backend.trade.model;

import com.app.backend.model.RiskResult;
import lombok.Builder;
import lombok.Data;

import java.util.List;

@Builder
@Data
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
    private ComboResult bestCombo;
    private RiskResult check;

}
