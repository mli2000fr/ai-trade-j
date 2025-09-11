package com.app.backend.trade.controller;

public class SwingTradeOptimParams {
    public int trendMaMin = 55;
    public int trendMaMax = 155;
    public int trendMaStep = 10;
    public int trendShortMaMin = 27;
    public int trendShortMaMax = 77;
    public int trendShortMaStep = 10;
    public int trendLongMaMin = 60;
    public int trendLongMaMax = 120;
    public int trendLongMaStep = 10;
    public double trendBreakoutMin = 0.013;
    public double trendBreakoutMax = 0.037;
    public double trendBreakoutStep = 0.002;

    public int smaShortMin = 27;
    public int smaShortMax = 77;
    public int smaLongMin = 55;
    public int smaLongMax = 155;

    public int rsiPeriodMin = 16;
    public int rsiPeriodMax = 26;
    public int rsiOversoldMin = 25;
    public int rsiOversoldMax = 35;
    public int rsiOverboughtMin = 65;
    public int rsiOverboughtMax = 75;
    public int rsiStep = 2;

    public int breakoutLookbackMin = 30;
    public int breakoutLookbackMax = 60;

    public int macdShortMin = 13;
    public int macdShortMax = 23;
    public int macdLongMin = 24;
    public int macdLongMax = 48;
    public int macdSignalMin = 8;
    public int macdSignalMax = 16;

    public int meanRevSmaMin = 35;
    public int meanRevSmaMax = 85;
    public double meanRevThresholdMin = 4.0;
    public double meanRevThresholdMax = 11.0;
    public double meanRevThresholdStep = 0.5;

    // Paramètre avancé pour la gestion du risque
    public double riskLevel = 0.01; // 1% du capital par trade
}
