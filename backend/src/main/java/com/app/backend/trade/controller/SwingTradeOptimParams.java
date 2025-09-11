package com.app.backend.trade.controller;

public class SwingTradeOptimParams {
    public int trendMaMin = 10;
    public int trendMaMax = 50;
    public int trendMaStep = 5;
    public int trendShortMaMin = 5;
    public int trendShortMaMax = 20;
    public int trendShortMaStep = 5;
    public int trendLongMaMin = 20;
    public int trendLongMaMax = 60;
    public int trendLongMaStep = 5;
    public double trendBreakoutMin = 0.01;
    public double trendBreakoutMax = 0.03;
    public double trendBreakoutStep = 0.002;

    public int smaShortMin = 5;
    public int smaShortMax = 20;
    public int smaLongMin = 20;
    public int smaLongMax = 60;

    public int rsiPeriodMin = 7;
    public int rsiPeriodMax = 14;
    public int rsiOversoldMin = 30;
    public int rsiOversoldMax = 40;
    public int rsiOverboughtMin = 60;
    public int rsiOverboughtMax = 70;
    public int rsiStep = 1;

    public int breakoutLookbackMin = 10;
    public int breakoutLookbackMax = 30;

    public int macdShortMin = 8;
    public int macdShortMax = 15;
    public int macdLongMin = 16;
    public int macdLongMax = 30;
    public int macdSignalMin = 5;
    public int macdSignalMax = 10;

    public int meanRevSmaMin = 10;
    public int meanRevSmaMax = 30;
    public double meanRevThresholdMin = 2.0;
    public double meanRevThresholdMax = 6.0;
    public double meanRevThresholdStep = 0.5;

    // Paramètre avancé pour la gestion du risque
    public double riskLevel = 0.01; // 1% du capital par trade
}
