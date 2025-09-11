package com.app.backend.trade.controller;

public class SwingTradeOptimParams {
    public int trendMaMin = 10;
    public int trendMaMax = 200;
    public int trendMaStep = 5;
    public int trendShortMaMin = 5;
    public int trendShortMaMax = 100;
    public int trendShortMaStep = 5;
    public int trendLongMaMin = 20;
    public int trendLongMaMax = 200;
    public int trendLongMaStep = 5;
    public double trendBreakoutMin = 0.001;
    public double trendBreakoutMax = 0.05;
    public double trendBreakoutStep = 0.001;

    public int smaShortMin = 5;
    public int smaShortMax = 100;
    public int smaLongMin = 10;
    public int smaLongMax = 200;

    public int rsiPeriodMin = 7;
    public int rsiPeriodMax = 35;
    public int rsiOversoldMin = 20;
    public int rsiOversoldMax = 40;
    public int rsiOverboughtMin = 60;
    public int rsiOverboughtMax = 80;
    public int rsiStep = 1;

    public int breakoutLookbackMin = 10;
    public int breakoutLookbackMax = 100;

    public int macdShortMin = 6;
    public int macdShortMax = 30;
    public int macdLongMin = 12;
    public int macdLongMax = 60;
    public int macdSignalMin = 3;
    public int macdSignalMax = 24;

    public int meanRevSmaMin = 10;
    public int meanRevSmaMax = 120;
    public double meanRevThresholdMin = 0.5;
    public double meanRevThresholdMax = 15.0;
    public double meanRevThresholdStep = 0.25;

    // Paramètre avancé pour la gestion du risque
    public double riskLevel = 0.01; // 1% du capital par trade
}
