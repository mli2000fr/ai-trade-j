package com.app.backend.model;

import lombok.Getter;

@Getter
public class RiskResult {
    /** Rendement total du backtest (capital final / capital initial - 1) */
    public final double rendement;
    /** Drawdown maximal observé (plus forte baisse du capital) */
    public final double maxDrawdown;
    /** Nombre total de trades réalisés */
    public final int tradeCount;
    /** Pourcentage de trades gagnants (win rate) */
    public final double winRate;
    /** Gain ou perte moyen par trade */
    public final double avgPnL;
    /** Profit factor (somme des gains / somme des pertes) */
    public final double profitFactor;
    /** Nombre moyen de bougies par trade (durée moyenne d'un trade) */
    public final double avgTradeBars;
    /** Maximum gain réalisé sur un trade */
    public final double maxTradeGain;
    /** Maximum perte réalisée sur un trade */
    public final double maxTradeLoss;

    public RiskResult(double rendement, double maxDrawdown, int tradeCount, double winRate, double avgPnL, double profitFactor, double avgTradeBars, double maxTradeGain, double maxTradeLoss) {
        this.rendement = rendement;
        this.maxDrawdown = maxDrawdown;
        this.tradeCount = tradeCount;
        this.winRate = winRate;
        this.avgPnL = avgPnL;
        this.profitFactor = profitFactor;
        this.avgTradeBars = avgTradeBars;
        this.maxTradeGain = maxTradeGain;
        this.maxTradeLoss = maxTradeLoss;
    }
}

