package com.app.backend.trade.model;

import lombok.*;

@Builder
@Getter
@Setter
public class RiskResult {
    /** Rendement total du backtest (capital final / capital initial - 1) */
    public double rendement;
    /** Drawdown maximal observé (plus forte baisse du capital) */
    public double maxDrawdown;
    /** Nombre total de trades réalisés */
    public int tradeCount;
    /** Pourcentage de trades gagnants (win rate) */
    public double winRate;
    /** Gain ou perte moyen par trade */
    public double avgPnL;
    /** Profit factor (somme des gains / somme des pertes) */
    public double profitFactor;
    /** Nombre moyen de bougies par trade (durée moyenne d'un trade) */
    public double avgTradeBars;
    /** Maximum gain réalisé sur un trade */
    public double maxTradeGain;
    /** Maximum perte réalisée sur un trade */
    public double maxTradeLoss;

    public double scoreSwingTrade = 0;

    public boolean fltredOut;

}

