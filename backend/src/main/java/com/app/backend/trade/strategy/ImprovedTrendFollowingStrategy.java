package com.app.backend.trade.strategy;

import org.ta4j.core.BarSeries;
import org.ta4j.core.Rule;
import org.ta4j.core.indicators.SMAIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.indicators.RSIIndicator;
import org.ta4j.core.rules.CrossedUpIndicatorRule;
import org.ta4j.core.rules.CrossedDownIndicatorRule;
import org.ta4j.core.rules.OverIndicatorRule;
import org.ta4j.core.rules.UnderIndicatorRule;
import org.ta4j.core.rules.AndRule;
import org.ta4j.core.rules.OrRule;
import org.ta4j.core.num.DecimalNum;
import org.ta4j.core.indicators.helpers.TransformIndicator;

/**
 * Version améliorée de la stratégie TrendFollowing
 * - Utilise des moyennes mobiles au lieu des extremes absolus
 * - Combine plusieurs indicateurs pour des signaux plus robustes
 * - Paramètres ajustables pour la sensibilité
 *
 * trendPeriod : Période de référence (10-30)
 * shortMaPeriod / longMaPeriod : Moyennes mobiles (5-15 / 15-25)
 * breakoutThreshold : Seuil de cassure (0.1% à 1%)
 * useRsiFilter : Activer/désactiver le filtre RSI
 */
public class ImprovedTrendFollowingStrategy implements TradeStrategy {
    private final int trendPeriod;
    private final int shortMaPeriod;
    private final int longMaPeriod;
    private final double breakoutThreshold; // Pourcentage au-dessus/en-dessous de la MA
    private final boolean useRsiFilter;
    private final int rsiPeriod;

    public ImprovedTrendFollowingStrategy() {
        this(20, 10, 20, 0.005, true, 14); // Valeurs par défaut
    }

    public ImprovedTrendFollowingStrategy(int trendPeriod) {
        this(trendPeriod, trendPeriod/2, trendPeriod, 0.005, true, 14);
    }

    public ImprovedTrendFollowingStrategy(int trendPeriod, int shortMaPeriod, int longMaPeriod,
                                        double breakoutThreshold, boolean useRsiFilter, int rsiPeriod) {
        this.trendPeriod = trendPeriod;
        this.shortMaPeriod = shortMaPeriod;
        this.longMaPeriod = longMaPeriod;
        this.breakoutThreshold = breakoutThreshold;
        this.useRsiFilter = useRsiFilter;
        this.rsiPeriod = rsiPeriod;
    }

    @Override
    public Rule getEntryRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        SMAIndicator shortSma = new SMAIndicator(close, shortMaPeriod);
        SMAIndicator longSma = new SMAIndicator(close, longMaPeriod);

        // Créer un indicateur avec threshold pour la moyenne mobile longue
        TransformIndicator longSmaWithThreshold = new TransformIndicator(longSma,
            value -> value.multipliedBy(DecimalNum.valueOf(1 + breakoutThreshold)));

        // Condition 1: Prix au-dessus de la moyenne mobile longue + threshold
        Rule priceAboveLongSmaRule = new OverIndicatorRule(close, longSmaWithThreshold);

        // Condition 2: Moyenne courte au-dessus de la moyenne longue (trend up)
        Rule shortAboveLongRule = new OverIndicatorRule(shortSma, longSma);

        // Condition 3: Prix franchit la moyenne courte vers le haut
        Rule priceCrossAboveShortSmaRule = new CrossedUpIndicatorRule(close, shortSma);

        // Condition de base plus souple : au moins 2 conditions sur 3
        Rule trendCondition = new AndRule(priceAboveLongSmaRule, shortAboveLongRule);
        Rule basicEntryRule = new OrRule(
            trendCondition,  // Trend fort
            new AndRule(shortAboveLongRule, priceCrossAboveShortSmaRule)  // Trend + signal
        );

        if (useRsiFilter) {
            // Filtre RSI plus permissif : éviter seulement la surachat extrême
            RSIIndicator rsi = new RSIIndicator(close, rsiPeriod);
            Rule rsiNotOverboughtRule = new UnderIndicatorRule(rsi, DecimalNum.valueOf(80)); // 80 au lieu de 75
            return new AndRule(basicEntryRule, rsiNotOverboughtRule);
        }

        return basicEntryRule;
    }

    @Override
    public Rule getExitRule(BarSeries series) {
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        SMAIndicator shortSma = new SMAIndicator(close, shortMaPeriod);
        SMAIndicator longSma = new SMAIndicator(close, longMaPeriod);

        // Créer un indicateur avec threshold pour la moyenne mobile longue
        TransformIndicator longSmaWithThreshold = new TransformIndicator(longSma,
            value -> value.multipliedBy(DecimalNum.valueOf(1 - breakoutThreshold)));

        // Condition 1: Prix en-dessous de la moyenne mobile longue - threshold
        Rule priceBelowLongSmaRule = new UnderIndicatorRule(close, longSmaWithThreshold);

        // Condition 2: Prix franchit la moyenne courte vers le bas
        Rule priceCrossBelowShortSmaRule = new CrossedDownIndicatorRule(close, shortSma);

        // Condition 3: Moyenne courte en-dessous de la moyenne longue (trend down)
        Rule shortBelowLongRule = new UnderIndicatorRule(shortSma, longSma);

        // Sortie agressive ou conservative
        Rule conservativeExit = new AndRule(priceBelowLongSmaRule, shortBelowLongRule);

        return new OrRule(priceCrossBelowShortSmaRule, conservativeExit);
    }

    @Override
    public String getName() {
        return "Improved Trend Following";
    }

    // Getters pour les paramètres
    public int getTrendPeriod() { return trendPeriod; }
    public int getShortMaPeriod() { return shortMaPeriod; }
    public int getLongMaPeriod() { return longMaPeriod; }
    public double getBreakoutThreshold() { return breakoutThreshold; }
    public boolean isUseRsiFilter() { return useRsiFilter; }
    public int getRsiPeriod() { return rsiPeriod; }
}
