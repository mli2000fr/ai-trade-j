package com.app.backend.trade.strategy;

import com.app.backend.model.RiskResult;
import com.app.backend.trade.model.OptimResult;
import com.app.backend.trade.util.TradeConstant;
import com.app.backend.trade.util.TradeUtils;
import org.springframework.stereotype.Controller;
import org.ta4j.core.BarSeries;
import org.ta4j.core.Rule;

import java.util.List;

@Controller
public class StrategieBackTest {

    // Paramètres de backtest plus réalistes
    public final static double INITIAL_CAPITAL = 10000;  // Capital initial de 10 000$ (plus réaliste)
    public final static double RISK_PER_TRADE = 0.15;    // Risque par trade de 1% (plus conservateur)
    public final static double STOP_LOSS_PCT = 0.05; // Stop loss à 2% (plus serré)
    public final static double TAKE_PROFIL_PCT = 0.1;    // Take profit à 6% (ratio risque/récompense 1:3)

    // Backtest générique pour une stratégie TradeStrategy (rendement simple)
    private double backtestStrategySimple(TradeStrategy strategy, BarSeries series) {
        Rule entryRule = strategy.getEntryRule(series);
        Rule exitRule = strategy.getExitRule(series);
        boolean inPosition = false;
        double entryPrice = 0.0;
        double totalReturn = 1.0;
        for (int i = 0; i < series.getBarCount(); i++) {
            if (!inPosition && entryRule.isSatisfied(i)) {
                entryPrice = series.getBar(i).getClosePrice().doubleValue();
                inPosition = true;
            } else if (inPosition && exitRule.isSatisfied(i)) {
                double exitPrice = series.getBar(i).getClosePrice().doubleValue();
                totalReturn *= (exitPrice / entryPrice);
                inPosition = false;
            }
        }
        // Si une position reste ouverte à la fin, on la clôture au dernier prix
        if (inPosition) {
            double exitPrice = series.getBar(series.getEndIndex()).getClosePrice().doubleValue();
            totalReturn *= (exitPrice / entryPrice);
        }
        return totalReturn - 1.0; // rendement total (ex: 0.25 = +25%)
    }

    /**
     * Backtest avec gestion du risque (stop loss, take profit, money management)
     * Retourne un objet RiskResult avec rendement et drawdown maximal
     */
    public RiskResult backtestStrategyRisk(TradeStrategy strategy, BarSeries series) {
        return backtestStrategyRisk(strategy, series, INITIAL_CAPITAL, RISK_PER_TRADE, STOP_LOSS_PCT, TAKE_PROFIL_PCT);
    }

    public RiskResult backtestStrategyRisk(TradeStrategy strategy, BarSeries series, double initialCapital, double riskPerTrade, double stopLossPct, double takeProfitPct) {
        Rule entryRule = strategy.getEntryRule(series);
        Rule exitRule = strategy.getExitRule(series);
        boolean inPosition = false;
        double entryPrice = 0.0;
        double capital = initialCapital;
        double positionSize = 0.0;
        double maxDrawdown = 0.0;
        double peakCapital = initialCapital;
        int tradeCount = 0;
        int winCount = 0;
        double totalGain = 0.0;
        double totalLoss = 0.0;
        double sumPnL = 0.0;
        double maxGain = Double.NEGATIVE_INFINITY;
        double maxLoss = Double.POSITIVE_INFINITY;
        int totalTradeBars = 0;
        int tradeStartIndex = 0;
        for (int i = 0; i < series.getBarCount(); i++) {
            double price = series.getBar(i).getClosePrice().doubleValue();
            if (!inPosition && entryRule.isSatisfied(i)) {
                // Entrée en position
                positionSize = capital * riskPerTrade;
                entryPrice = price;
                inPosition = true;
                tradeStartIndex = i;
            } else if (inPosition) {
                // Gestion du stop loss / take profit
                double stopLossPrice = entryPrice * (1 - stopLossPct);
                double takeProfitPrice = entryPrice * (1 + takeProfitPct);
                boolean stopLossHit = price <= stopLossPrice;
                boolean takeProfitHit = price >= takeProfitPrice;
                boolean exitSignal = exitRule.isSatisfied(i);
                if (stopLossHit || takeProfitHit || exitSignal) {
                    double exitPrice = price;
                    if (stopLossHit) exitPrice = stopLossPrice;
                    if (takeProfitHit) exitPrice = takeProfitPrice;
                    double pnl = positionSize * ((exitPrice - entryPrice) / entryPrice);
                    capital += pnl;
                    tradeCount++;
                    sumPnL += pnl;
                    if (pnl > 0) {
                        winCount++;
                        totalGain += pnl;
                        if (pnl > maxGain) maxGain = pnl;
                    } else {
                        totalLoss += Math.abs(pnl);
                        if (pnl < maxLoss) maxLoss = pnl;
                    }
                    totalTradeBars += (i - tradeStartIndex + 1);
                    inPosition = false;
                    // Drawdown
                    if (capital > peakCapital) peakCapital = capital;
                    double drawdown = (peakCapital - capital) / peakCapital;
                    if (drawdown > maxDrawdown) maxDrawdown = drawdown;
                }
            }
        }
        // Si une position reste ouverte à la fin, on la clôture au dernier prix
        if (inPosition) {
            double price = series.getBar(series.getEndIndex()).getClosePrice().doubleValue();
            double pnl = positionSize * ((price - entryPrice) / entryPrice);
            capital += pnl;
            tradeCount++;
            sumPnL += pnl;
            if (pnl > 0) {
                winCount++;
                totalGain += pnl;
                if (pnl > maxGain) maxGain = pnl;
            } else {
                totalLoss += Math.abs(pnl);
                if (pnl < maxLoss) maxLoss = pnl;
            }
            totalTradeBars += (series.getEndIndex() - tradeStartIndex + 1);
        }
        double rendement = (capital / initialCapital) - 1.0;
        double winRate = tradeCount > 0 ? (double) winCount / tradeCount : 0.0;
        double avgPnL = tradeCount > 0 ? sumPnL / tradeCount : 0.0;
        double profitFactor = totalLoss > 0 ? totalGain / totalLoss : 0.0;
        double avgTradeBars = tradeCount > 0 ? (double) totalTradeBars / tradeCount : 0.0;
        double maxTradeGain = (maxGain == Double.NEGATIVE_INFINITY) ? 0.0 : maxGain;
        double maxTradeLoss = (maxLoss == Double.POSITIVE_INFINITY) ? 0.0 : maxLoss;
        RiskResult riskResult = RiskResult.builder()
                .rendement(rendement)
                .tradeCount(tradeCount)
                .winRate(winRate)
                .maxDrawdown(maxDrawdown)
                .avgPnL(avgPnL)
                .profitFactor(profitFactor)
                .avgTradeBars(avgTradeBars)
                .maxTradeGain(maxTradeGain)
                .maxTradeLoss(maxTradeLoss)
                .scoreSwingTrade(0)
                .build();
        double scoreSwingTrade = TradeUtils.calculerScoreSwingTrade(riskResult);
        riskResult.setScoreSwingTrade(scoreSwingTrade);
        return riskResult;
    }

    public RiskResult backtestStrategy(TradeStrategy strategy, BarSeries series) {
        RiskResult riskResult =  backtestStrategy(strategy, series, INITIAL_CAPITAL, RISK_PER_TRADE, STOP_LOSS_PCT, TAKE_PROFIL_PCT);
        riskResult.setScoreSwingTrade(TradeUtils.calculerScoreSwingTrade(riskResult));
        return riskResult;
    }

    public RiskResult backtestStrategy(TradeStrategy strategy, BarSeries series, double initialCapital, double riskPerTrade, double stopLossPct, double takeProfitPct) {
        Rule entryRule = strategy.getEntryRule(series);
        Rule exitRule = strategy.getExitRule(series);
        boolean inPosition = false;
        double entryPrice = 0.0;
        double capital = initialCapital;
        double positionSize = 0.0;
        double maxDrawdown = 0.0;
        double peakCapital = initialCapital;
        int tradeCount = 0;
        int winCount = 0;
        double totalGain = 0.0;
        double totalLoss = 0.0;
        double sumPnL = 0.0;
        double maxGain = Double.NEGATIVE_INFINITY;
        double maxLoss = Double.POSITIVE_INFINITY;
        int totalTradeBars = 0;
        int tradeStartIndex = 0;
        for (int i = 0; i < series.getBarCount(); i++) {
            double price = series.getBar(i).getClosePrice().doubleValue();
            if (!inPosition && entryRule.isSatisfied(i)) {
                // Entrée en position
                positionSize = capital * riskPerTrade;
                entryPrice = price;
                inPosition = true;
                tradeStartIndex = i;
            } else if (inPosition) {
                // Gestion du stop loss / take profit
                double stopLossPrice = entryPrice * (1 - stopLossPct);
                double takeProfitPrice = entryPrice * (1 + takeProfitPct);
                boolean stopLossHit = price <= stopLossPrice;
                boolean takeProfitHit = price >= takeProfitPrice;
                boolean exitSignal = exitRule.isSatisfied(i);
                if (stopLossHit || takeProfitHit || exitSignal) {
                    double exitPrice = price;
                    if (stopLossHit) exitPrice = stopLossPrice;
                    if (takeProfitHit) exitPrice = takeProfitPrice;
                    double pnl = positionSize * ((exitPrice - entryPrice) / entryPrice);
                    capital += pnl;
                    tradeCount++;
                    sumPnL += pnl;
                    if (pnl > 0) {
                        winCount++;
                        totalGain += pnl;
                        if (pnl > maxGain) maxGain = pnl;
                    } else {
                        totalLoss += Math.abs(pnl);
                        if (pnl < maxLoss) maxLoss = pnl;
                    }
                    totalTradeBars += (i - tradeStartIndex + 1);
                    inPosition = false;
                    // Drawdown
                    if (capital > peakCapital) peakCapital = capital;
                    double drawdown = (peakCapital - capital) / peakCapital;
                    if (drawdown > maxDrawdown) maxDrawdown = drawdown;
                }
            }
        }
        // Si une position reste ouverte à la fin, on la clôture au dernier prix
        if (inPosition) {
            double price = series.getBar(series.getEndIndex()).getClosePrice().doubleValue();
            double pnl = positionSize * ((price - entryPrice) / entryPrice);
            capital += pnl;
            tradeCount++;
            sumPnL += pnl;
            if (pnl > 0) {
                winCount++;
                totalGain += pnl;
                if (pnl > maxGain) maxGain = pnl;
            } else {
                totalLoss += Math.abs(pnl);
                if (pnl < maxLoss) maxLoss = pnl;
            }
            totalTradeBars += (series.getEndIndex() - tradeStartIndex + 1);
        }
        double rendement = (capital / initialCapital) - 1.0;
        double winRate = tradeCount > 0 ? (double) winCount / tradeCount : 0.0;
        double avgPnL = tradeCount > 0 ? sumPnL / tradeCount : 0.0;
        double profitFactor = totalLoss > 0 ? totalGain / totalLoss : 0.0;
        double avgTradeBars = tradeCount > 0 ? (double) totalTradeBars / tradeCount : 0.0;
        double maxTradeGain = (maxGain == Double.NEGATIVE_INFINITY) ? 0.0 : maxGain;
        double maxTradeLoss = (maxLoss == Double.POSITIVE_INFINITY) ? 0.0 : maxLoss;
        RiskResult optimResult = RiskResult.builder()
                .rendement(rendement)
                .tradeCount(tradeCount)
                .winRate(winRate)
                .maxDrawdown(maxDrawdown)
                .avgPnL(avgPnL)
                .profitFactor(profitFactor)
                .avgTradeBars(avgTradeBars)
                .maxTradeGain(maxTradeGain)
                .maxTradeLoss(maxTradeLoss)
                .scoreSwingTrade(0).build();
        return optimResult;
    }


    // Backtest pour BreakoutStrategy
    public RiskResult backtestBreakoutStrategy(BarSeries series, int lookbackPeriod) {
        BreakoutStrategy strategy = new BreakoutStrategy(lookbackPeriod);
        return backtestStrategyRisk(strategy, series);
    }

    // Backtest pour MacdStrategy
    public RiskResult backtestMacdStrategy(BarSeries series, int shortPeriod, int longPeriod, int signalPeriod) {
        MacdStrategy strategy = new MacdStrategy(shortPeriod, longPeriod, signalPeriod);
        return backtestStrategyRisk(strategy, series);
    }

    // Backtest pour MeanReversionStrategy
    public RiskResult backtestMeanReversionStrategy(BarSeries series, int smaPeriod, double threshold) {
        MeanReversionStrategy strategy = new MeanReversionStrategy(smaPeriod, threshold);
        return backtestStrategyRisk(strategy, series);
    }

    // Backtest pour RsiStrategy
    public RiskResult backtestRsiStrategy(BarSeries series, int rsiPeriod, double oversoldThreshold, double overboughtThreshold) {
        RsiStrategy strategy = new RsiStrategy(rsiPeriod, oversoldThreshold, overboughtThreshold);
        return backtestStrategyRisk(strategy, series);
    }

    // Backtest pour SmaCrossoverStrategy
    public RiskResult backtestSmaCrossoverStrategy(BarSeries series, int shortPeriod, int longPeriod) {
        SmaCrossoverStrategy strategy = new SmaCrossoverStrategy(shortPeriod, longPeriod);
        return backtestStrategyRisk(strategy, series);
    }

    // Backtest pour TrendFollowingStrategy
    public RiskResult backtestTrendFollowingStrategy(BarSeries series, int trendPeriod) {
        TrendFollowingStrategy strategy = new TrendFollowingStrategy(trendPeriod);
        return backtestStrategyRisk(strategy, series);
    }

    // Backtest pour ImprovedTrendFollowingStrategy
    public RiskResult backtestImprovedTrendFollowingStrategy(BarSeries series, int trendPeriod, int shortMaPeriod, int longMaPeriod, double breakoutThreshold, boolean useRsiFilter, int rsiPeriod) {
        ImprovedTrendFollowingStrategy strategy = new ImprovedTrendFollowingStrategy(trendPeriod, shortMaPeriod, longMaPeriod, breakoutThreshold, useRsiFilter, rsiPeriod);
        return backtestStrategyRisk(strategy, series);
    }

    // Backtest pour ImprovedTrendFollowingStrategy avec paramètres par défaut
    public RiskResult backtestImprovedTrendFollowingStrategy(BarSeries series, int trendPeriod) {
        ImprovedTrendFollowingStrategy strategy = new ImprovedTrendFollowingStrategy(trendPeriod);
        return backtestStrategyRisk(strategy, series);
    }

    /**
     * Optimisation des paramètres pour MacdStrategy
     */
    public MacdParams optimiseMacdParameters(BarSeries series, int shortMin, int shortMax, int longMin, int longMax, int signalMin, int signalMax) {
        double bestReturn = Double.NEGATIVE_INFINITY;
        int bestShort = shortMin, bestLong = longMin, bestSignal = signalMin;
        double earlyStopThreshold = 0.3; // Arrêt anticipé si rendement > 30%
        // Pas adaptatifs
        int shortStep = (shortMax - shortMin > 20) ? 4 : 2;
        int longStep = (longMax - longMin > 20) ? 4 : 2;
        int signalStep = (signalMax - signalMin > 20) ? 4 : 2;
        int shortCount = (shortMax - shortMin) / shortStep + 1;
        int longCount = (longMax - longMin) / longStep + 1;
        int signalCount = (signalMax - signalMin) / signalStep + 1;
        int totalCombinaisons = shortCount * longCount * signalCount;
        java.util.Random rand = new java.util.Random();
        int maxRandomTests = 80;
        boolean useRandomSearch = totalCombinaisons > TradeConstant.RANDO_COUNT;
        int tested = 0;
        if (useRandomSearch) {
            // Random Search
            for (int i = 0; i < maxRandomTests; i++) {
                int shortPeriod = shortMin + rand.nextInt(shortCount) * shortStep;
                int longPeriod = longMin + rand.nextInt(longCount) * longStep;
                int signalPeriod = signalMin + rand.nextInt(signalCount) * signalStep;
                RiskResult result = backtestMacdStrategy(series, shortPeriod, longPeriod, signalPeriod);
                tested++;
                if (result.rendement > bestReturn) {
                    bestReturn = result.rendement;
                    bestShort = shortPeriod;
                    bestLong = longPeriod;
                    bestSignal = signalPeriod;
                }
                if (bestReturn > earlyStopThreshold) break;
            }
        } else {
            // Recherche exhaustive mais avec pas adaptatifs
            for (int shortPeriod = shortMin; shortPeriod <= shortMax; shortPeriod += shortStep) {
                for (int longPeriod = longMin; longPeriod <= longMax; longPeriod += longStep) {
                    for (int signalPeriod = signalMin; signalPeriod <= signalMax; signalPeriod += signalStep) {
                        RiskResult result = backtestMacdStrategy(series, shortPeriod, longPeriod, signalPeriod);
                        tested++;
                        if (result.rendement > bestReturn) {
                            bestReturn = result.rendement;
                            bestShort = shortPeriod;
                            bestLong = longPeriod;
                            bestSignal = signalPeriod;
                        }
                        if (bestReturn > earlyStopThreshold) break;
                    }
                    if (bestReturn > earlyStopThreshold) break;
                }
                if (bestReturn > earlyStopThreshold) break;
            }
        }
        // Optionnel : log du nombre de tests
        // System.out.println("Paramètres testés: " + tested + ", randomSearch: " + useRandomSearch);
        return new MacdParams(bestShort, bestLong, bestSignal, bestReturn);
    }

    /**
     * Optimisation des paramètres pour BreakoutStrategy
     */
    public BreakoutParams optimiseBreakoutParameters(BarSeries series, int lookbackMin, int lookbackMax) {
        double bestReturn = Double.NEGATIVE_INFINITY;
        int bestLookback = lookbackMin;
        double earlyStopThreshold = 0.3; // Arrêt anticipé si rendement > 30%
        int lookbackStep = (lookbackMax - lookbackMin > 20) ? 4 : 2;
        int totalCombinaisons = ((lookbackMax - lookbackMin) / lookbackStep) + 1;
        java.util.Random rand = new java.util.Random();
        int maxRandomTests = 50;
        boolean useRandomSearch = totalCombinaisons > TradeConstant.RANDO_COUNT;
        int tested = 0;
        if (useRandomSearch) {
            // Random Search
            for (int i = 0; i < maxRandomTests; i++) {
                int lookback = lookbackMin + rand.nextInt((lookbackMax - lookbackMin) / lookbackStep + 1) * lookbackStep;
                RiskResult result = backtestBreakoutStrategy(series, lookback);
                tested++;
                if (result.rendement > bestReturn) {
                    bestReturn = result.rendement;
                    bestLookback = lookback;
                }
                if (bestReturn > earlyStopThreshold) break;
            }
        } else {
            // Recherche exhaustive mais avec pas adaptatif
            for (int lookback = lookbackMin; lookback <= lookbackMax; lookback += lookbackStep) {
                RiskResult result = backtestBreakoutStrategy(series, lookback);
                tested++;
                if (result.rendement > bestReturn) {
                    bestReturn = result.rendement;
                    bestLookback = lookback;
                }
                if (bestReturn > earlyStopThreshold) break;
            }
        }
        // Optionnel : log du nombre de tests
        // System.out.println("Paramètres testés: " + tested + ", randomSearch: " + useRandomSearch);
        return new BreakoutParams(bestLookback, bestReturn);
    }

    /**
     * Optimisation des paramètres pour MeanReversionStrategy
     */
    public MeanReversionParams optimiseMeanReversionParameters(BarSeries series, int smaMin, int smaMax, double thresholdMin, double thresholdMax, double thresholdStep) {
        double bestReturn = Double.NEGATIVE_INFINITY;
        int bestSma = smaMin;
        double bestThreshold = thresholdMin;
        double earlyStopThreshold = 0.3; // Arrêt anticipé si rendement > 30%
        // Pas adaptatifs
        int smaStep = (smaMax - smaMin > 20) ? 4 : 2;
        double thresholdStepAdapt = (thresholdMax - thresholdMin > 20 * thresholdStep) ? thresholdStep * 2 : thresholdStep;
        int smaCount = (smaMax - smaMin) / smaStep + 1;
        int thresholdCount = (int) ((thresholdMax - thresholdMin) / thresholdStepAdapt) + 1;
        int totalCombinaisons = smaCount * thresholdCount;
        java.util.Random rand = new java.util.Random();
        int maxRandomTests = 80;
        boolean useRandomSearch = totalCombinaisons > TradeConstant.RANDO_COUNT;
        int tested = 0;
        if (useRandomSearch) {
            // Random Search
            for (int i = 0; i < maxRandomTests; i++) {
                int sma = smaMin + rand.nextInt(smaCount) * smaStep;
                double threshold = thresholdMin + rand.nextInt(thresholdCount) * thresholdStepAdapt;
                RiskResult result = backtestMeanReversionStrategy(series, sma, threshold);
                tested++;
                if (result.rendement > bestReturn) {
                    bestReturn = result.rendement;
                    bestSma = sma;
                    bestThreshold = threshold;
                }
                if (bestReturn > earlyStopThreshold) break;
            }
        } else {
            // Recherche exhaustive mais avec pas adaptatifs
            for (int sma = smaMin; sma <= smaMax; sma += smaStep) {
                for (double threshold = thresholdMin; threshold <= thresholdMax; threshold += thresholdStepAdapt) {
                    RiskResult result = backtestMeanReversionStrategy(series, sma, threshold);
                    tested++;
                    if (result.rendement > bestReturn) {
                        bestReturn = result.rendement;
                        bestSma = sma;
                        bestThreshold = threshold;
                    }
                    if (bestReturn > earlyStopThreshold) break;
                }
                if (bestReturn > earlyStopThreshold) break;
            }
        }
        // Optionnel : log du nombre de tests
        // System.out.println("Paramètres testés: " + tested + ", randomSearch: " + useRandomSearch);
        return new MeanReversionParams(bestSma, bestThreshold, bestReturn);
    }

    /**
     * Optimisation des paramètres pour RsiStrategy (version optimisée)
     */
    public RsiParams optimiseRsiParameters(BarSeries series, int rsiMin, int rsiMax, double oversoldMin, double oversoldMax, double oversoldStep, double overboughtMin, double overboughtMax, double overboughtStep) {
        double bestReturn = Double.NEGATIVE_INFINITY;
        int bestRsi = rsiMin;
        double bestOversold = oversoldMin;
        double bestOverbought = overboughtMin;
        double earlyStopThreshold = 0.3; // Arrêt anticipé si rendement > 30%
        // Pas adaptatifs
        int rsiStep = (rsiMax - rsiMin > 20) ? 4 : 2;
        double oversoldStepAdapt = (oversoldMax - oversoldMin > 20 * oversoldStep) ? oversoldStep * 2 : oversoldStep;
        double overboughtStepAdapt = (overboughtMax - overboughtMin > 20 * overboughtStep) ? overboughtStep * 2 : overboughtStep;
        // Calcul du nombre total de combinaisons
        int rsiCount = (rsiMax - rsiMin) / rsiStep + 1;
        int oversoldCount = (int) ((oversoldMax - oversoldMin) / oversoldStepAdapt) + 1;
        int overboughtCount = (int) ((overboughtMax - overboughtMin) / overboughtStepAdapt) + 1;
        int totalCombinaisons = rsiCount * oversoldCount * overboughtCount;
        java.util.Random rand = new java.util.Random();
        int maxRandomTests = 80;
        boolean useRandomSearch = totalCombinaisons > TradeConstant.RANDO_COUNT;
        int tested = 0;
        if (useRandomSearch) {
            // Random Search
            for (int i = 0; i < maxRandomTests; i++) {
                int rsi = rsiMin + rand.nextInt(rsiCount) * rsiStep;
                double oversold = oversoldMin + rand.nextInt(oversoldCount) * oversoldStepAdapt;
                double overbought = overboughtMin + rand.nextInt(overboughtCount) * overboughtStepAdapt;
                RiskResult result = backtestRsiStrategy(series, rsi, oversold, overbought);
                tested++;
                if (result.rendement > bestReturn) {
                    bestReturn = result.rendement;
                    bestRsi = rsi;
                    bestOversold = oversold;
                    bestOverbought = overbought;
                }
                if (bestReturn > earlyStopThreshold) break;
            }
        } else {
            // Recherche exhaustive mais avec pas adaptatifs
            for (int rsi = rsiMin; rsi <= rsiMax; rsi += rsiStep) {
                for (double oversold = oversoldMin; oversold <= oversoldMax; oversold += oversoldStepAdapt) {
                    for (double overbought = overboughtMin; overbought <= overboughtMax; overbought += overboughtStepAdapt) {
                        RiskResult result = backtestRsiStrategy(series, rsi, oversold, overbought);
                        tested++;
                        if (result.rendement > bestReturn) {
                            bestReturn = result.rendement;
                            bestRsi = rsi;
                            bestOversold = oversold;
                            bestOverbought = overbought;
                        }
                        if (bestReturn > earlyStopThreshold) break;
                    }
                    if (bestReturn > earlyStopThreshold) break;
                }
                if (bestReturn > earlyStopThreshold) break;
            }
        }
        // Optionnel : log du nombre de tests
        // System.out.println("Paramètres testés: " + tested + ", randomSearch: " + useRandomSearch);
        return new RsiParams(bestRsi, bestOversold, bestOverbought, bestReturn);
    }

    /**
     * Optimisation des paramètres pour SmaCrossoverStrategy
     */
    public SmaCrossoverParams optimiseSmaCrossoverParameters(BarSeries series, int shortMin, int shortMax, int longMin, int longMax) {
        double bestReturn = Double.NEGATIVE_INFINITY;
        int bestShort = shortMin;
        int bestLong = longMin;
        double earlyStopThreshold = 0.3; // Arrêt anticipé si rendement > 30%
        // Pas adaptatifs
        int shortStep = (shortMax - shortMin > 20) ? 4 : 2;
        int longStep = (longMax - longMin > 20) ? 4 : 2;
        int totalCombinaisons = 0;
        for (int shortPeriod = shortMin; shortPeriod <= shortMax; shortPeriod += shortStep) {
            for (int longPeriod = longMin; longPeriod <= longMax; longPeriod += longStep) {
                totalCombinaisons++;
            }
        }
        java.util.Random rand = new java.util.Random();
        int maxRandomTests = 100;
        boolean useRandomSearch = totalCombinaisons > TradeConstant.RANDO_COUNT;
        int tested = 0;
        if (useRandomSearch) {
            // Random Search
            for (int i = 0; i < maxRandomTests; i++) {
                int shortPeriod = shortMin + rand.nextInt((shortMax - shortMin) / shortStep + 1) * shortStep;
                int longPeriod = longMin + rand.nextInt((longMax - longMin) / longStep + 1) * longStep;
                RiskResult result = backtestSmaCrossoverStrategy(series, shortPeriod, longPeriod);
                tested++;
                if (result.rendement > bestReturn) {
                    bestReturn = result.rendement;
                    bestShort = shortPeriod;
                    bestLong = longPeriod;
                }
                if (bestReturn > earlyStopThreshold) break;
            }
        } else {
            // Recherche exhaustive mais avec pas adaptatifs
            for (int shortPeriod = shortMin; shortPeriod <= shortMax; shortPeriod += shortStep) {
                for (int longPeriod = longMin; longPeriod <= longMax; longPeriod += longStep) {
                    RiskResult result = backtestSmaCrossoverStrategy(series, shortPeriod, longPeriod);
                    tested++;
                    if (result.rendement > bestReturn) {
                        bestReturn = result.rendement;
                        bestShort = shortPeriod;
                        bestLong = longPeriod;
                    }
                    if (bestReturn > earlyStopThreshold) break;
                }
                if (bestReturn > earlyStopThreshold) break;
            }
        }
        // Optionnel : log du nombre de tests
        // System.out.println("Paramètres testés: " + tested + ", randomSearch: " + useRandomSearch);
        return new SmaCrossoverParams(bestShort, bestLong, bestReturn);
    }

    /**
     * Optimisation des paramètres pour TrendFollowingStrategy
     */
    public TrendFollowingParams optimiseTrendFollowingParameters(BarSeries series, int trendMin, int trendMax) {
        double bestReturn = Double.NEGATIVE_INFINITY;
        int bestTrend = trendMin;
        for (int trendPeriod = trendMin; trendPeriod <= trendMax; trendPeriod++) {
            RiskResult result = backtestTrendFollowingStrategy(series, trendPeriod);
            if (result.rendement > bestReturn) {
                bestReturn = result.rendement;
                bestTrend = trendPeriod;
            }
        }
        return new TrendFollowingParams(bestTrend, bestReturn);
    }

    /**
     * Optimisation des paramètres pour ImprovedTrendFollowingStrategy
     */
    public ImprovedTrendFollowingParams optimiseImprovedTrendFollowingParameters(BarSeries series,
                                                                                 int trendMin, int trendMax, int shortMaMin, int shortMaMax, int longMaMin, int longMaMax,
                                                                                 double thresholdMin, double thresholdMax, double thresholdStep) {
        return this.optimiseImprovedTrendFollowingParameters(series, trendMin, trendMax, shortMaMin, shortMaMax, longMaMin, longMaMax, thresholdMin, thresholdMax, thresholdStep, null);
    }
    public ImprovedTrendFollowingParams optimiseImprovedTrendFollowingParameters(BarSeries series,
            int trendMin, int trendMax, int shortMaMin, int shortMaMax, int longMaMin, int longMaMax,
            double thresholdMin, double thresholdMax, double thresholdStep, Integer maxCombos) {
        double bestReturn = Double.NEGATIVE_INFINITY;
        int bestTrend = trendMin;
        int bestShortMa = shortMaMin;
        int bestLongMa = longMaMin;
        double bestThreshold = thresholdMin;
        boolean bestUseRsi = true;
        int bestRsiPeriod = 14;
        double earlyStopThreshold = 0.3; // Arrêt anticipé si rendement > 30%
        int rsiPeriod = 14;

        // Calcul du nombre total de combinaisons
        int trendStep = (trendMax - trendMin > 20) ? 4 : 2;
        int shortMaStep = (shortMaMax - shortMaMin > 20) ? 4 : 2;
        int longMaStep = (longMaMax - longMaMin > 20) ? 6 : 3;
        int thresholdCount = (int) Math.ceil((thresholdMax - thresholdMin) / thresholdStep) + 1;
        int totalCombinaisons = 0;
        for (int trendPeriod = trendMin; trendPeriod <= trendMax; trendPeriod += trendStep) {
            for (int shortMa = shortMaMin; shortMa <= shortMaMax; shortMa += shortMaStep) {
                for (int longMa = longMaMin; longMa <= longMaMax; longMa += longMaStep) {
                    if (shortMa >= longMa) continue;
                    totalCombinaisons += 2 * thresholdCount; // 2 pour useRsi true/false
                }
            }
        }

        java.util.Random rand = new java.util.Random();

        int maxRandomTests = Math.min(totalCombinaisons, Math.max(TradeConstant.RANDO_COUNT, ((totalCombinaisons > TradeConstant.RANDO_COUNT * 10) ? (totalCombinaisons / 4) : TradeConstant.RANDO_COUNT)));
        boolean useRandomSearch = (totalCombinaisons > TradeConstant.RANDO_COUNT);
        if(maxCombos != null){
            useRandomSearch = (totalCombinaisons > maxCombos);
            maxRandomTests = 1000;
        }
        int tested = 0;
        if (useRandomSearch) {
            System.out.println("[Optimisation] ImprovedTrendFollowing: totalCombinaisons=" + totalCombinaisons + ", randomTests=" + maxRandomTests + " (" + (100.0 * maxRandomTests / totalCombinaisons) + "%)");
            // Random Search
            for (int i = 0; i < maxRandomTests; i++) {
                int trendPeriod = trendMin + rand.nextInt((trendMax - trendMin) / trendStep + 1) * trendStep;
                int shortMa = shortMaMin + rand.nextInt((shortMaMax - shortMaMin) / shortMaStep + 1) * shortMaStep;
                int longMa = longMaMin + rand.nextInt((longMaMax - longMaMin) / longMaStep + 1) * longMaStep;
                if (shortMa >= longMa) continue;
                double threshold = thresholdMin + rand.nextInt(thresholdCount) * thresholdStep;
                boolean useRsi = rand.nextBoolean();
                RiskResult result = backtestImprovedTrendFollowingStrategy(series, trendPeriod, shortMa, longMa, threshold, useRsi, rsiPeriod);
                tested++;
                if (result.rendement > bestReturn) {
                    bestReturn = result.rendement;
                    bestTrend = trendPeriod;
                    bestShortMa = shortMa;
                    bestLongMa = longMa;
                    bestThreshold = threshold;
                    bestUseRsi = useRsi;
                }
                if (bestReturn > earlyStopThreshold) break;
            }
        } else {
            // Recherche exhaustive mais avec pas adaptatifs
            for (int trendPeriod = trendMin; trendPeriod <= trendMax; trendPeriod += trendStep) {
                for (int shortMa = shortMaMin; shortMa <= shortMaMax; shortMa += shortMaStep) {
                    for (int longMa = longMaMin; longMa <= longMaMax; longMa += longMaStep) {
                        if (shortMa >= longMa) continue;
                        for (double threshold = thresholdMin; threshold <= thresholdMax; threshold += thresholdStep) {
                            for (boolean useRsi : new boolean[]{true, false}) {
                                RiskResult result = backtestImprovedTrendFollowingStrategy(series, trendPeriod, shortMa, longMa, threshold, useRsi, rsiPeriod);
                                tested++;
                                if (result.rendement > bestReturn) {
                                    bestReturn = result.rendement;
                                    bestTrend = trendPeriod;
                                    bestShortMa = shortMa;
                                    bestLongMa = longMa;
                                    bestThreshold = threshold;
                                    bestUseRsi = useRsi;
                                }
                                if (bestReturn > earlyStopThreshold) break;
                            }
                            if (bestReturn > earlyStopThreshold) break;
                        }
                        if (bestReturn > earlyStopThreshold) break;
                    }
                    if (bestReturn > earlyStopThreshold) break;
                }
                if (bestReturn > earlyStopThreshold) break;
            }
        }
        // Optionnel : log du nombre de tests
        // System.out.println("Paramètres testés: " + tested + ", randomSearch: " + useRandomSearch);
        return new ImprovedTrendFollowingParams(bestTrend, bestShortMa, bestLongMa, bestThreshold, bestUseRsi, rsiPeriod, bestReturn);
    }

    /**
     * Interface fonctionnelle pour l'optimisation de paramètres
     */
    @FunctionalInterface
    public interface Optimizer {
        Object optimise(BarSeries series);
    }

    /**
     * Interface fonctionnelle pour le backtest avec paramètres optimisés
     */
    @FunctionalInterface
    public interface ParamBacktest {
        RiskResult backtest(BarSeries series, Object params);
    }

    /**
     * Rolling Window Backtest
     * Optimise sur une fenêtre, teste sur la suivante, répète en glissant la fenêtre
     */
    public static class RollingWindowResult {
        public final int startOptIdx;
        public final int endOptIdx;
        public final int startTestIdx;
        public final int endTestIdx;
        public final Object params;
        public final RiskResult result;

        public RollingWindowResult(int startOptIdx, int endOptIdx, int startTestIdx, int endTestIdx, Object params, RiskResult result) {
            this.startOptIdx = startOptIdx;
            this.endOptIdx = endOptIdx;
            this.startTestIdx = startTestIdx;
            this.endTestIdx = endTestIdx;
            this.params = params;
            this.result = result;
        }
    }

    /*
    La série de bougies (BarSeries)
    La taille de la fenêtre d’optimisation
    La taille de la fenêtre de test (validation)
    Les plages de paramètres à optimiser
    Le type de stratégie à utiliser (via une interface ou un enum, ou simplement en passant la fonction d’optimisation et de backtest en paramètre)
     */
    public java.util.List<RollingWindowResult> runRollingWindowBacktest(
            BarSeries series,
            int windowOptSize,
            int windowTestSize,
            int stepSize,
            Optimizer optimizer,
            ParamBacktest backtestFunc
    ) {
        java.util.List<RollingWindowResult> results = new java.util.ArrayList<>();
        int totalBars = series.getBarCount();
        for (int startOpt = 0; startOpt + windowOptSize + windowTestSize <= totalBars; startOpt += stepSize) {
            int endOpt = startOpt + windowOptSize - 1;
            int startTest = endOpt + 1;
            int endTest = startTest + windowTestSize - 1;
            BarSeries optSeries = series.getSubSeries(startOpt, endOpt + 1);
            BarSeries testSeries = series.getSubSeries(startTest, endTest + 1);
            Object params = optimizer.optimise(optSeries);
            RiskResult result = backtestFunc.backtest(testSeries, params);
            results.add(new RollingWindowResult(startOpt, endOpt, startTest, endTest, params, result));
        }
        return results;
    }

    /**
     * Walk-Forward Analysis
     * Optimise sur une fenêtre, teste sur la suivante, avance la fenêtre et répète
     */
    public static class WalkForwardResult {
        public final int startOptIdx;
        public final int endOptIdx;
        public final int startTestIdx;
        public final int endTestIdx;
        public final Object params;
        public final RiskResult result;

        public WalkForwardResult(int startOptIdx, int endOptIdx, int startTestIdx, int endTestIdx, Object params, RiskResult result) {
            this.startOptIdx = startOptIdx;
            this.endOptIdx = endOptIdx;
            this.startTestIdx = startTestIdx;
            this.endTestIdx = endTestIdx;
            this.params = params;
            this.result = result;
        }
    }

    /*
    La série de bougies (BarSeries)
    La taille de la fenêtre d’optimisation
    La taille de la fenêtre de test (validation)
    Les plages de paramètres à optimiser
    Le type de stratégie à utiliser (via une interface ou un enum, ou simplement en passant la fonction d’optimisation et de backtest en paramètre)
     */
    public java.util.List<WalkForwardResult> runWalkForwardBacktest(
            BarSeries series,
            int windowOptSize,
            int windowTestSize,
            Optimizer optimizer,
            ParamBacktest backtestFunc
    ) {
        java.util.List<WalkForwardResult> results = new java.util.ArrayList<>();
        int totalBars = series.getBarCount();
        int startOpt = 0;
        while (startOpt + windowOptSize + windowTestSize <= totalBars) {
            int endOpt = startOpt + windowOptSize - 1;
            int startTest = endOpt + 1;
            int endTest = startTest + windowTestSize - 1;
            BarSeries optSeries = series.getSubSeries(startOpt, endOpt + 1);
            BarSeries testSeries = series.getSubSeries(startTest, endTest + 1);
            Object params = optimizer.optimise(optSeries);
            RiskResult result = backtestFunc.backtest(testSeries, params);
            results.add(new WalkForwardResult(startOpt, endOpt, startTest, endTest, params, result));
            startOpt = startTest; // Avance la fenêtre
        }
        return results;
    }

    // Rolling Window et Walk-Forward pour MACD
    public java.util.List<RollingWindowResult> runRollingWindowBacktestMacd(
            BarSeries series,
            int windowOptSize,
            int windowTestSize,
            int stepSize,
            int shortMin, int shortMax,
            int longMin, int longMax,
            int signalMin, int signalMax
    ) {
        Optimizer optimizer = (BarSeries optSeries) -> optimiseMacdParameters(optSeries, shortMin, shortMax, longMin, longMax, signalMin, signalMax);
        ParamBacktest backtestFunc = (BarSeries testSeries, Object params) -> {
            MacdParams p = (MacdParams) params;
            return backtestMacdStrategy(testSeries, p.shortPeriod, p.longPeriod, p.signalPeriod);
        };
        return runRollingWindowBacktest(series, windowOptSize, windowTestSize, stepSize, optimizer, backtestFunc);
    }

    public java.util.List<WalkForwardResult> runWalkForwardBacktestMacd(
            BarSeries series,
            int windowOptSize,
            int windowTestSize,
            int shortMin, int shortMax,
            int longMin, int longMax,
            int signalMin, int signalMax
    ) {
        Optimizer optimizer = (BarSeries optSeries) -> optimiseMacdParameters(optSeries, shortMin, shortMax, longMin, longMax, signalMin, signalMax);
        ParamBacktest backtestFunc = (BarSeries testSeries, Object params) -> {
            MacdParams p = (MacdParams) params;
            return backtestMacdStrategy(testSeries, p.shortPeriod, p.longPeriod, p.signalPeriod);
        };
        return runWalkForwardBacktest(series, windowOptSize, windowTestSize, optimizer, backtestFunc);
    }

    // Rolling Window et Walk-Forward pour Breakout
    public java.util.List<RollingWindowResult> runRollingWindowBacktestBreakout(
            BarSeries series,
            int windowOptSize,
            int windowTestSize,
            int stepSize,
            int lookbackMin, int lookbackMax
    ) {
        Optimizer optimizer = (BarSeries optSeries) -> optimiseBreakoutParameters(optSeries, lookbackMin, lookbackMax);
        ParamBacktest backtestFunc = (BarSeries testSeries, Object params) -> {
            BreakoutParams p = (BreakoutParams) params;
            return backtestBreakoutStrategy(testSeries, p.lookbackPeriod);
        };
        return runRollingWindowBacktest(series, windowOptSize, windowTestSize, stepSize, optimizer, backtestFunc);
    }

    public java.util.List<WalkForwardResult> runWalkForwardBacktestBreakout(
            BarSeries series,
            int windowOptSize,
            int windowTestSize,
            int lookbackMin, int lookbackMax
    ) {
        Optimizer optimizer = (BarSeries optSeries) -> optimiseBreakoutParameters(optSeries, lookbackMin, lookbackMax);
        ParamBacktest backtestFunc = (BarSeries testSeries, Object params) -> {
            BreakoutParams p = (BreakoutParams) params;
            return backtestBreakoutStrategy(testSeries, p.lookbackPeriod);
        };
        return runWalkForwardBacktest(series, windowOptSize, windowTestSize, optimizer, backtestFunc);
    }

    // Rolling Window et Walk-Forward pour MeanReversion
    public java.util.List<RollingWindowResult> runRollingWindowBacktestMeanReversion(
            BarSeries series,
            int windowOptSize,
            int windowTestSize,
            int stepSize,
            int smaMin, int smaMax,
            double thresholdMin, double thresholdMax, double thresholdStep
    ) {
        Optimizer optimizer = (BarSeries optSeries) -> optimiseMeanReversionParameters(optSeries, smaMin, smaMax, thresholdMin, thresholdMax, thresholdStep);
        ParamBacktest backtestFunc = (BarSeries testSeries, Object params) -> {
            MeanReversionParams p = (MeanReversionParams) params;
            return backtestMeanReversionStrategy(testSeries, p.smaPeriod, p.threshold);
        };
        return runRollingWindowBacktest(series, windowOptSize, windowTestSize, stepSize, optimizer, backtestFunc);
    }

    public java.util.List<WalkForwardResult> runWalkForwardBacktestMeanReversion(
            BarSeries series,
            int windowOptSize,
            int windowTestSize,
            int smaMin, int smaMax,
            double thresholdMin, double thresholdMax, double thresholdStep
    ) {
        Optimizer optimizer = (BarSeries optSeries) -> optimiseMeanReversionParameters(optSeries, smaMin, smaMax, thresholdMin, thresholdMax, thresholdStep);
        ParamBacktest backtestFunc = (BarSeries testSeries, Object params) -> {
            MeanReversionParams p = (MeanReversionParams) params;
            return backtestMeanReversionStrategy(testSeries, p.smaPeriod, p.threshold);
        };
        return runWalkForwardBacktest(series, windowOptSize, windowTestSize, optimizer, backtestFunc);
    }

    // Rolling Window et Walk-Forward pour RSI
    public java.util.List<RollingWindowResult> runRollingWindowBacktestRsi(
            BarSeries series,
            int windowOptSize,
            int windowTestSize,
            int stepSize,
            int rsiMin, int rsiMax,
            double oversoldMin, double oversoldMax, double oversoldStep,
            double overboughtMin, double overboughtMax, double overboughtStep
    ) {
        Optimizer optimizer = (BarSeries optSeries) -> optimiseRsiParameters(optSeries, rsiMin, rsiMax, oversoldMin, oversoldMax, oversoldStep, overboughtMin, overboughtMax, overboughtStep);
        ParamBacktest backtestFunc = (BarSeries testSeries, Object params) -> {
            RsiParams p = (RsiParams) params;
            return backtestRsiStrategy(testSeries, p.rsiPeriod, p.oversold, p.overbought);
        };
        return runRollingWindowBacktest(series, windowOptSize, windowTestSize, stepSize, optimizer, backtestFunc);
    }

    public java.util.List<WalkForwardResult> runWalkForwardBacktestRsi(
            BarSeries series,
            int windowOptSize,
            int windowTestSize,
            int rsiMin, int rsiMax,
            double oversoldMin, double oversoldMax, double oversoldStep,
            double overboughtMin, double overboughtMax, double overboughtStep
    ) {
        Optimizer optimizer = (BarSeries optSeries) -> optimiseRsiParameters(optSeries, rsiMin, rsiMax, oversoldMin, oversoldMax, oversoldStep, overboughtMin, overboughtMax, overboughtStep);
        ParamBacktest backtestFunc = (BarSeries testSeries, Object params) -> {
            RsiParams p = (RsiParams) params;
            return backtestRsiStrategy(testSeries, p.rsiPeriod, p.oversold, p.overbought);
        };
        return runWalkForwardBacktest(series, windowOptSize, windowTestSize, optimizer, backtestFunc);
    }

    // Rolling Window et Walk-Forward pour SMA Crossover
    public java.util.List<RollingWindowResult> runRollingWindowBacktestSmaCrossover(
            BarSeries series,
            int windowOptSize,
            int windowTestSize,
            int stepSize,
            int shortMin, int shortMax,
            int longMin, int longMax
    ) {
        Optimizer optimizer = (BarSeries optSeries) -> optimiseSmaCrossoverParameters(optSeries, shortMin, shortMax, longMin, longMax);
        ParamBacktest backtestFunc = (BarSeries testSeries, Object params) -> {
            SmaCrossoverParams p = (SmaCrossoverParams) params;
            return backtestSmaCrossoverStrategy(testSeries, p.shortPeriod, p.longPeriod);
        };
        return runRollingWindowBacktest(series, windowOptSize, windowTestSize, stepSize, optimizer, backtestFunc);
    }

    public java.util.List<WalkForwardResult> runWalkForwardBacktestSmaCrossover(
            BarSeries series,
            int windowOptSize,
            int windowTestSize,
            int shortMin, int shortMax,
            int longMin, int longMax
    ) {
        Optimizer optimizer = (BarSeries optSeries) -> optimiseSmaCrossoverParameters(optSeries, shortMin, shortMax, longMin, longMax);
        ParamBacktest backtestFunc = (BarSeries testSeries, Object params) -> {
            SmaCrossoverParams p = (SmaCrossoverParams) params;
            return backtestSmaCrossoverStrategy(testSeries, p.shortPeriod, p.longPeriod);
        };
        return runWalkForwardBacktest(series, windowOptSize, windowTestSize, optimizer, backtestFunc);
    }

    // Rolling Window et Walk-Forward pour ImprovedTrendFollowing
    public java.util.List<RollingWindowResult> runRollingWindowBacktestImprovedTrendFollowing(
            BarSeries series,
            int windowOptSize,
            int windowTestSize,
            int stepSize,
            int trendMin, int trendMax,
            int shortMaMin, int shortMaMax,
            int longMaMin, int longMaMax,
            double thresholdMin, double thresholdMax, double thresholdStep
    ) {
        Optimizer optimizer = (BarSeries optSeries) -> optimiseImprovedTrendFollowingParameters(optSeries,
            trendMin, trendMax, shortMaMin, shortMaMax, longMaMin, longMaMax, thresholdMin, thresholdMax, thresholdStep);
        ParamBacktest backtestFunc = (BarSeries testSeries, Object params) -> {
            ImprovedTrendFollowingParams p = (ImprovedTrendFollowingParams) params;
            return backtestImprovedTrendFollowingStrategy(testSeries, p.trendPeriod, p.shortMaPeriod, p.longMaPeriod, p.breakoutThreshold, p.useRsiFilter, p.rsiPeriod);
        };
        return runRollingWindowBacktest(series, windowOptSize, windowTestSize, stepSize, optimizer, backtestFunc);
    }

    public java.util.List<WalkForwardResult> runWalkForwardBacktestImprovedTrendFollowing(
            BarSeries series,
            int windowOptSize,
            int windowTestSize,
            int trendMin, int trendMax,
            int shortMaMin, int shortMaMax,
            int longMaMin, int longMaMax,
            double thresholdMin, double thresholdMax, double thresholdStep
    ) {
        Optimizer optimizer = (BarSeries optSeries) -> optimiseImprovedTrendFollowingParameters(optSeries,
            trendMin, trendMax, shortMaMin, shortMaMax, longMaMin, longMaMax, thresholdMin, thresholdMax, thresholdStep);
        ParamBacktest backtestFunc = (BarSeries testSeries, Object params) -> {
            ImprovedTrendFollowingParams p = (ImprovedTrendFollowingParams) params;
            return backtestImprovedTrendFollowingStrategy(testSeries, p.trendPeriod, p.shortMaPeriod, p.longMaPeriod, p.breakoutThreshold, p.useRsiFilter, p.rsiPeriod);
        };
        return runWalkForwardBacktest(series, windowOptSize, windowTestSize, optimizer, backtestFunc);
    }

    /**
     * Optimisation des paramètres pour toutes les combinaisons de stratégies IN/OUT
     * Teste plusieurs combinaisons de stratégies d'entrée et de sortie, optimise les paramètres et retourne la meilleure stratégie combinée
     */
    public BestInOutStrategy testAllCrossedStrategies(
            BarSeries series,
            int smaShortMin, int smaShortMax, int smaLongMin, int smaLongMax,
            int rsiMin, int rsiMax, double oversoldMin, double oversoldMax, double oversoldStep, double overboughtMin, double overboughtMax, double overboughtStep,
            int trendMin, int trendMax
    ) {
        double bestReturn = Double.NEGATIVE_INFINITY;
        BestInOutStrategy bestCombo = null;

        // SMA Crossover IN / RSI OUT
        SmaCrossoverParams smaParams = optimiseSmaCrossoverParameters(series, smaShortMin, smaShortMax, smaLongMin, smaLongMax);
        RsiParams rsiParams = optimiseRsiParameters(series, rsiMin, rsiMax, oversoldMin, oversoldMax, oversoldStep, overboughtMin, overboughtMax, overboughtStep);
        CombinedTradeStrategy smaRsiStrategy = new CombinedTradeStrategy(
                new SmaCrossoverStrategy(smaParams.shortPeriod, smaParams.longPeriod),
                new RsiStrategy(rsiParams.rsiPeriod, rsiParams.oversold, rsiParams.overbought)
        );
        RiskResult resultSmaRsi = backtestStrategyRisk(smaRsiStrategy, series);
        if (resultSmaRsi.rendement > bestReturn) {
            bestReturn = resultSmaRsi.rendement;
            bestCombo = new BestInOutStrategy("SMA Crossover", smaParams, "RSI", rsiParams, resultSmaRsi);
        }

        // RSI IN / TrendFollowing OUT
        TrendFollowingParams trendParams = optimiseTrendFollowingParameters(series, trendMin, trendMax);
        CombinedTradeStrategy rsiTrendStrategy = new CombinedTradeStrategy(
                new RsiStrategy(rsiParams.rsiPeriod, rsiParams.oversold, rsiParams.overbought),
                new TrendFollowingStrategy(trendParams.trendPeriod)
        );
        RiskResult resultRsiTrend = backtestStrategyRisk(rsiTrendStrategy, series);
        if (resultRsiTrend.rendement > bestReturn) {
            bestReturn = resultRsiTrend.rendement;
            bestCombo = new BestInOutStrategy("RSI", rsiParams, "TrendFollowing", trendParams, resultRsiTrend);
        }

        // TrendFollowing IN / SMA Crossover OUT
        CombinedTradeStrategy trendSmaStrategy = new CombinedTradeStrategy(
                new TrendFollowingStrategy(trendParams.trendPeriod),
                new SmaCrossoverStrategy(smaParams.shortPeriod, smaParams.longPeriod)
        );
        RiskResult resultTrendSma = backtestStrategyRisk(trendSmaStrategy, series);
        if (resultTrendSma.rendement > bestReturn) {
            bestReturn = resultTrendSma.rendement;
            bestCombo = new BestInOutStrategy("TrendFollowing", trendParams, "SMA Crossover", smaParams, resultTrendSma);
        }

        // On peut ajouter d'autres combinaisons ici selon les stratégies disponibles

        return bestCombo;
    }

    /**
     * Classe pour stocker tous les meilleurs paramètres de toutes les stratégies
     */
    public static class AllBestParams {
        public final ImprovedTrendFollowingParams improvedTrendFollowing;
        public final SmaCrossoverParams smaCrossover;
        public final RsiParams rsi;
        public final BreakoutParams breakout;
        public final MacdParams macd;
        public final MeanReversionParams meanReversion;

        // Performance ranking pour faciliter l'analyse
        public final java.util.Map<String, Double> performanceRanking;
        public final java.util.Map<String, RiskResult> detailedResults;

        public AllBestParams(ImprovedTrendFollowingParams improvedTrendFollowing,
                            SmaCrossoverParams smaCrossover,
                            RsiParams rsi,
                            BreakoutParams breakout,
                            MacdParams macd,
                            MeanReversionParams meanReversion,
                            java.util.Map<String, Double> performanceRanking,
                            java.util.Map<String, RiskResult> detailedResults) {
            this.improvedTrendFollowing = improvedTrendFollowing;
            this.smaCrossover = smaCrossover;
            this.rsi = rsi;
            this.breakout = breakout;
            this.macd = macd;
            this.meanReversion = meanReversion;
            this.performanceRanking = performanceRanking;
            this.detailedResults = detailedResults;
        }

        /**
         * Retourne les paramètres de la meilleure stratégie
         */
        public Object getBestStrategyParams() {
            String bestStrategy = performanceRanking.entrySet().stream()
                .max(java.util.Map.Entry.comparingByValue())
                .map(java.util.Map.Entry::getKey)
                .orElse("Mean Reversion");

            switch (bestStrategy) {
                case "Improved Trend": return improvedTrendFollowing;
                case "SMA Crossover": return smaCrossover;
                case "RSI": return rsi;
                case "Breakout": return breakout;
                case "MACD": return macd;
                case "Mean Reversion": return meanReversion;
                default: return meanReversion;
            }
        }

        /**
         * Retourne le nom de la meilleure stratégie
         */
        public String getBestStrategyName() {
            return performanceRanking.entrySet().stream()
                .max(java.util.Map.Entry.comparingByValue())
                .map(java.util.Map.Entry::getKey)
                .orElse("Mean Reversion");
        }

        /**
         * Retourne la performance de la meilleure stratégie
         */
        public double getBestPerformance() {
            return performanceRanking.values().stream()
                .mapToDouble(Double::doubleValue)
                .max()
                .orElse(0.0);
        }

        /**
         * Export en JSON pour sauvegarde/analyse
         */
        public String toJson() {
            com.google.gson.JsonObject obj = new com.google.gson.JsonObject();

            // Improved Trend Following
            com.google.gson.JsonObject itfObj = new com.google.gson.JsonObject();
            itfObj.addProperty("trendPeriod", improvedTrendFollowing.trendPeriod);
            itfObj.addProperty("shortMaPeriod", improvedTrendFollowing.shortMaPeriod);
            itfObj.addProperty("longMaPeriod", improvedTrendFollowing.longMaPeriod);
            itfObj.addProperty("breakoutThreshold", improvedTrendFollowing.breakoutThreshold);
            itfObj.addProperty("useRsiFilter", improvedTrendFollowing.useRsiFilter);
            itfObj.addProperty("rsiPeriod", improvedTrendFollowing.rsiPeriod);
            itfObj.addProperty("performance", improvedTrendFollowing.performance);
            obj.add("improvedTrendFollowing", itfObj);

            // SMA Crossover
            com.google.gson.JsonObject smaObj = new com.google.gson.JsonObject();
            smaObj.addProperty("shortPeriod", smaCrossover.shortPeriod);
            smaObj.addProperty("longPeriod", smaCrossover.longPeriod);
            smaObj.addProperty("performance", smaCrossover.performance);
            obj.add("smaCrossover", smaObj);

            // RSI
            com.google.gson.JsonObject rsiObj = new com.google.gson.JsonObject();
            rsiObj.addProperty("rsiPeriod", rsi.rsiPeriod);
            rsiObj.addProperty("oversold", rsi.oversold);
            rsiObj.addProperty("overbought", rsi.overbought);
            rsiObj.addProperty("performance", rsi.performance);
            obj.add("rsi", rsiObj);

            // Breakout
            com.google.gson.JsonObject breakoutObj = new com.google.gson.JsonObject();
            breakoutObj.addProperty("lookbackPeriod", breakout.lookbackPeriod);
            breakoutObj.addProperty("performance", breakout.performance);
            obj.add("breakout", breakoutObj);

            // MACD
            com.google.gson.JsonObject macdObj = new com.google.gson.JsonObject();
            macdObj.addProperty("shortPeriod", macd.shortPeriod);
            macdObj.addProperty("longPeriod", macd.longPeriod);
            macdObj.addProperty("signalPeriod", macd.signalPeriod);
            macdObj.addProperty("performance", macd.performance);
            obj.add("macd", macdObj);

            // Mean Reversion
            com.google.gson.JsonObject mrObj = new com.google.gson.JsonObject();
            mrObj.addProperty("smaPeriod", meanReversion.smaPeriod);
            mrObj.addProperty("threshold", meanReversion.threshold);
            mrObj.addProperty("performance", meanReversion.performance);
            obj.add("meanReversion", mrObj);

            // Performance ranking
            com.google.gson.JsonObject rankingObj = new com.google.gson.JsonObject();
            performanceRanking.forEach(rankingObj::addProperty);
            obj.add("performanceRanking", rankingObj);

            // Best strategy summary
            com.google.gson.JsonObject bestObj = new com.google.gson.JsonObject();
            bestObj.addProperty("strategyName", getBestStrategyName());
            bestObj.addProperty("performance", getBestPerformance());
            obj.add("bestStrategy", bestObj);

            return new com.google.gson.GsonBuilder().setPrettyPrinting().create().toJson(obj);
        }
    }

    // Classes de paramètres pour le retour des optimisations
    public static class MacdParams {
        public final int shortPeriod, longPeriod, signalPeriod;
        public final double performance;
        public MacdParams(int shortPeriod, int longPeriod, int signalPeriod, double performance) {
            this.shortPeriod = shortPeriod;
            this.longPeriod = longPeriod;
            this.signalPeriod = signalPeriod;
            this.performance = performance;
        }
    }

    public static class BreakoutParams {
        public final int lookbackPeriod;
        public final double performance;
        public BreakoutParams(int lookbackPeriod, double performance) {
            this.lookbackPeriod = lookbackPeriod;
            this.performance = performance;
        }
    }

    public static class MeanReversionParams {
        public final int smaPeriod;
        public final double threshold, performance;
        public MeanReversionParams(int smaPeriod, double threshold, double performance) {
            this.smaPeriod = smaPeriod;
            this.threshold = threshold;
            this.performance = performance;
        }
    }

    public static class RsiParams {
        public final int rsiPeriod;
        public final double oversold, overbought, performance;
        public RsiParams(int rsiPeriod, double oversold, double overbought, double performance) {
            this.rsiPeriod = rsiPeriod;
            this.oversold = oversold;
            this.overbought = overbought;
            this.performance = performance;
        }
        @Override
        public String toString() {
            return "RsiParams{" +
                    "rsiPeriod=" + rsiPeriod +
                    ", oversold=" + oversold +
                    ", overbought=" + overbought +
                    ", performance=" + performance +
                    '}';
        }
    }

    public static class SmaCrossoverParams {
        public final int shortPeriod, longPeriod;
        public final double performance;
        public SmaCrossoverParams(int shortPeriod, int longPeriod, double performance) {
            this.shortPeriod = shortPeriod;
            this.longPeriod = longPeriod;
            this.performance = performance;
        }
        @Override
        public String toString() {
            return "SmaCrossoverParams{" +
                    "shortPeriod=" + shortPeriod +
                    ", longPeriod=" + longPeriod +
                    ", performance=" + performance +
                    '}';
        }
    }

    public static class TrendFollowingParams {
        public final int trendPeriod;
        public final double performance;
        public TrendFollowingParams(int trendPeriod, double performance) {
            this.trendPeriod = trendPeriod;
            this.performance = performance;
        }
        @Override
        public String toString() {
            return "TrendFollowingParams{" +
                    "trendPeriod=" + trendPeriod +
                    ", performance=" + performance +
                    '}';
        }
    }

    public static class ImprovedTrendFollowingParams {
        public final int trendPeriod;
        public final int shortMaPeriod;
        public final int longMaPeriod;
        public final double breakoutThreshold;
        public final boolean useRsiFilter;
        public final int rsiPeriod;
        public final double performance;

        public ImprovedTrendFollowingParams(int trendPeriod, int shortMaPeriod, int longMaPeriod,
                                            double breakoutThreshold, boolean useRsiFilter, int rsiPeriod, double performance) {
            this.trendPeriod = trendPeriod;
            this.shortMaPeriod = shortMaPeriod;
            this.longMaPeriod = longMaPeriod;
            this.breakoutThreshold = breakoutThreshold;
            this.useRsiFilter = useRsiFilter;
            this.rsiPeriod = rsiPeriod;
            this.performance = performance;
        }
        @Override
        public String toString() {
            return "ImprovedTrendFollowingParams{" +
                    "trendPeriod=" + trendPeriod +
                    ", shortMaPeriod=" + shortMaPeriod +
                    ", longMaPeriod=" + longMaPeriod +
                    ", breakoutThreshold=" + breakoutThreshold +
                    ", useRsiFilter=" + useRsiFilter +
                    ", rsiPeriod=" + rsiPeriod +
                    ", performance=" + performance +
                    '}';
        }
    }

    /**
     * Stratégie combinée : permet d'utiliser une stratégie pour l'entrée et une autre pour la sortie
     */
    public static class CombinedTradeStrategy implements TradeStrategy {
        private final TradeStrategy entryStrategy;
        private final TradeStrategy exitStrategy;

        public CombinedTradeStrategy(TradeStrategy entryStrategy, TradeStrategy exitStrategy) {
            this.entryStrategy = entryStrategy;
            this.exitStrategy = exitStrategy;
        }

        @Override
        public org.ta4j.core.Rule getEntryRule(org.ta4j.core.BarSeries series) {
            return entryStrategy.getEntryRule(series);
        }

        @Override
        public org.ta4j.core.Rule getExitRule(org.ta4j.core.BarSeries series) {
            return exitStrategy.getExitRule(series);
        }

        @Override
        public String getName() {
            return "Combined(" + entryStrategy.getName() + " / " + exitStrategy.getName() + ")";
        }
    }

    /**
     * Affiche les résultats d'une liste de RollingWindowResult (SMA Crossover) dans la console
     */
    public static void printRollingWindowResultsSmaCrossover(List<RollingWindowResult> results) {
        for (RollingWindowResult res : results) {
            SmaCrossoverParams params = (SmaCrossoverParams) res.params;
            RiskResult metrics = res.result;
            System.out.println(
                "Fenêtre optimisation: " + res.startOptIdx + "-" + res.endOptIdx +
                ", fenêtre test: " + res.startTestIdx + "-" + res.endTestIdx +
                ", SMA params: short=" + params.shortPeriod + ", long=" + params.longPeriod +
                ", rendement test: " + metrics.rendement +
                ", drawdown: " + metrics.maxDrawdown +
                ", win rate: " + metrics.winRate +
                ", nb trades: " + metrics.tradeCount +
                ", profit factor: " + metrics.profitFactor
            );
        }
    }

    /**
     * Affiche les résultats d'une liste de RollingWindowResult (RSI) dans la console
     */
    public static void printRollingWindowResultsRsi(List<RollingWindowResult> results) {
        for (RollingWindowResult res : results) {
            RsiParams params = (RsiParams) res.params;
            RiskResult metrics = res.result;
            System.out.println(
                "Fenêtre optimisation: " + res.startOptIdx + "-" + res.endOptIdx +
                ", fenêtre test: " + res.startTestIdx + "-" + res.endTestIdx +
                ", RSI params: period=" + params.rsiPeriod + ", oversold=" + params.oversold + ", overbought=" + params.overbought +
                ", rendement test: " + metrics.rendement +
                ", drawdown: " + metrics.maxDrawdown +
                ", win rate: " + metrics.winRate +
                ", nb trades: " + metrics.tradeCount +
                ", profit factor: " + metrics.profitFactor
            );
        }
    }

    /**
     * Affiche les résultats d'une liste de RollingWindowResult (MACD) dans la console
     */
    public static void printRollingWindowResultsMacd(List<RollingWindowResult> results) {
        for (RollingWindowResult res : results) {
            MacdParams params = (MacdParams) res.params;
            RiskResult metrics = res.result;
            System.out.println(
                "Fenêtre optimisation: " + res.startOptIdx + "-" + res.endOptIdx +
                ", fenêtre test: " + res.startTestIdx + "-" + res.endTestIdx +
                ", MACD params: short=" + params.shortPeriod + ", long=" + params.longPeriod + ", signal=" + params.signalPeriod +
                ", rendement test: " + metrics.rendement +
                ", drawdown: " + metrics.maxDrawdown +
                ", win rate: " + metrics.winRate +
                ", nb trades: " + metrics.tradeCount +
                ", profit factor: " + metrics.profitFactor
            );
        }
    }

    /**
     * Affiche les résultats génériques d'une liste de RollingWindowResult dans la console
     */
    public static void printRollingWindowResultsGeneric(List<RollingWindowResult> results) {
        for (RollingWindowResult res : results) {
            RiskResult metrics = res.result;
            System.out.println(
                "Fenêtre optimisation: " + res.startOptIdx + "-" + res.endOptIdx +
                ", fenêtre test: " + res.startTestIdx + "-" + res.endTestIdx +
                ", params: " + res.params.toString() +
                ", rendement test: " + metrics.rendement +
                ", drawdown: " + metrics.maxDrawdown +
                ", win rate: " + metrics.winRate +
                ", nb trades: " + metrics.tradeCount +
                ", profit factor: " + metrics.profitFactor
            );
        }
    }

    /**
     * Structure pour stocker la meilleure combinaison IN/OUT et ses paramètres
     */
    public static class BestInOutStrategy {
        public final String entryName;
        public final Object entryParams;
        public final String exitName;
        public final Object exitParams;
        public final RiskResult result;

        public BestInOutStrategy(String entryName, Object entryParams, String exitName, Object exitParams, RiskResult result) {
            this.entryName = entryName;
            this.entryParams = entryParams;
            this.exitName = exitName;
            this.exitParams = exitParams;
            this.result = result;
        }
    }

    /**
     * Exporte les résultats d'une liste de WalkForwardResult en JSON
     * Nécessite la dépendance Gson (com.google.gson.Gson)
     */
    public static String exportWalkForwardResultsToJson(List<WalkForwardResult> results) {
        com.google.gson.JsonArray arr = new com.google.gson.JsonArray();
        for (WalkForwardResult res : results) {
            com.google.gson.JsonObject obj = new com.google.gson.JsonObject();
            obj.addProperty("startOptIdx", res.startOptIdx);
            obj.addProperty("endOptIdx", res.endOptIdx);
            obj.addProperty("startTestIdx", res.startTestIdx);
            obj.addProperty("endTestIdx", res.endTestIdx);

            // Gestion générique des paramètres selon le type
            com.google.gson.JsonObject paramObj = new com.google.gson.JsonObject();
            if (res.params instanceof TrendFollowingParams) {
                TrendFollowingParams params = (TrendFollowingParams) res.params;
                paramObj.addProperty("trendPeriod", params.trendPeriod);
                paramObj.addProperty("performance", params.performance);
            } else if (res.params instanceof SmaCrossoverParams) {
                SmaCrossoverParams params = (SmaCrossoverParams) res.params;
                paramObj.addProperty("shortPeriod", params.shortPeriod);
                paramObj.addProperty("longPeriod", params.longPeriod);
                paramObj.addProperty("performance", params.performance);
            } else if (res.params instanceof RsiParams) {
                RsiParams params = (RsiParams) res.params;
                paramObj.addProperty("rsiPeriod", params.rsiPeriod);
                paramObj.addProperty("oversold", params.oversold);
                paramObj.addProperty("overbought", params.overbought);
                paramObj.addProperty("performance", params.performance);
            } else if (res.params instanceof MacdParams) {
                MacdParams params = (MacdParams) res.params;
                paramObj.addProperty("shortPeriod", params.shortPeriod);
                paramObj.addProperty("longPeriod", params.longPeriod);
                paramObj.addProperty("signalPeriod", params.signalPeriod);
                paramObj.addProperty("performance", params.performance);
            } else {
                paramObj.addProperty("params", res.params.toString());
            }
            obj.add("params", paramObj);

            RiskResult metrics = res.result;
            com.google.gson.JsonObject resultObj = new com.google.gson.JsonObject();
            resultObj.addProperty("rendement", metrics.rendement);
            resultObj.addProperty("maxDrawdown", metrics.maxDrawdown);
            resultObj.addProperty("tradeCount", metrics.tradeCount);
            resultObj.addProperty("winRate", metrics.winRate);
            resultObj.addProperty("avgPnL", metrics.avgPnL);
            resultObj.addProperty("profitFactor", metrics.profitFactor);
            resultObj.addProperty("avgTradeBars", metrics.avgTradeBars);
            resultObj.addProperty("maxTradeGain", metrics.maxTradeGain);
            resultObj.addProperty("maxTradeLoss", metrics.maxTradeLoss);
            obj.add("result", resultObj);
            arr.add(obj);
        }
        return new com.google.gson.GsonBuilder().setPrettyPrinting().create().toJson(arr);
    }

    /**
     * Affiche les résultats d'une liste de WalkForwardResult (TrendFollowing) dans la console
     */
    public static void printWalkForwardResults(List<WalkForwardResult> results) {
        for (WalkForwardResult res : results) {
            TrendFollowingParams params = (TrendFollowingParams) res.params;
            RiskResult metrics = res.result;
            System.out.println(
                "Fenêtre optimisation: " + res.startOptIdx + "-" + res.endOptIdx +
                ", fenêtre test: " + res.startTestIdx + "-" + res.endTestIdx +
                ", Trend params: period=" + params.trendPeriod +
                ", rendement test: " + metrics.rendement +
                ", drawdown: " + metrics.maxDrawdown +
                ", win rate: " + metrics.winRate +
                ", nb trades: " + metrics.tradeCount +
                ", profit factor: " + metrics.profitFactor
            );
        }
    }

    /**
     * Affiche les résultats d'une liste de WalkForwardResult (SMA Crossover) dans la console
     */
    public static void printWalkForwardResultsSmaCrossover(List<WalkForwardResult> results) {
        for (WalkForwardResult res : results) {
            SmaCrossoverParams params = (SmaCrossoverParams) res.params;
            RiskResult metrics = res.result;
            System.out.println(
                "Fenêtre optimisation: " + res.startOptIdx + "-" + res.endOptIdx +
                ", fenêtre test: " + res.startTestIdx + "-" + res.endTestIdx +
                ", SMA params: short=" + params.shortPeriod + ", long=" + params.longPeriod +
                ", rendement test: " + metrics.rendement +
                ", drawdown: " + metrics.maxDrawdown +
                ", win rate: " + metrics.winRate +
                ", nb trades: " + metrics.tradeCount +
                ", profit factor: " + metrics.profitFactor
            );
        }
    }

    /**
     * Affiche les résultats d'une liste de WalkForwardResult (RSI) dans la console
     */
    public static void printWalkForwardResultsRsi(List<WalkForwardResult> results) {
        for (WalkForwardResult res : results) {
            RsiParams params = (RsiParams) res.params;
            RiskResult metrics = res.result;
            System.out.println(
                "Fenêtre optimisation: " + res.startOptIdx + "-" + res.endOptIdx +
                ", fenêtre test: " + res.startTestIdx + "-" + res.endTestIdx +
                ", RSI params: period=" + params.rsiPeriod + ", oversold=" + params.oversold + ", overbought=" + params.overbought +
                ", rendement test: " + metrics.rendement +
                ", drawdown: " + metrics.maxDrawdown +
                ", win rate: " + metrics.winRate +
                ", nb trades: " + metrics.tradeCount +
                ", profit factor: " + metrics.profitFactor
            );
        }
    }

    /**
     * Affiche les résultats d'une liste de WalkForwardResult (MACD) dans la console
     */
    public static void printWalkForwardResultsMacd(List<WalkForwardResult> results) {
        for (WalkForwardResult res : results) {
            MacdParams params = (MacdParams) res.params;
            RiskResult metrics = res.result;
            System.out.println(
                "Fenêtre optimisation: " + res.startOptIdx + "-" + res.endOptIdx +
                ", fenêtre test: " + res.startTestIdx + "-" + res.endTestIdx +
                ", MACD params: short=" + params.shortPeriod + ", long=" + params.longPeriod + ", signal=" + params.signalPeriod +
                ", rendement test: " + metrics.rendement +
                ", drawdown: " + metrics.maxDrawdown +
                ", win rate: " + metrics.winRate +
                ", nb trades: " + metrics.tradeCount +
                ", profit factor: " + metrics.profitFactor
            );
        }
    }

    /**
     * Affiche les résultats génériques d'une liste de WalkForwardResult dans la console
     */
    public static void printWalkForwardResultsGeneric(List<WalkForwardResult> results) {
        for (WalkForwardResult res : results) {
            RiskResult metrics = res.result;
            System.out.println(
                "Fenêtre optimisation: " + res.startOptIdx + "-" + res.endOptIdx +
                ", fenêtre test: " + res.startTestIdx + "-" + res.endTestIdx +
                ", params: " + res.params.toString() +
                ", rendement test: " + metrics.rendement +
                ", drawdown: " + metrics.maxDrawdown +
                ", win rate: " + metrics.winRate +
                ", nb trades: " + metrics.tradeCount +
                ", profit factor: " + metrics.profitFactor
            );
        }
    }
}
