package com.app.backend.trade.strategy;

import org.ta4j.core.BarSeries;
import org.ta4j.core.Rule;
import java.util.*;

/**
 * Permet de gérer dynamiquement plusieurs stratégies de trading.
 * Peut combiner les signaux (majorité, pondération, etc.) ou sélectionner une stratégie active.
 */
public class StrategyManager {
    private final List<TradeStrategy> strategies;
    private CombinationMode combinationMode;
    private TradeStrategy selectedStrategy;

    public enum CombinationMode {
        MAJORITY, // Décision par vote majoritaire
        ALL,      // Tous les signaux doivent être d'accord
        ANY,      // Un seul signal suffit
        SINGLE    // Utiliser une seule stratégie (selectedStrategy)
    }

    public StrategyManager(List<TradeStrategy> strategies, CombinationMode mode) {
        this.strategies = new ArrayList<>(strategies);
        this.combinationMode = mode;
        if (!strategies.isEmpty()) {
            this.selectedStrategy = strategies.get(0);
        }
    }

    public void setCombinationMode(CombinationMode mode) {
        this.combinationMode = mode;
    }

    public void setSelectedStrategy(TradeStrategy strategy) {
        if (strategies.contains(strategy)) {
            this.selectedStrategy = strategy;
        }
    }

    public List<TradeStrategy> getStrategies() {
        return Collections.unmodifiableList(strategies);
    }

    public Rule getCombinedEntryRule(BarSeries series) {
        switch (combinationMode) {
            case MAJORITY:
                return majorityRule(series, true);
            case ALL:
                return allRule(series, true);
            case ANY:
                return anyRule(series, true);
            case SINGLE:
            default:
                return selectedStrategy.getEntryRule(series);
        }
    }

    public Rule getCombinedExitRule(BarSeries series) {
        switch (combinationMode) {
            case MAJORITY:
                return majorityRule(series, false);
            case ALL:
                return allRule(series, false);
            case ANY:
                return anyRule(series, false);
            case SINGLE:
            default:
                return selectedStrategy.getExitRule(series);
        }
    }

    // Combine les règles d'entrée/sortie par vote majoritaire
    private Rule majorityRule(BarSeries series, boolean entry) {
        List<Rule> rules = new ArrayList<>();
        for (TradeStrategy s : strategies) {
            rules.add(entry ? s.getEntryRule(series) : s.getExitRule(series));
        }
        return (index, tradingRecord) -> {
            int count = 0;
            for (Rule r : rules) {
                if (r.isSatisfied(index, tradingRecord)) count++;
            }
            return count > rules.size() / 2;
        };
    }

    // Combine les règles d'entrée/sortie : tous doivent être d'accord
    private Rule allRule(BarSeries series, boolean entry) {
        List<Rule> rules = new ArrayList<>();
        for (TradeStrategy s : strategies) {
            rules.add(entry ? s.getEntryRule(series) : s.getExitRule(series));
        }
        return (index, tradingRecord) -> {
            for (Rule r : rules) {
                if (!r.isSatisfied(index, tradingRecord)) return false;
            }
            return true;
        };
    }

    // Combine les règles d'entrée/sortie : un seul suffit
    private Rule anyRule(BarSeries series, boolean entry) {
        List<Rule> rules = new ArrayList<>();
        for (TradeStrategy s : strategies) {
            rules.add(entry ? s.getEntryRule(series) : s.getExitRule(series));
        }
        return (index, tradingRecord) -> {
            for (Rule r : rules) {
                if (r.isSatisfied(index, tradingRecord)) return true;
            }
            return false;
        };
    }
}

