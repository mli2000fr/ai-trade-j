package com.app.backend.trade.service;

import com.app.backend.trade.strategy.*;
import org.springframework.stereotype.Service;
import org.ta4j.core.BarSeries;
import org.ta4j.core.Rule;
import java.util.*;

@Service
public class StrategyService {
    private final List<TradeStrategy> allStrategies;
    private final StrategyManager strategyManager;

    public StrategyService() {
        // Instanciation de toutes les stratégies disponibles
        allStrategies = Arrays.asList(
                new SmaCrossoverStrategy(),
                new RsiStrategy(),
                new MacdStrategy(),
                new BreakoutStrategy(),
                new MeanReversionStrategy(),
                new TrendFollowingStrategy()
        );
        // Par défaut, toutes les stratégies sont actives, mode majorité
        strategyManager = new StrategyManager(allStrategies, StrategyManager.CombinationMode.MAJORITY);
    }

    public List<TradeStrategy> getAllStrategies() {
        return allStrategies;
    }

    public void setActiveStrategies(List<TradeStrategy> strategies) {
        // Permet de changer dynamiquement les stratégies actives
        strategyManager.getStrategies().clear();
        strategyManager.getStrategies().addAll(strategies);
    }

    public void setCombinationMode(StrategyManager.CombinationMode mode) {
        strategyManager.setCombinationMode(mode);
    }

    public Rule getEntryRule(BarSeries series) {
        return strategyManager.getCombinedEntryRule(series);
    }

    public Rule getExitRule(BarSeries series) {
        return strategyManager.getCombinedExitRule(series);
    }

    public StrategyManager getStrategyManager() {
        return strategyManager;
    }
}

