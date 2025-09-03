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
    private final List<String> logs = new ArrayList<>();

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

    public List<TradeStrategy> getActiveStrategies() {
        return strategyManager.getActiveStrategies();
    }

    public void setActiveStrategies(List<TradeStrategy> strategies) {
        // Permet de changer dynamiquement les stratégies actives
        strategyManager.getStrategies().clear();
        strategyManager.getStrategies().addAll(strategies);
    }

    public void setActiveStrategiesByNames(List<String> strategyNames) {
        List<TradeStrategy> selected = new ArrayList<>();
        for (String name : strategyNames) {
            getAllStrategies().stream()
                .filter(s -> s.getName().equalsIgnoreCase(name))
                .findFirst()
                .ifPresent(selected::add);
        }
        strategyManager.setActiveStrategies(selected);
        addLog("Stratégies actives modifiées : " + strategyNames);
    }

    public void setCombinationMode(StrategyManager.CombinationMode mode) {
        strategyManager.setCombinationMode(mode);
        addLog("Mode de combinaison changé : " + mode.name());
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

    public List<String> getAllStrategyNames() {
        List<String> names = new ArrayList<>();
        for (TradeStrategy s : allStrategies) {
            names.add(s.getName());
        }
        return names;
    }

    public List<String> getActiveStrategyNames() {
        List<String> names = new ArrayList<>();
        for (TradeStrategy s : getActiveStrategies()) {
            names.add(s.getName());
        }
        return names;
    }

    public void addLog(String log) {
        if (logs.size() > 50) logs.remove(0); // Limite à 50 logs récents
        logs.add("[" + new Date() + "] " + log);
    }

    public List<String> getLogs() {
        return new ArrayList<>(logs);
    }
}
