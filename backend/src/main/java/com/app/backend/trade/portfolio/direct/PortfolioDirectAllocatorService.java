package com.app.backend.trade.portfolio.direct;

import com.app.backend.trade.model.PreditLsdm;
import com.app.backend.trade.model.SignalType;
import com.app.backend.trade.portfolio.MultiSymbolPortfolioManager; // pour ModelMetrics
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Service d'allocation directe : convertit prédictions + métriques en poids cibles.
 * Objectif initial simple (baseline) avant un modèle appris end-to-end.
 * Long-only par défaut (SELL => poids 0). Short ignoré tant que planOrders ne gère pas le négatif.
 */
@Service
public class PortfolioDirectAllocatorService {

    /**
     * Calcule des poids cibles directement, en appliquant les contraintes d'exposition.
     * @param predictions map symbol->PreditLsdm
     * @param metricsCache map symbol->metrics modèle LSTM
     * @param nav valeur liquidative estimée (non utilisée ici mais laissé pour évolution future risk-par-capital)
     * @param maxGrossExposure exposition totale max (somme des poids)
     * @param maxWeightPerSymbol plafond par symbole
     * @param allowShort autoriser shorts (non utilisé pour l'instant)
     * @return map symbol->weight (0..maxWeightPerSymbol), somme ≤ maxGrossExposure
     */
    public Map<String, Double> predictTargetWeights(Map<String, PreditLsdm> predictions,
                                                    Map<String, MultiSymbolPortfolioManager.ModelMetrics> metricsCache,
                                                    double nav,
                                                    double maxGrossExposure,
                                                    double maxWeightPerSymbol,
                                                    boolean allowShort) {
        if (predictions == null || predictions.isEmpty()) return Collections.emptyMap();
        // 1. Score brut par symbole
        Map<String, Double> rawScores = new HashMap<>();
        for (Map.Entry<String, PreditLsdm> e : predictions.entrySet()) {
            String sym = e.getKey();
            PreditLsdm pred = e.getValue();
            MultiSymbolPortfolioManager.ModelMetrics m = metricsCache.get(sym);
            if (pred == null || m == null) continue;
            SignalType sig = pred.getSignal();
            if (sig != SignalType.BUY && !allowShort) continue; // long-only
            if (!riskFilter(m)) continue;
            double directionFactor = (sig == SignalType.BUY) ? 1.0 : (allowShort && sig == SignalType.SELL ? -1.0 : 0.0);
            if (directionFactor == 0.0) continue;
            double stability = m.totalTrades != null && m.totalTrades > 50 ? 1.0 : 0.7;
            double score = directionFactor * safe(m.businessScore) * safe(m.profitFactor) * safe(m.winRate) * (1 - safe(m.maxDrawdown)) * stability;
            if (score > 0) rawScores.put(sym, score);
        }
        if (rawScores.isEmpty()) return Collections.emptyMap();
        // 2. Normalisation des scores -> poids candidats
        double sum = rawScores.values().stream().mapToDouble(Double::doubleValue).sum();
        Map<String, Double> weights = rawScores.entrySet().stream()
                .collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue() / sum));
        // 3. Application plafonds individuels + scale exposition
        Map<String, Double> capped = new LinkedHashMap<>();
        double gross = 0.0;
        for (Map.Entry<String, Double> e : weights.entrySet()) {
            double w = Math.min(e.getValue() * maxGrossExposure, maxWeightPerSymbol);
            capped.put(e.getKey(), w);
            gross += Math.abs(w);
        }
        if (gross > maxGrossExposure && gross > 0) {
            double ratio = maxGrossExposure / gross;
            capped.replaceAll((k, v) -> v * ratio);
        }
        return capped;
    }

    private boolean riskFilter(MultiSymbolPortfolioManager.ModelMetrics m) {
        return m.profitFactor != null && m.profitFactor > 1.0
                && m.winRate != null && m.winRate > 0.5
                && m.maxDrawdown != null && m.maxDrawdown < 0.35
                && m.businessScore != null && m.businessScore > 0.0;
    }

    private double safe(Double v) { return v == null ? 0.0 : v; }
}

