package com.app.backend.trade.portfolio.rl;

import com.app.backend.trade.model.PreditLsdm;
import com.app.backend.trade.portfolio.learning.PortfolioAllocationTrainer;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * Service d'inférence RL (policy offline) : convertit les logits en poids.
 * Stratégie:
 * - Charge le dernier policy model RL
 * - Construit features via allocationTrainer.buildInferenceFeatures
 * - Produit logits -> softmax (long-only) ou softmax sur |logits| avec signe (short autorisé)
 * - Applique contraintes gross exposure et plafond par symbole
 */
@Service
public class PortfolioRlAllocatorService {

    private final PortfolioRlPolicyRepository policyRepository;
    private final PortfolioRlConfig rlConfig;
    private final PortfolioAllocationTrainer allocationTrainer;

    public PortfolioRlAllocatorService(PortfolioRlPolicyRepository policyRepository,
                                       PortfolioRlConfig rlConfig,
                                       PortfolioAllocationTrainer allocationTrainer) {
        this.policyRepository = policyRepository;
        this.rlConfig = rlConfig;
        this.allocationTrainer = allocationTrainer;
    }

    public Map<String, Double> inferWeights(Map<String, PreditLsdm> universePreds,
                                            double nav,
                                            double maxGrossExposure,
                                            double maxWeightPerSymbol,
                                            boolean allowShort,
                                            String tri) {
        if (universePreds == null || universePreds.isEmpty()) return Collections.emptyMap();
        PortfolioRlPolicyRepository.LoadedRlPolicy loaded = policyRepository.loadLatest(tri, rlConfig);
        if (loaded == null || loaded.model == null) return Collections.emptyMap();
        List<String> syms = new ArrayList<>();
        List<double[]> feats = new ArrayList<>();
        for (String sym : universePreds.keySet()) {
            double[] f = allocationTrainer.buildInferenceFeatures(sym, tri);
            if (f != null && f.length == loaded.model.getInputSize()) { syms.add(sym); feats.add(f); }
        }
        if (feats.isEmpty()) return Collections.emptyMap();
        double[][] featsArr = feats.toArray(new double[0][]);
        double[] logits = loaded.model.predictBatch(featsArr);
        // Filtrage neutre: retirer NaN
        List<Integer> validIdx = new ArrayList<>();
        for (int i = 0; i < logits.length; i++) { if (!Double.isNaN(logits[i])) validIdx.add(i); }
        if (validIdx.isEmpty()) return Collections.emptyMap();
        Map<String, Double> rawWeights = new LinkedHashMap<>();
        if (!allowShort) {
            // Long-only: softmax sur logits positifs > 0
            double maxLogit = validIdx.stream().mapToDouble(i -> logits[i]).max().orElse(0.0);
            double sumExp = 0.0;
            double[] expVals = new double[logits.length];
            for (int i : validIdx) {
                double l = logits[i];
                if (l <= 0) { expVals[i] = 0; continue; }
                double e = Math.exp(l - maxLogit); // stabilité
                expVals[i] = e; sumExp += e;
            }
            if (sumExp == 0) return Collections.emptyMap();
            for (int i : validIdx) {
                if (expVals[i] == 0) continue;
                double w = expVals[i] / sumExp; // proportion
                rawWeights.put(syms.get(i), w);
            }
        } else {
            // Long/short: softmax sur |logit| puis signe
            double maxAbs = validIdx.stream().mapToDouble(i -> Math.abs(logits[i])).max().orElse(0.0);
            double sumExp = 0.0; double[] expAbs = new double[logits.length];
            for (int i : validIdx) {
                double a = Math.abs(logits[i]);
                if (a == 0) { expAbs[i] = 0; continue; }
                double e = Math.exp(a - maxAbs); expAbs[i] = e; sumExp += e;
            }
            if (sumExp == 0) return Collections.emptyMap();
            for (int i : validIdx) {
                if (expAbs[i] == 0) continue;
                double base = expAbs[i] / sumExp; // magnitude
                double signed = base * Math.signum(logits[i]);
                rawWeights.put(syms.get(i), signed);
            }
        }
        if (rawWeights.isEmpty()) return Collections.emptyMap();
        // Appliquer gross exposure et plafond
        double gross = rawWeights.values().stream().mapToDouble(Math::abs).sum();
        Map<String, Double> finalWeights = new LinkedHashMap<>();
        for (Map.Entry<String, Double> e : rawWeights.entrySet()) {
            double w = e.getValue() * (gross > 0 ? (maxGrossExposure / gross) : 0.0); // scale à maxGrossExposure
            if (Math.abs(w) > maxWeightPerSymbol) {
                w = Math.signum(w) * maxWeightPerSymbol;
            }
            finalWeights.put(e.getKey(), w);
        }
        // Re-normaliser si dépassement après capping
        double newGross = finalWeights.values().stream().mapToDouble(Math::abs).sum();
        if (newGross > maxGrossExposure && newGross > 0) {
            double ratio = maxGrossExposure / newGross;
            finalWeights.replaceAll((k,v) -> v * ratio);
        }
        return finalWeights;
    }
}

