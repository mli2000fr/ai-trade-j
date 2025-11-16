package com.app.backend.trade.portfolio.advanced;

import com.app.backend.trade.model.PreditLsdm;
import com.app.backend.trade.portfolio.MultiSymbolPortfolioManager;
import com.app.backend.trade.portfolio.learning.PortfolioAllocationTrainer;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Service d'inférence pour le modèle multi-tête.
 * Produit directement des poids (possiblement short si side<0) avant contraintes globales.
 */
@Service
public class PortfolioMultiHeadAllocatorService {

    private final PortfolioMultiHeadRepository repository;
    private final PortfolioMultiHeadConfig config;
    private final PortfolioAllocationTrainer allocationTrainer; // réutilise buildInferenceFeatures

    public PortfolioMultiHeadAllocatorService(PortfolioMultiHeadRepository repository,
                                              PortfolioMultiHeadConfig config,
                                              PortfolioAllocationTrainer allocationTrainer) {
        this.repository = repository;
        this.config = config;
        this.allocationTrainer = allocationTrainer;
    }

    /**
     * Infère poids cibles à partir des prédictions LSTM et métriques.
     * @param universePreds prédictions du cycle (symbol -> PreditLsdm)
     * @param nav valeur liquidative (pour info futur risk sizing)
     * @param maxGrossExposure borne somme |poids|
     * @param maxWeightPerSymbol plafond individuel
     * @param allowShort autoriser shorts (side négatif)
     * @param tri clé de filtrage modèle
     * @return map symbol->poids cible
     */
    public Map<String, Double> inferWeights(Map<String, PreditLsdm> universePreds,
                                            double nav,
                                            double maxGrossExposure,
                                            double maxWeightPerSymbol,
                                            boolean allowShort,
                                            String tri) {
        if (universePreds == null || universePreds.isEmpty()) return Collections.emptyMap();
        PortfolioMultiHeadRepository.LoadedMultiHead loaded = repository.loadLatest(tri, config);
        if (loaded == null || loaded.model == null) return Collections.emptyMap();
        List<String> syms = new ArrayList<>();
        List<double[]> feats = new ArrayList<>();
        for (String sym : universePreds.keySet()) {
            double[] f = allocationTrainer.buildInferenceFeatures(sym, tri);
            if (f != null && f.length == loaded.model.getInputSize()) {
                syms.add(sym); feats.add(f);
            }
        }
        if (feats.isEmpty()) return Collections.emptyMap();
        double[][] raw = loaded.model.predictBatch(feats.toArray(new double[0][])); // shape [n,3]
        Map<String, Double> preliminary = new LinkedHashMap<>();
        for (int i = 0; i < syms.size(); i++) {
            double pSelect = raw[i][0];
            double wRaw = raw[i][1];
            double side = raw[i][2]; // -1..1
            if (pSelect < config.getSelectThreshold()) continue; // filtrage probabilité
            double w = pSelect * wRaw; // pondération par certitude
            if (!allowShort && side < 0) continue; // ignorer shorts si non autorisés
            double oriented = w * (allowShort ? side : Math.max(0.0, side));
            if (oriented <= 0) continue;
            preliminary.put(syms.get(i), oriented);
        }
        if (preliminary.isEmpty()) return Collections.emptyMap();
        // Normalisation + contraintes
        double sumAbs = preliminary.values().stream().mapToDouble(Math::abs).sum();
        Map<String, Double> capped = preliminary.entrySet().stream()
                .collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue() / sumAbs));
        // Appliquer gross exposure et plafonds
        double gross = 0.0;
        for (Map.Entry<String, Double> e : capped.entrySet()) {
            double w = e.getValue() * maxGrossExposure;
            if (Math.abs(w) > maxWeightPerSymbol) {
                w = Math.signum(w) * maxWeightPerSymbol;
            }
            e.setValue(w); gross += Math.abs(w);
        }
        if (gross > maxGrossExposure && gross > 0) {
            double ratio = maxGrossExposure / gross;
            capped.replaceAll((k,v) -> v * ratio);
        }
        return capped;
    }
}

