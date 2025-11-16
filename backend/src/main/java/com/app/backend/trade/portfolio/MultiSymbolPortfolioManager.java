package com.app.backend.trade.portfolio;

import com.app.backend.trade.controller.LstmHelper;
import com.app.backend.trade.model.PreditLsdm;
import com.app.backend.trade.model.SignalType;
import com.app.backend.trade.model.alpaca.Order;
import com.app.backend.trade.service.AlpacaService;
import com.app.backend.trade.model.CompteEntity;
import com.app.backend.trade.model.PortfolioDto;
import com.app.backend.trade.portfolio.learning.PortfolioAllocationModelRepository;
import com.app.backend.trade.portfolio.learning.PortfolioAllocationModelRepository.LoadedAllocationModel;
import com.app.backend.trade.portfolio.learning.PortfolioAllocationTrainer;
import com.app.backend.trade.portfolio.learning.PortfolioLearningConfig;
import com.app.backend.trade.portfolio.direct.PortfolioDirectAllocatorService;
import com.app.backend.trade.portfolio.advanced.PortfolioMultiHeadAllocatorService;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Gestion multi-symboles : collecte signaux LSTM déjà entraînés + construction allocation + exécution ordres.
 */
@Service
public class MultiSymbolPortfolioManager {

    private static final Logger logger = LoggerFactory.getLogger(MultiSymbolPortfolioManager.class);

    // Dépendances
    private final LstmHelper lstmHelper;
    private final AlpacaService alpacaService;
    private final JdbcTemplate jdbcTemplate;
    // Nouveau: modèle d'allocation sauvegardé + builder de features
    private final PortfolioAllocationModelRepository allocationRepo;
    private final PortfolioLearningConfig allocationConfig;
    private final PortfolioAllocationTrainer allocationTrainer;

    // Paramètres de risque / config simples (ajustables)
    private double maxWeightPerSymbol = 0.03;          // 3% du NAV
    private double maxGrossExposure = 0.90;            // 60% capital investi max
    private double minScoreThreshold = 0.10;           // score minimal pour ouverture
    private int maxNewPositionsPerCycle = 15;          // limiter churn
    private boolean allowShort = false;                // si SELL => sortir position plutôt que short

    // Cache signaux du cycle courant
    private final Map<String, PreditLsdm> lastPredictions = new HashMap<>();
    private final Map<String, ModelMetrics> metricsCache = new HashMap<>();

    // Service optionnel d'allocation directe (poids multi-symboles)
    private PortfolioDirectAllocatorService directAllocatorService; // setter injection facultative
    private PortfolioMultiHeadAllocatorService multiHeadAllocatorService; // nouveau service avancé

    public MultiSymbolPortfolioManager(LstmHelper lstmHelper,
                                       AlpacaService alpacaService,
                                       JdbcTemplate jdbcTemplate,
                                       PortfolioAllocationModelRepository allocationRepo,
                                       PortfolioLearningConfig allocationConfig,
                                       PortfolioAllocationTrainer allocationTrainer) {
        this.lstmHelper = lstmHelper;
        this.alpacaService = alpacaService;
        this.jdbcTemplate = jdbcTemplate;
        this.allocationRepo = allocationRepo;
        this.allocationConfig = allocationConfig;
        this.allocationTrainer = allocationTrainer;
    }

    /** DTO interne pour métriques de modèle. */
    public static class ModelMetrics {
        public Double profitFactor;
        public Double winRate;
        public Double maxDrawdown;
        public Double businessScore;
        public Double rendement;
        public Double sumProfit;
        public Integer totalTrades;
        public Double mse;
        public Double rmse;
    }

    /**
     * Exécution complète d'un cycle quotidien : collecte signaux -> scoring -> allocation -> ordres.
     */
    public Map<String, Object> runDailyCycle(CompteEntity compte, List<String> universe, String tri) {
        Map<String, Object> report = new LinkedHashMap<>();
        PortfolioDto portfolio = alpacaService.getPortfolioWithPositions(compte);
        double nav = estimateNav(portfolio);
        Map<String, Integer> currentQtyMap = extractCurrentQuantities(portfolio);

        // 1. Collecte des prédictions (cache DB sinon calcul) + métriques
        List<PreditLsdm> preds = new ArrayList<>();
        for (String sym : universe) {
            try {
                PreditLsdm p = lstmHelper.getPreditFromDB(sym, tri);
                if (p == null) {
                    p = lstmHelper.getPredit(sym, tri); // fallback -> calcul
                }
                if (p != null && p.getSignal() != null) {
                    lastPredictions.put(sym, p);
                    metricsCache.computeIfAbsent(sym, s -> fetchMetrics(s));
                    preds.add(p);
                }
            } catch (Exception e) {
                logger.warn("Prediction échouée pour {} : {}", sym, e.getMessage());
            }
        }
        report.put("predictions_count", preds.size());

        // 2. Scoring ou allocation directe
        Map<String, Double> scores;
        Map<String, Double> targetWeights;
        boolean usedDirectAllocator = false;
        boolean usedMultiHeadAllocator = false;
        try {
            if (multiHeadAllocatorService != null) {
                targetWeights = multiHeadAllocatorService.inferWeights(lastPredictions, nav, maxGrossExposure, maxWeightPerSymbol, allowShort, tri);
                if (!targetWeights.isEmpty()) {
                    usedMultiHeadAllocator = true;
                    scores = new HashMap<>();
                } else {
                    // fallback vers direct ou scoring
                    if (directAllocatorService != null) {
                        targetWeights = directAllocatorService.predictTargetWeights(lastPredictions, metricsCache, nav, maxGrossExposure, maxWeightPerSymbol, allowShort);
                        usedDirectAllocator = true;
                        scores = new HashMap<>();
                    } else {
                        // scoring classique
                        LoadedAllocationModel loaded = allocationRepo.loadLatestModel(tri, allocationConfig);
                        if (loaded != null && loaded.model != null) {
                            // Préparation batch features
                            List<String> syms = new ArrayList<>();
                            List<double[]> featsList = new ArrayList<>();
                            for (String sym : lastPredictions.keySet()) {
                                double[] f = allocationTrainer.buildInferenceFeatures(sym, tri);
                                if (f != null && f.length == loaded.model.getInputSize()) {
                                    syms.add(sym);
                                    featsList.add(f);
                                }
                            }
                            if (!featsList.isEmpty()) {
                                double[][] feats = featsList.toArray(new double[0][]);
                                double[] sc = loaded.model.predictBatch(feats);
                                Map<String, Double> m = new HashMap<>();
                                for (int i = 0; i < syms.size(); i++) {
                                    m.put(syms.get(i), sc[i]);
                                }
                                scores = m;
                            } else {
                                scores = computeScores(preds);
                            }
                            report.put("allocation_model_version", loaded.algoVersion);
                        } else {
                            scores = computeScores(preds);
                            report.put("allocation_model_version", "none");
                        }
                        targetWeights = buildTargetWeights(scores);
                    }
                }
            } else if (directAllocatorService != null) {
                targetWeights = directAllocatorService.predictTargetWeights(lastPredictions, metricsCache, nav, maxGrossExposure, maxWeightPerSymbol, allowShort);
                usedDirectAllocator = true;
                scores = new HashMap<>();
            } else {
                LoadedAllocationModel loaded = allocationRepo.loadLatestModel(tri, allocationConfig);
                if (loaded != null && loaded.model != null) {
                    // Préparation batch features
                    List<String> syms = new ArrayList<>();
                    List<double[]> featsList = new ArrayList<>();
                    for (String sym : lastPredictions.keySet()) {
                        double[] f = allocationTrainer.buildInferenceFeatures(sym, tri);
                        if (f != null && f.length == loaded.model.getInputSize()) {
                            syms.add(sym);
                            featsList.add(f);
                        }
                    }
                    if (!featsList.isEmpty()) {
                        double[][] feats = featsList.toArray(new double[0][]);
                        double[] sc = loaded.model.predictBatch(feats);
                        Map<String, Double> m = new HashMap<>();
                        for (int i = 0; i < syms.size(); i++) {
                            m.put(syms.get(i), sc[i]);
                        }
                        scores = m;
                    } else {
                        scores = computeScores(preds);
                    }
                    report.put("allocation_model_version", loaded.algoVersion);
                } else {
                    scores = computeScores(preds);
                    report.put("allocation_model_version", "none");
                }
                targetWeights = buildTargetWeights(scores);
            }
        } catch (Exception e) {
            logger.warn("Erreur allocation (multi-head/direct/scoring): {}", e.getMessage());
            if (directAllocatorService != null) {
                targetWeights = directAllocatorService.predictTargetWeights(lastPredictions, metricsCache, nav, maxGrossExposure, maxWeightPerSymbol, allowShort);
                usedDirectAllocator = true;
                scores = new HashMap<>();
            } else {
                scores = computeScores(preds);
                targetWeights = buildTargetWeights(scores);
            }
        }
        report.put("used_direct_allocator", usedDirectAllocator);
        report.put("used_multihead_allocator", usedMultiHeadAllocator);
        report.put("scores", scores);

        // Ajustement SELL (si long-only) seulement si on n'a pas déjà forcé via direct allocator
        if (!usedDirectAllocator) {
            for (Map.Entry<String, PreditLsdm> ent : lastPredictions.entrySet()) {
                SignalType sig = ent.getValue().getSignal();
                if (!allowShort && sig == SignalType.SELL) {
                    int held = currentQtyMap.getOrDefault(ent.getKey(), 0);
                    if (held > 0) { targetWeights.put(ent.getKey(), 0.0); }
                }
            }
        }
        report.put("targetWeights", targetWeights);

        // 4. Ordres
        List<Map<String, Object>> ordersPlanned = planOrders(compte, portfolio, targetWeights, nav);
        report.put("ordersPlanned", ordersPlanned);

        // 5. Exécution
        List<String> executed = new ArrayList<>();
        for (Map<String, Object> ord : ordersPlanned) {
            if (Boolean.TRUE.equals(ord.get("skip"))) continue;
            try {
                Order o = alpacaService.placeOrder(
                        compte,
                        (String) ord.get("symbol"),
                        (Double) ord.get("qty"),
                        (String) ord.get("side"),
                        null,
                        null,
                        null,
                        null,
                        true,
                        false
                );
                executed.add(o.getId());
                ord.put("status", "EXECUTED");
            } catch (Exception ex) {
                ord.put("status", "FAILED:" + ex.getMessage());
                logger.warn("Ordre échoué {} : {}", ord, ex.getMessage());
            }
        }
        report.put("executed_orders", executed);
        report.put("nav_estimated", nav);
        return report;
    }

    /** Calcul scores multi-critères. */
    private Map<String, Double> computeScores(List<PreditLsdm> preds) {
        Map<String, Double> scores = new HashMap<>();
        for (PreditLsdm p : preds) {
            String sym = extractSymbol(p);
            if (sym == null) continue;
            ModelMetrics m = metricsCache.get(sym);
            if (m == null) continue; // correction parenthèses
            if (!riskFilter(m)) continue;
            double directionFactor = p.getSignal() == SignalType.BUY ? 1.0 : p.getSignal() == SignalType.SELL ? (allowShort ? -1.0 : 0.0) : 0.0;
            if (directionFactor == 0.0) continue;
            double stability = m.totalTrades != null && m.totalTrades > 50 ? 1.0 : 0.7;
            double score = directionFactor * safe(m.businessScore) * safe(m.profitFactor) * safe(m.winRate) * (1 - safe(m.maxDrawdown)) * stability;
            if (score >= minScoreThreshold) {
                scores.put(sym, score);
            }
        }
        return scores;
    }

    private boolean riskFilter(ModelMetrics m) {
        return m.profitFactor != null && m.profitFactor > 1.0
                && m.winRate != null && m.winRate > 0.5
                && m.maxDrawdown != null && m.maxDrawdown < 0.35
                && m.businessScore != null && m.businessScore > 0.0;
    }

    private double safe(Double v) {return v == null ? 0.0 : v;}

    /** Construction des poids cibles (normalisation + contraintes). */
    private Map<String, Double> buildTargetWeights(Map<String, Double> scores) {
        if (scores.isEmpty()) return Collections.emptyMap();
        double sum = scores.values().stream().mapToDouble(Double::doubleValue).sum();
        Map<String, Double> raw = scores.entrySet().stream()
                .collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue() / sum));
        Map<String, Double> capped = new HashMap<>();
        double gross = 0.0;
        for (Map.Entry<String, Double> e : raw.entrySet()) {
            double w = Math.min(e.getValue() * maxGrossExposure, maxWeightPerSymbol);
            capped.put(e.getKey(), w);
            gross += Math.abs(w);
        }
        if (gross > maxGrossExposure && gross > 0) {
            double ratio = maxGrossExposure / gross;
            capped.replaceAll((k,v) -> v * ratio);
        }
        return capped;
    }

    /** Planification des ordres à partir des poids cibles. */
    private List<Map<String, Object>> planOrders(CompteEntity compte, PortfolioDto portfolio, Map<String, Double> targetWeights, double nav) {
        List<Map<String, Object>> orders = new ArrayList<>();
        Map<String, Integer> currentQty = extractCurrentQuantities(portfolio);
        int newPositions = 0;
        for (Map.Entry<String, Double> e : targetWeights.entrySet()) {
            String sym = e.getKey();
            double targetW = e.getValue();
            PreditLsdm pred = lastPredictions.get(sym);
            if (pred == null) continue;
            SignalType sig = pred.getSignal();
            Double lastPrice = null;
            try { lastPrice = alpacaService.getLastPrice(compte, sym); } catch (Exception ex) { /* ignore */ }
            if (lastPrice == null || lastPrice <= 0) {
                Map<String, Object> fail = new HashMap<>();
                fail.put("symbol", sym); fail.put("skip", true); fail.put("reason", "prix_invalide");
                orders.add(fail); continue;
            }
            int current = currentQty.getOrDefault(sym, 0);
            double targetValue = targetW * nav;
            int targetQty = (int) Math.floor(targetValue / lastPrice);
            int delta = targetQty - current;
            if (delta == 0) continue;
            String side;
            if (delta > 0) {
                side = "buy";
                if (current == 0) {
                    newPositions++;
                    if (newPositions > maxNewPositionsPerCycle) {
                        Map<String, Object> skip = new HashMap<>();
                        skip.put("symbol", sym); skip.put("skip", true); skip.put("reason", "limite_new_positions");
                        orders.add(skip); continue;
                    }
                }
            } else {
                side = "sell";
                delta = Math.abs(delta);
            }
            Map<String, Object> ord = new HashMap<>();
            ord.put("symbol", sym);
            ord.put("side", side);
            ord.put("qty", (double) delta);
            ord.put("targetWeight", targetW);
            ord.put("currentQty", current);
            ord.put("targetQty", targetQty);
            ord.put("signal", sig != null ? sig.name() : null);
            orders.add(ord);
        }
        return orders;
    }

    private Map<String, Integer> extractCurrentQuantities(PortfolioDto portfolio) {
        Map<String, Integer> map = new HashMap<>();
        if (portfolio.getPositions() != null) {
            for (Map<String, Object> pos : portfolio.getPositions()) {
                Object symObj = pos.get("symbol");
                Object qtyObj = pos.get("qty");
                if (symObj != null && qtyObj != null) {
                    try {
                        map.put(symObj.toString(), Integer.parseInt(qtyObj.toString()));
                    } catch (NumberFormatException ignored) {}
                }
            }
        }
        return map;
    }

    /** Estimation NAV = cash + somme market_value positions. */
    private double estimateNav(PortfolioDto dto) {
        double cash = 0.0;
        if (dto.getAccount() != null && dto.getAccount().get("cash") != null) {
            try { cash = Double.parseDouble(dto.getAccount().get("cash").toString()); } catch (Exception ignored) {}
        }
        double mv = 0.0;
        if (dto.getPositions() != null) {
            for (Map<String, Object> pos : dto.getPositions()) {
                Object mvObj = pos.get("market_value");
                if (mvObj != null) {
                    try { mv += Double.parseDouble(mvObj.toString()); } catch (Exception ignored) {}
                }
            }
        }
        return cash + mv;
    }

    /** Récupération métriques modèle (dernière ligne lstm_models) pour un symbole. */
    private ModelMetrics fetchMetrics(String symbol) {
        try {
            String sql = "SELECT profit_factor, win_rate, max_drawdown, business_score, rendement, sum_profit, total_trades, mse, rmse FROM trade_ai.lstm_models WHERE symbol = ? ORDER BY updated_date DESC LIMIT 1";
            return jdbcTemplate.query(sql, ps -> ps.setString(1, symbol), rs -> {
                if (rs.next()) {
                    ModelMetrics m = new ModelMetrics();
                    m.profitFactor = rs.getDouble("profit_factor");
                    m.winRate = rs.getDouble("win_rate");
                    m.maxDrawdown = rs.getDouble("max_drawdown");
                    m.businessScore = rs.getDouble("business_score");
                    m.rendement = rs.getDouble("rendement");
                    m.sumProfit = rs.getDouble("sum_profit");
                    m.totalTrades = rs.getInt("total_trades");
                    m.mse = rs.getDouble("mse");
                    m.rmse = rs.getDouble("rmse");
                    return m;
                }
                return null;
            });
        } catch (Exception e) {
            logger.warn("Fetch metrics échoué pour {} : {}", symbol, e.getMessage());
            return null;
        }
    }

    /** Extraction du symbole depuis PreditLsdm via cache interne. */
    private String extractSymbol(PreditLsdm p) {
        for (Map.Entry<String,PreditLsdm> ent : lastPredictions.entrySet()) {
            if (ent.getValue() == p) return ent.getKey();
        }
        return null;
    }

    // Setters de configuration
    public void setMaxWeightPerSymbol(double v) { this.maxWeightPerSymbol = v; }
    public void setMaxGrossExposure(double v) { this.maxGrossExposure = v; }
    public void setMinScoreThreshold(double v) { this.minScoreThreshold = v; }
    public void setMaxNewPositionsPerCycle(int v) { this.maxNewPositionsPerCycle = v; }
    public void setAllowShort(boolean allowShort) { this.allowShort = allowShort; }

    // Setter optionnel (Spring pourra l'injecter si bean défini)
    @org.springframework.beans.factory.annotation.Autowired(required = false)
    public void setDirectAllocatorService(PortfolioDirectAllocatorService svc) { this.directAllocatorService = svc; }
    @org.springframework.beans.factory.annotation.Autowired(required = false)
    public void setMultiHeadAllocatorService(PortfolioMultiHeadAllocatorService svc) { this.multiHeadAllocatorService = svc; }

    public Map<String, PreditLsdm> getLastPredictions() { return Collections.unmodifiableMap(lastPredictions); }
    public Map<String, ModelMetrics> getMetricsCache() { return Collections.unmodifiableMap(metricsCache); }
}
