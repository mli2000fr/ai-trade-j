package com.app.backend.trade.portfolio.learning;

import com.app.backend.trade.model.PreditLsdm;
import com.app.backend.trade.model.SignalType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalDate;
import java.util.*;

/**
 * Trainer supervisé pour le modèle d'allocation.
 * Construit un dataset à partir de l'historique des signaux (table signal_lstm) et des prix (daily_value).
 */
@Component
public class PortfolioAllocationTrainer {

    private static final Logger logger = LoggerFactory.getLogger(PortfolioAllocationTrainer.class);
    private final JdbcTemplate jdbcTemplate;
    private final PortfolioLearningConfig config;

    public PortfolioAllocationTrainer(JdbcTemplate jdbcTemplate, PortfolioLearningConfig config) {
        this.jdbcTemplate = jdbcTemplate;
        this.config = config;
    }

    /**
     * Entraîne un modèle sur un ensemble de symboles.
     * @param symbols liste de symboles
     * @param tri clé tri / style utilisé pour filtrer signaux
     */
    public PortfolioAllocationModel trainModel(List<String> symbols, String tri) {
        List<double[]> featureRows = new ArrayList<>();
        List<Double> targetRows = new ArrayList<>();
        for (String sym : symbols) {
            try {
                buildSymbolDataset(sym, tri, featureRows, targetRows);
            } catch (Exception e) {
                logger.warn("Dataset partiel ignoré {}: {}", sym, e.getMessage());
            }
        }
        if (featureRows.isEmpty()) {
            logger.warn("Dataset vide => modèle placeholder.");
            return new PortfolioAllocationModel(1, config);
        }
        int inputSize = featureRows.get(0).length;
        PortfolioAllocationModel model = new PortfolioAllocationModel(inputSize, config);
        int n = featureRows.size();
        double[][] X = new double[n][inputSize];
        double[][] y = new double[n][1];
        for (int i = 0; i < n; i++) { X[i] = featureRows.get(i); y[i][0] = targetRows.get(i); }
        if (config.isNormalizeFeatures()) { normalizeInPlace(X); }
        org.nd4j.linalg.api.ndarray.INDArray inX = org.nd4j.linalg.factory.Nd4j.create(X);
        org.nd4j.linalg.api.ndarray.INDArray inY = org.nd4j.linalg.factory.Nd4j.create(y);
        for (int epoch = 0; epoch < config.getEpochs(); epoch++) { model.getNetwork().fit(inX, inY); }
        logger.info("Entraînement terminé: {} lignes, inputSize={}", n, inputSize);
        return model;
    }

    /**
     * Construit les lignes features + cible pour un symbole donné.
     * Cible = rendement futur moyen sur lookaheadBars (différence close / close présent).
     */
    private void buildSymbolDataset(String symbol, String tri, List<double[]> featureRows, List<Double> targetRows) {
        String sqlSignals = "SELECT lstm_created_at, signal_lstm, price_lstm, price_clo FROM signal_lstm WHERE symbol = ? AND tri = ? ORDER BY lstm_created_at ASC";
        List<Map<String, Object>> signals = jdbcTemplate.queryForList(sqlSignals, symbol, tri);
        if (signals.size() < config.getMinHistoryBars()) return;
        // Préparer séquence des closes (trading bars séquentielles)
        List<Double> closes = new ArrayList<>();
        for (Map<String, Object> row : signals) {
            Double c = asDouble(row.get("price_clo"));
            if (c != null && c > 0) closes.add(c); else closes.add(Double.NaN);
        }
        // Metrics modèle (une seule récupération)
        String sqlMetric = "SELECT profit_factor, win_rate, max_drawdown, business_score, total_trades FROM trade_ai.lstm_models WHERE symbol = ? ORDER BY updated_date DESC LIMIT 1";
        Map<String, Object> metric = null;
        try { metric = jdbcTemplate.queryForMap(sqlMetric, symbol); } catch (Exception ignored) {}
        double profitFactor = safeMetric(metric, "profit_factor");
        double winRate = safeMetric(metric, "win_rate");
        double maxDD = safeMetric(metric, "max_drawdown");
        double businessScore = safeMetric(metric, "business_score");
        double totalTrades = safeMetric(metric, "total_trades");

        // Parcours séquentiel (trading days) pour label horizonBars
        for (int i = 0; i < signals.size(); i++) {
            int futureIndex = i + config.getLookaheadBars();
            if (futureIndex >= signals.size()) break; // fin
            Double startClose = closes.get(i);
            Double endClose = closes.get(futureIndex);
            if (startClose == null || endClose == null || startClose.isNaN() || endClose.isNaN() || startClose == 0) continue;
            double futureRet = (endClose - startClose) / startClose;
            Map<String,Object> row = signals.get(i);
            String signalStr = String.valueOf(row.get("signal_lstm"));
            SignalType sig;
            try { sig = SignalType.valueOf(signalStr); } catch (Exception e) { sig = SignalType.NONE; }
            double signalBuy = sig == SignalType.BUY ? 1.0 : 0.0;
            double signalSell = sig == SignalType.SELL ? 1.0 : 0.0;
            Double pricePred = asDouble(row.get("price_lstm"));
            Double priceClose = asDouble(row.get("price_clo"));
            double deltaPred = (pricePred != null && priceClose != null && priceClose != 0) ? (pricePred - priceClose) / priceClose : 0.0;
            // Volatilité rolling basée sur closes jusqu'à i
            double vol20 = computeVolatilityRolling(closes, i, 20);
            double[] features = new double[]{signalBuy, signalSell, deltaPred, vol20, profitFactor, winRate, maxDD, businessScore, totalTrades};
            featureRows.add(features);
            targetRows.add(futureRet);
        }
    }

    public double[] buildInferenceFeatures(String symbol, String tri) {
        String sqlLast = "SELECT lstm_created_at, signal_lstm, price_lstm, price_clo FROM signal_lstm WHERE symbol = ? AND tri = ? ORDER BY lstm_created_at DESC LIMIT 25";
        List<Map<String, Object>> recent = jdbcTemplate.queryForList(sqlLast, symbol, tri);
        if (recent.isEmpty()) return null;
        // Reconstituer closes (ordre DESC actuellement -> inverser pour volatilité cohérente)
        Collections.reverse(recent);
        List<Double> closes = new ArrayList<>();
        for (Map<String,Object> r : recent) {
            Double c = asDouble(r.get("price_clo"));
            if (c != null && c > 0) closes.add(c);
        }
        Map<String,Object> last = recent.get(recent.size()-1);
        String sqlMetric = "SELECT profit_factor, win_rate, max_drawdown, business_score, total_trades FROM trade_ai.lstm_models WHERE symbol = ? ORDER BY updated_date DESC LIMIT 1";
        Map<String, Object> metric = null;
        try { metric = jdbcTemplate.queryForMap(sqlMetric, symbol); } catch (Exception ignored) {}
        double profitFactor = safeMetric(metric, "profit_factor");
        double winRate = safeMetric(metric, "win_rate");
        double maxDD = safeMetric(metric, "max_drawdown");
        double businessScore = safeMetric(metric, "business_score");
        double totalTrades = safeMetric(metric, "total_trades");
        String signalStr = String.valueOf(last.get("signal_lstm"));
        SignalType sig;
        try { sig = SignalType.valueOf(signalStr); } catch (Exception e) { sig = SignalType.NONE; }
        double signalBuy = sig == SignalType.BUY ? 1.0 : 0.0;
        double signalSell = sig == SignalType.SELL ? 1.0 : 0.0;
        Double pricePred = asDouble(last.get("price_lstm"));
        Double priceClose = asDouble(last.get("price_clo"));
        double deltaPred = (pricePred != null && priceClose != null && priceClose != 0) ? (pricePred - priceClose) / priceClose : 0.0;
        double vol20 = computeVolatility(closes, Math.min(20, closes.size()));
        double[] f = new double[]{signalBuy, signalSell, deltaPred, vol20, profitFactor, winRate, maxDD, businessScore, totalTrades};
        if (config.isNormalizeFeatures()) { f = normalizeSingle(f); }
        return f;
    }

    private void normalizeInPlace(double[][] X) {
        int cols = X[0].length;
        for (int j = 0; j < cols; j++) {
            double sum = 0.0; for (double[] row : X) sum += row[j];
            double mean = sum / X.length;
            double var = 0.0; for (double[] row : X) var += (row[j]-mean)*(row[j]-mean);
            var /= Math.max(1, X.length-1);
            double std = var <= 0 ? 1.0 : Math.sqrt(var);
            for (double[] row : X) row[j] = (row[j]-mean)/std;
        }
    }

    private double[] normalizeSingle(double[] f) { return f; } // placeholder simple (dataset global déjà normalisé)

    private double computeVolatilityRolling(List<Double> closes, int idx, int window) {
        if (idx < window) return 0.0;
        List<Double> slice = closes.subList(idx - window, idx);
        return computeVolatility(slice, window);
    }

    private double computeFutureReturn(Map<LocalDate, Double> closeMap, LocalDate start, int horizon) {
        Double startPrice = closeMap.get(start);
        if (startPrice == null) return Double.NaN;
        LocalDate futureDate = start.plusDays(horizon);
        Double futurePrice = closeMap.get(futureDate);
        if (futurePrice == null) return Double.NaN;
        return (futurePrice - startPrice) / startPrice;
    }

    private double computeVolatility(List<Double> closes, int window) {
        if (closes.size() < window + 1) return 0.0;
        int start = closes.size() - window - 1;
        List<Double> sub = closes.subList(start, closes.size());
        List<Double> returns = new ArrayList<>();
        for (int i = 1; i < sub.size(); i++) {
            double prev = sub.get(i - 1);
            double curr = sub.get(i);
            if (prev != 0) returns.add((curr - prev) / prev);
        }
        if (returns.isEmpty()) return 0.0;
        double mean = returns.stream().mapToDouble(d -> d).average().orElse(0.0);
        double var = 0.0;
        for (double r : returns) var += (r - mean) * (r - mean);
        var /= returns.size();
        return Math.sqrt(var);
    }

    private double safeMetric(Map<String, Object> metric, String key) {
        if (metric == null || !metric.containsKey(key) || metric.get(key) == null) return 0.0;
        try { return Double.parseDouble(metric.get(key).toString()); } catch (Exception e) { return 0.0; }
    }

    private Double asDouble(Object o) {
        if (o == null) return null;
        try { return Double.parseDouble(o.toString()); } catch (Exception e) { return null; }
    }
}
