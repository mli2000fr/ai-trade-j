package com.app.backend.trade.portfolio.learning;

import com.app.backend.trade.model.PreditLsdm;
import com.app.backend.trade.model.SignalType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalDate;
import java.util.*;

/**
 * Trainer supervisé pour le modèle d'allocation.
 * Construit un dataset à partir de l'historique des signaux (table signal_lstm) et des prix (daily_value).
 */
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
            logger.warn("Dataset vide => retour modèle non entraîné.");
            return new PortfolioAllocationModel(1, config); // placeholder
        }
        int inputSize = featureRows.get(0).length;
        PortfolioAllocationModel model = new PortfolioAllocationModel(inputSize, config);
        // Préparation INDArray (concat)
        int n = featureRows.size();
        double[][] X = new double[n][inputSize];
        double[][] y = new double[n][1];
        for (int i = 0; i < n; i++) {
            X[i] = featureRows.get(i);
            y[i][0] = targetRows.get(i);
        }
        org.nd4j.linalg.api.ndarray.INDArray inX = org.nd4j.linalg.factory.Nd4j.create(X);
        org.nd4j.linalg.api.ndarray.INDArray inY = org.nd4j.linalg.factory.Nd4j.create(y);
        for (int epoch = 0; epoch < config.getEpochs(); epoch++) {
            model.getNetwork().fit(inX, inY);
        }
        logger.info("Entraînement terminé: {} lignes, inputSize={} ", n, inputSize);
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
        // Map date -> close pour daily_value
        String sqlPrices = "SELECT date, close FROM daily_value WHERE symbol = ? ORDER BY date ASC";
        Map<LocalDate, Double> closeMap = new HashMap<>();
        jdbcTemplate.query(sqlPrices, ps -> ps.setString(1, symbol), rs -> {
            while (rs.next()) {
                try { closeMap.put(rs.getDate("date").toLocalDate(), Double.valueOf(rs.getString("close"))); } catch (Exception ignored) {}
            }
        });
        // Rolling stats pour volatilité simple (std 20 derniers rendements)
        List<Double> prevCloses = new ArrayList<>();
        for (Map<String, Object> row : signals) {
            LocalDate d = ((java.sql.Date) row.get("lstm_created_at")).toLocalDate();
            Double closeToday = closeMap.get(d);
            if (closeToday == null) continue;
            // Future window
            double futureRet = computeFutureReturn(closeMap, d, config.getLookaheadBars());
            if (Double.isNaN(futureRet)) continue;
            // Rendement direct pour features
            if (!prevCloses.isEmpty()) {
                double lastClosePrev = prevCloses.get(prevCloses.size() - 1);
            }
            prevCloses.add(closeToday);
            double vol20 = computeVolatility(prevCloses, 20);
            String signalStr = (String) row.get("signal_lstm");
            SignalType sig;
            try { sig = SignalType.valueOf(signalStr); } catch (Exception e) { sig = SignalType.NONE; }
            double signalBuy = sig == SignalType.BUY ? 1.0 : 0.0;
            double signalSell = sig == SignalType.SELL ? 1.0 : 0.0;
            Double pricePred = asDouble(row.get("price_lstm"));
            Double priceClose = asDouble(row.get("price_clo"));
            double deltaPred = (pricePred != null && priceClose != null && priceClose != 0) ? (pricePred - priceClose) / priceClose : 0.0;
            // Metrics modèle (dernière ligne)
            String sqlMetric = "SELECT profit_factor, win_rate, max_drawdown, business_score, total_trades FROM trade_ai.lstm_models WHERE symbol = ? ORDER BY updated_date DESC LIMIT 1";
            Map<String, Object> metric = null;
            try { metric = jdbcTemplate.queryForMap(sqlMetric, symbol); } catch (Exception ignored) {}
            double profitFactor = safeMetric(metric, "profit_factor");
            double winRate = safeMetric(metric, "win_rate");
            double maxDD = safeMetric(metric, "max_drawdown");
            double businessScore = safeMetric(metric, "business_score");
            double totalTrades = safeMetric(metric, "total_trades");
            double[] features = new double[]{signalBuy, signalSell, deltaPred, vol20, profitFactor, winRate, maxDD, businessScore, totalTrades};
            featureRows.add(features);
            targetRows.add(futureRet);
        }
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

