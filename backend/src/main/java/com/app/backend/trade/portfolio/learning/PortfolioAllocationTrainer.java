package com.app.backend.trade.portfolio.learning;

import com.app.backend.trade.model.SignalType;
import com.app.backend.trade.controller.LstmHelper;
import com.app.backend.trade.util.TradeUtils;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
    private final PortfolioAllocationModelRepository repository;
    // Ajout: accès aux prédictions LSTM datées pour backfill
    private final LstmHelper lstmHelper;

    public PortfolioAllocationTrainer(JdbcTemplate jdbcTemplate, PortfolioLearningConfig config,
                                      PortfolioAllocationModelRepository repository,
                                      LstmHelper lstmHelper) {
        this.jdbcTemplate = jdbcTemplate;
        this.config = config;
        this.repository = repository;
        this.lstmHelper = lstmHelper;
    }

    /**
     * Entraîne un modèle avec validation walk-forward et early stopping; sélectionne le meilleur split.
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
        int n = featureRows.size();
        if (n == 0) {
            logger.warn("Dataset vide => modèle placeholder.");
            PortfolioAllocationModel placeholder = new PortfolioAllocationModel(1, config);
            // Sauvegarde du placeholder pour traçabilité
            try {
                String algoVersion = String.format("portfolio-mlp-%dx%d", config.getHidden1(), config.getHidden2());
                String notes = "auto-save (dataset vide)";
                repository.saveModel(tri, algoVersion, notes, placeholder);
            } catch (Exception e) {
                logger.warn("Échec sauvegarde modèle placeholder: {}", e.getMessage());
            }
            return placeholder;
        }
        int inputSize = featureRows.get(0).length;

        // Créer splits walk-forward
        List<int[]> splits = createWalkForwardSplits(n, Math.max(1, config.getWalkForwardSplits()));
        double bestValLoss = Double.POSITIVE_INFINITY;
        PortfolioAllocationModel bestModel = null;

        for (int s = 0; s < splits.size(); s++) {
            int[] split = splits.get(s);
            int trainEnd = split[0];
            int valEnd = split[1];
            if (trainEnd <= 1 || valEnd - trainEnd < 1) continue;
            double[][] Xtrain = new double[trainEnd][inputSize];
            double[][] ytrain = new double[trainEnd][1];
            for (int i = 0; i < trainEnd; i++) { Xtrain[i] = featureRows.get(i); ytrain[i][0] = targetRows.get(i); }
            double[][] Xval = new double[valEnd - trainEnd][inputSize];
            double[][] yval = new double[valEnd - trainEnd][1];
            for (int i = trainEnd; i < valEnd; i++) { Xval[i - trainEnd] = featureRows.get(i); yval[i - trainEnd][0] = targetRows.get(i); }

            // Normalisation par split (stats calculées sur train uniquement)
            double[][] stats = config.isNormalizeFeatures() ? computeMeansStds(Xtrain) : null;
            if (stats != null) {
                applyNormalizationInPlace(Xtrain, stats[0], stats[1]);
                applyNormalizationInPlace(Xval, stats[0], stats[1]);
            }

            // Préparer réseau et early stopping
            PortfolioAllocationModel model = new PortfolioAllocationModel(inputSize, config);
            if (stats != null) { model.setFeatureMeans(stats[0]); model.setFeatureStds(stats[1]); }
            org.nd4j.linalg.api.ndarray.INDArray inX = org.nd4j.linalg.factory.Nd4j.create(Xtrain);
            org.nd4j.linalg.api.ndarray.INDArray inY = org.nd4j.linalg.factory.Nd4j.create(ytrain);
            org.nd4j.linalg.api.ndarray.INDArray valX = org.nd4j.linalg.factory.Nd4j.create(Xval);
            org.nd4j.linalg.api.ndarray.INDArray valY = org.nd4j.linalg.factory.Nd4j.create(yval);

            double bestSplitVal = Double.POSITIVE_INFINITY;
            int patienceLeft = config.getPatienceEs();
            org.nd4j.linalg.api.ndarray.INDArray bestParams = model.getNetwork().params().dup();

            for (int epoch = 0; epoch < config.getEpochs(); epoch++) {
                model.getNetwork().fit(inX, inY);
                double valLoss = customValLoss(model, valX, valY);
                if (valLoss + config.getMinDeltaEs() < bestSplitVal) {
                    bestSplitVal = valLoss;
                    patienceLeft = config.getPatienceEs();
                    bestParams.assign(model.getNetwork().params());
                } else {
                    patienceLeft--;
                    if (patienceLeft <= 0) {
                        break; // early stop
                    }
                }
            }
            // Restaurer meilleurs paramètres du split
            model.getNetwork().setParams(bestParams);

            // Évaluer val finale (custom loss)
            double finalVal = customValLoss(model, valX, valY);
            logger.info("Split {}: valLoss={} (best={})", s+1, finalVal, bestSplitVal);
            if (finalVal < bestValLoss) {
                bestValLoss = finalVal;
                bestModel = model;
            }
        }

        if (bestModel == null) {
            // fallback: entraînement simple sur tout
            double[][] X = new double[n][inputSize];
            double[][] y = new double[n][1];
            for (int i = 0; i < n; i++) { X[i] = featureRows.get(i); y[i][0] = targetRows.get(i); }
            PortfolioAllocationModel model = new PortfolioAllocationModel(inputSize, config);
            if (config.isNormalizeFeatures()) {
                double[][] stats = computeMeansStds(X);
                model.setFeatureMeans(stats[0]);
                model.setFeatureStds(stats[1]);
                applyNormalizationInPlace(X, stats[0], stats[1]);
            }
            org.nd4j.linalg.api.ndarray.INDArray inX = org.nd4j.linalg.factory.Nd4j.create(X);
            org.nd4j.linalg.api.ndarray.INDArray inY = org.nd4j.linalg.factory.Nd4j.create(y);
            for (int epoch = 0; epoch < config.getEpochs(); epoch++) { model.getNetwork().fit(inX, inY); }
            bestModel = model;
            logger.warn("Walk-forward non applicable, fallback entraînement global.");
        }
        logger.info("Sélection modèle avec valLoss={}.", bestValLoss);

        // Sauvegarde du meilleur modèle
        try {
            String algoVersion = String.format("portfolio-mlp-%dx%d", config.getHidden1(), config.getHidden2());
            String notes = "auto-save (post-train)";
            repository.saveModel(tri, algoVersion, notes, bestModel);
            logger.info("Modèle sauvegardé pour tri={} version={}", tri, algoVersion);
        } catch (Exception e) {
            logger.error("Échec sauvegarde du modèle: {}", e.getMessage());
        }

        return bestModel;
    }

    private List<int[]> createWalkForwardSplits(int total, int k) {
        List<int[]> splits = new ArrayList<>();
        if (total < 10 || k < 1) return splits;
        for (int i = 1; i <= k; i++) {
            int trainEnd = (int) Math.floor((double) total * i / (k + 1));
            int valEnd = (int) Math.floor((double) total * (i + 1) / (k + 1));
            if (trainEnd >= 1 && valEnd > trainEnd) {
                splits.add(new int[]{trainEnd, valEnd});
            }
        }
        return splits;
    }

    private double customValLoss(PortfolioAllocationModel model,
                                 org.nd4j.linalg.api.ndarray.INDArray X,
                                 org.nd4j.linalg.api.ndarray.INDArray y) {
        org.nd4j.linalg.api.ndarray.INDArray pred = model.getNetwork().output(X, false); // shape [n,1]
        return model.computeCustomLoss(y, pred);
    }


    /**
     * Construit les lignes features + cible pour un symbole donné.
     * Cible = rendement futur moyen sur lookaheadBars (différence close / close présent).
     */
    private void buildSymbolDataset(String symbol, String tri, List<double[]> featureRows, List<Double> targetRows) {
        String sqlSignals = "SELECT lstm_created_at, signal_lstm, price_lstm, price_clo FROM signal_lstm WHERE symbol = ? AND tri = ? ORDER BY lstm_created_at ASC";
        List<java.util.Map<String, Object>> signals = jdbcTemplate.queryForList(sqlSignals, symbol, tri);

        int minBars = config.getMinHistoryBars();
        if (signals.size() < minBars) {
            try {
                // 1) Construire l'ensemble des dates trading récentes cibles
                java.util.Set<java.time.LocalDate> existingDates = new java.util.HashSet<>();
                for (java.util.Map<String, Object> r : signals) {
                    Object d = r.get("lstm_created_at");
                    if (d instanceof java.sql.Date) {
                        existingDates.add(((java.sql.Date) d).toLocalDate());
                    } else if (d instanceof java.util.Date) {
                        existingDates.add(new java.sql.Date(((java.util.Date) d).getTime()).toLocalDate());
                    }
                }
                java.util.List<java.time.LocalDate> candidateDays = new java.util.ArrayList<>();
                java.time.LocalDate ref = TradeUtils.getLastTradingDayBefore(java.time.LocalDate.now());
                // Générer au plus minBars derniers jours de bourse
                while (candidateDays.size() < minBars) {
                    if (candidateDays.isEmpty() || !ref.isEqual(candidateDays.get(candidateDays.size() - 1))) {
                        candidateDays.add(ref);
                    }
                    // reculer d'au moins 1 jour pour éviter boucle infinie sur jours non ouvrés
                    ref = TradeUtils.getLastTradingDayBefore(ref.minusDays(1));
                }

                // 2) Appeler getPreditAtDate pour les dates manquantes seulement
                int toInsert = minBars - signals.size();
                int inserted = 0;
                for (java.time.LocalDate d : candidateDays) {
                    if (inserted >= toInsert) break;
                    if (existingDates.contains(d)) continue;
                    try {
                        lstmHelper.getPreditAtDate(symbol, tri, d);
                        inserted++;
                    } catch (Exception ex) {
                        logger.warn("Backfill échoué {}@{}: {}", symbol, d, ex.getMessage());
                    }
                }

                // 3) Relire les signaux pour poursuivre le traitement
                signals = jdbcTemplate.queryForList(sqlSignals, symbol, tri);
                if (signals.size() < minBars) {
                    logger.warn("Backfill incomplet pour {} tri={} ({}/{})", symbol, tri, signals.size(), minBars);
                }
            } catch (Exception e) {
                logger.warn("Backfill des signaux impossible pour {}: {}", symbol, e.getMessage());
            }
        }

        if (signals.size() < minBars) return; // garde-fou si backfill insuffisant

        // Préparer séquence des closes (trading bars séquentielles)
        List<Double> closes = new ArrayList<>();
        for (java.util.Map<String, Object> row : signals) {
            Double c = asDouble(row.get("price_clo"));
            if (c != null && c > 0) closes.add(c); else closes.add(Double.NaN);
        }
        // Metrics modèle (une seule récupération)
        String sqlMetric = "SELECT profit_factor, win_rate, max_drawdown, business_score, total_trades FROM trade_ai.lstm_models WHERE symbol = ? ORDER BY rendement DESC LIMIT 1";
        java.util.Map<String, Object> metric = null;
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
            java.util.Map<String,Object> row = signals.get(i);
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
        List<java.util.Map<String, Object>> recent = jdbcTemplate.queryForList(sqlLast, symbol, tri);
        if (recent.isEmpty()) return null;
        // Reconstituer closes (ordre DESC actuellement -> inverser pour volatilité cohérente)
        java.util.Collections.reverse(recent);
        List<Double> closes = new ArrayList<>();
        for (java.util.Map<String,Object> r : recent) {
            Double c = asDouble(r.get("price_clo"));
            if (c != null && c > 0) closes.add(c);
        }
        java.util.Map<String,Object> last = recent.get(recent.size()-1);
        String sqlMetric = "SELECT profit_factor, win_rate, max_drawdown, business_score, total_trades FROM trade_ai.lstm_models WHERE symbol = ? ORDER BY updated_date DESC LIMIT 1";
        java.util.Map<String, Object> metric = null;
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
        return f;
    }

    private double[][] computeMeansStds(double[][] X) {
        int rows = X.length, cols = X[0].length;
        double[] means = new double[cols];
        double[] stds = new double[cols];
        for (int j = 0; j < cols; j++) {
            double sum = 0; for (int i = 0; i < rows; i++) sum += X[i][j];
            double mean = sum / rows; means[j] = mean;
            double var = 0; for (int i = 0; i < rows; i++) { double d = X[i][j]-mean; var += d*d; }
            var /= Math.max(1, rows-1);
            stds[j] = var <= 0 ? 1.0 : Math.sqrt(var);
        }
        return new double[][]{means, stds};
    }

    private void applyNormalizationInPlace(double[][] X, double[] means, double[] stds) {
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[i].length; j++) {
                double std = stds[j] == 0 ? 1.0 : stds[j];
                X[i][j] = (X[i][j] - means[j]) / std;
            }
        }
    }

    private void normalizeInPlace(double[][] X) { /* deprecated, remplacée par computeMeansStds+applyNormalizationInPlace */ }

    private double[] normalizeSingle(double[] f) { return f; } // non utilisé désormais

    private double computeVolatilityRolling(List<Double> closes, int idx, int window) {
        if (idx < window) return 0.0;
        List<Double> slice = closes.subList(idx - window, idx);
        return computeVolatility(slice, window);
    }

    private double computeFutureReturn(java.util.Map<java.time.LocalDate, Double> closeMap, java.time.LocalDate start, int horizon) {
        Double startPrice = closeMap.get(start);
        if (startPrice == null) return Double.NaN;
        java.time.LocalDate futureDate = start.plusDays(horizon);
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

    private double safeMetric(java.util.Map<String, Object> metric, String key) {
        if (metric == null || !metric.containsKey(key) || metric.get(key) == null) return 0.0;
        try { return Double.parseDouble(metric.get(key).toString()); } catch (Exception e) { return 0.0; }
    }

    private Double asDouble(Object o) {
        if (o == null) return null;
        try { return Double.parseDouble(o.toString()); } catch (Exception e) { return null; }
    }
}
