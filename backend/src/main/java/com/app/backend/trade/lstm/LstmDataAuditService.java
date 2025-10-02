package com.app.backend.trade.lstm;

import com.app.backend.trade.controller.LstmHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.ta4j.core.BarSeries;

import java.util.*;

public class LstmDataAuditService {
    private static final Logger logger = LoggerFactory.getLogger(LstmDataAuditService.class);

    private final LstmHelper lstmHelper;

    public LstmDataAuditService(LstmHelper lstmHelper) {
        this.lstmHelper = lstmHelper;
    }

    /**
     * Audit des données pour tous les symboles filtrés.
     * @param windowSize taille de la fenêtre d'entrée
     * @param horizonBars horizon de prédiction
     * @return liste des symboles valides
     */
    public List<String> auditAllSymbols(int windowSize, int horizonBars) {
        List<String> symbols = lstmHelper.getSymbolFitredFromTabSingle("score_swing_trade");
        List<String> validSymbols = new ArrayList<>();
        int minBars = windowSize + horizonBars + 400;
        for (String symbol : symbols) {
            BarSeries series = lstmHelper.getBarBySymbol(symbol, null);
            int barCount = series.getBarCount();
            if (barCount < minBars) {
                logger.warn("[AUDIT] Symbol {} exclu: barCount={} < minBars={}", symbol, barCount, minBars);
                continue;
            }
            // Calcul des log-returns
            List<Double> logReturns = new ArrayList<>();
            for (int i = 1; i < barCount; i++) {
                double prev = series.getBar(i-1).getClosePrice().doubleValue();
                double curr = series.getBar(i).getClosePrice().doubleValue();
                if (prev > 0 && curr > 0) {
                    logReturns.add(Math.log(curr / prev));
                } else {
                    logReturns.add(Double.NaN);
                }
            }
            // Stats
            double mean = logReturns.stream().filter(d -> !d.isNaN()).mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
            double std = Math.sqrt(logReturns.stream().filter(d -> !d.isNaN()).mapToDouble(d -> Math.pow(d - mean, 2)).average().orElse(0));
            long zeroCount = logReturns.stream().filter(d -> !Double.isNaN(d) && d == 0.0).count();
            long nanCount = logReturns.stream().filter(d -> Double.isNaN(d)).count();
            double propZero = zeroCount / (double) logReturns.size();
            double propNaN = nanCount / (double) logReturns.size();
            // Histogramme (20 bins)
            double min = logReturns.stream().filter(d -> !Double.isNaN(d)).mapToDouble(Double::doubleValue).min().orElse(0);
            double max = logReturns.stream().filter(d -> !Double.isNaN(d)).mapToDouble(Double::doubleValue).max().orElse(0);
            int[] hist = new int[20];
            for (double v : logReturns) {
                if (Double.isNaN(v)) continue;
                int bin = (int) ((v - min) / (max - min + 1e-12) * 20);
                if (bin < 0) bin = 0;
                if (bin > 19) bin = 19;
                hist[bin]++;
            }
            // Logging
            logger.info("[AUDIT] {}: bars={}, mean={}, std={}, propZero={}, propNaN={}, var={}",
                    symbol, barCount, String.format("%.5f", mean), String.format("%.5f", std), String.format("%.4f", propZero), String.format("%.4f", propNaN), String.format("%.5e", std*std));
            logger.info("[AUDIT] {}: histogram (20 bins) {}", symbol, Arrays.toString(hist));
            // Critère d'acceptation
            if (std*std < 1e-5) {
                logger.warn("[AUDIT] Symbol {} exclu: variance log-return < 1e-5", symbol);
                continue;
            }
            validSymbols.add(symbol);
        }
        logger.info("[AUDIT] Symboles valides: {}", validSymbols);
        return validSymbols;
    }
}
