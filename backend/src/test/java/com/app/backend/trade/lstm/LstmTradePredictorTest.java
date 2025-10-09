package com.app.backend.trade.lstm;

import com.app.backend.trade.model.TradeStylePrediction;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.ta4j.core.Bar;
import org.ta4j.core.BarSeries;
import org.ta4j.core.BaseBarSeriesBuilder;
import org.ta4j.core.num.DecimalNum;

import java.time.ZonedDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class LstmTradePredictorTest {
    private LstmTradePredictor predictor;
    private LstmConfig config;
    private MultiLayerNetwork model;
    private LstmTradePredictor.ScalerSet scalers;

    @BeforeEach
    void setUp() {
        predictor = new LstmTradePredictor(null, null);
        config = new LstmConfig();
        config.setWindowSize(5);
        config.setFeatures(List.of("close", "rsi"));
        config.setThresholdAtrMin(0.001);
        config.setThresholdAtrMax(0.02);
        config.setRsiOverboughtLimit(70);
        config.setVolumeMinRatio(0.5);
        config.setAggressivenessBoost(1.0);
        config.setEntryPercentileQuantile(0.8);
        config.setEntryDeltaFloor(0.001);
        config.setEntryOrLogic(true);
        config.setUseLogReturnTarget(true);
        config.setHorizonBars(1);
        config.setCapital(10000);
        config.setRiskPct(0.01);
        config.setBusinessProfitFactorCap(3.0);
        config.setBusinessDrawdownGamma(1.0);
        // SUPPRESSION: Mock minimal model
        // model = new MultiLayerNetwork(null);
        // model.init();
        // Mock scalers
        scalers = new LstmTradePredictor.ScalerSet();
        scalers.featureScalers.put("close", new LstmTradePredictor.FeatureScaler(LstmTradePredictor.FeatureScaler.Type.MINMAX));
        scalers.featureScalers.put("rsi", new LstmTradePredictor.FeatureScaler(LstmTradePredictor.FeatureScaler.Type.ZSCORE));
        scalers.labelScaler = new LstmTradePredictor.FeatureScaler(LstmTradePredictor.FeatureScaler.Type.ZSCORE);
    }

    private BarSeries createSeries(double[] closes, double[] volumes) {
        BarSeries series = new BaseBarSeriesBuilder().withName("TEST").build();
        ZonedDateTime now = ZonedDateTime.now();
        for (int i = 0; i < closes.length; i++) {
            Bar bar = new org.ta4j.core.BaseBar(
                java.time.Duration.ofMinutes(1),
                now.plusMinutes(i),
                String.valueOf(closes[i]), // open
                String.valueOf(closes[i]), // high
                String.valueOf(closes[i]), // low
                String.valueOf(closes[i]), // close
                String.valueOf(volumes[i]) // volume
            );
            series.addBar(bar);
        }
        return series;
    }

    @Test
    void testInsufficientSeries() {
        BarSeries series = createSeries(new double[]{100, 101, 102}, new double[]{1000, 1000, 1000});
        TradeStylePrediction pred = predictor.predictTradeStyle("TEST", series, config, null, scalers);
        assertEquals("insuffisant", pred.comment);
        assertEquals(pred.lastClose, pred.predictedClose);
        assertEquals("STABLE", pred.tendance);
        assertEquals("HOLD", pred.action);
    }

    @Test
    void testModelAndScalersNull() {
        BarSeries series = createSeries(new double[]{100, 101, 102, 103, 104, 105, 106}, new double[]{1000, 1000, 1000, 1000, 1000, 1000, 1000});
        TradeStylePrediction pred = predictor.predictTradeStyle("TEST", series, config, null, null);
        assertTrue(pred.comment.contains("Modèle ou scalers reconstruits"));
    }

    @Test
    void testPredictionUp() {
        BarSeries series = createSeries(new double[]{100, 101, 102, 103, 104, 110, 115}, new double[]{1000, 1000, 1000, 1000, 1000, 1000, 1000});
        TradeStylePrediction pred = predictor.predictTradeStyle("TEST", series, config, null, scalers);
        assertEquals("UP", pred.tendance);
        assertTrue(pred.action.equals("BUY") || pred.action.equals("HOLD"));
    }

    @Test
    void testPredictionDown() {
        BarSeries series = createSeries(new double[]{120, 119, 118, 117, 116, 110, 105}, new double[]{1000, 1000, 1000, 1000, 1000, 1000, 1000});
        TradeStylePrediction pred = predictor.predictTradeStyle("TEST", series, config, null, scalers);
        assertTrue(pred.tendance.equals("DOWN") || pred.tendance.equals("STABLE"));
        assertTrue(pred.action.equals("SELL") || pred.action.equals("HOLD"));
    }

    @Test
    void testPredictionStable() {
        BarSeries series = createSeries(new double[]{100, 100, 100, 100, 100, 100, 100}, new double[]{1000, 1000, 1000, 1000, 1000, 1000, 1000});
        TradeStylePrediction pred = predictor.predictTradeStyle("TEST", series, config, null, scalers);
        assertEquals("STABLE", pred.tendance);
        assertEquals("HOLD", pred.action);
    }

    @Test
    void testContrarianCase() {
        BarSeries series = createSeries(new double[]{100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90}, new double[]{1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000});
        TradeStylePrediction pred = predictor.predictTradeStyle("TEST", series, config, null, scalers);
        assertTrue(pred.contrarianAdjusted || pred.tendance.equals("DOWN"));
    }

    @Test
    void testPredictionEqualsLastClose() {
        BarSeries series = createSeries(new double[]{100, 101, 102, 103, 104, 105, 105}, new double[]{1000, 1000, 1000, 1000, 1000, 1000, 1000});
        TradeStylePrediction pred = predictor.predictTradeStyle("TEST", series, config, null, scalers);
        assertTrue(pred.comment.contains("identique au dernier close"));
    }

    @Test
    void testExtremeATR() {
        BarSeries series = createSeries(new double[]{100, 120, 80, 130, 70, 140, 60}, new double[]{1000, 1000, 1000, 1000, 1000, 1000, 1000});
        TradeStylePrediction pred = predictor.predictTradeStyle("TEST", series, config, null, scalers);
        assertTrue(pred.atrPct > config.getThresholdAtrMax() || pred.atrPct < config.getThresholdAtrMin());
    }

    @Test
    void testLowVolume() {
        BarSeries series = createSeries(new double[]{100, 101, 102, 103, 104, 105, 106}, new double[]{100, 100, 100, 100, 100, 100, 100});
        TradeStylePrediction pred = predictor.predictTradeStyle("TEST", series, config, null, scalers);
        assertTrue(pred.volumeRatio < 0.7);
    }

    @Test
    void testComplexReturnValues() {
        BarSeries series = createSeries(
                new double[]{100, 105, 110, 120, 130, 140, 150, 160, 170, 180, 190},
                new double[]{1000, 2000, 1500, 3000, 2500, 4000, 3500, 5000, 4500, 6000, 5500}
        );
        TradeStylePrediction pred = predictor.predictTradeStyle("TEST", series, config, null, scalers);
        assertEquals("TEST", pred.symbol);
        assertTrue(pred.lastClose > 0);
        assertTrue(pred.predictedClose > 0);
        assertTrue(pred.deltaPct >= -1 && pred.deltaPct <= 1);
        assertNotNull(pred.tendance);
        assertNotNull(pred.action);
        assertTrue(pred.atrPct >= 0);
        assertTrue(pred.rsi >= 0 && pred.rsi <= 100);
        assertTrue(pred.volumeRatio >= 0);
        assertTrue(pred.thresholdAtrAdaptive >= 0);
        assertTrue(pred.percentileThreshold >= 0 && pred.percentileThreshold <= 1);
        assertTrue(pred.signalStrength >= -1 && pred.signalStrength <= 1);
        assertNotNull(pred.comment);
        // Vérifie les filtres
        assertEquals(pred.rsi > config.getRsiOverboughtLimit() || pred.rsi < 30, pred.rsiFiltered);
        assertEquals(pred.volumeRatio < config.getVolumeMinRatio(), pred.volumeFiltered);
        assertEquals(pred.contrarianReason != null && !pred.contrarianReason.isEmpty(), pred.contrarianAdjusted);
        // Vérifie la logique d'entrée
        assertEquals(config.isEntryOrLogic(), pred.entryLogicOr);
        // Vérifie l'agressivité
        assertEquals(config.getAggressivenessBoost(), pred.aggressivenessBoost);
        // Vérifie la taille de fenêtre
        assertEquals(config.getWindowSize(), pred.windowSize);
    }

    @Test
    void testExtremeSignalStrengthAndPercentile() {
        BarSeries series = createSeries(
                new double[]{100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100},
                new double[]{10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000}
        );
        TradeStylePrediction pred = predictor.predictTradeStyle("TEST", series, config, null, scalers);
        assertTrue(pred.signalStrength > 0.9 || pred.signalStrength < -0.9);
        assertTrue(pred.percentileThreshold > 0.9 || pred.percentileThreshold < 0.1);
    }

    @Test
    void testContrarianReasonFilled() {
        BarSeries series = createSeries(
                new double[]{100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90},
                new double[]{1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000}
        );
        TradeStylePrediction pred = predictor.predictTradeStyle("TEST", series, config, null, scalers);
        if (pred.contrarianAdjusted) {
            assertNotNull(pred.contrarianReason);
            assertFalse(pred.contrarianReason.isEmpty());
        }
    }

    @Test
    void testRsiAndVolumeFilters() {
        BarSeries series = createSeries(
                new double[]{100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110},
                new double[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1} // dernier volume faible pour activer le filtrage
        );
        TradeStylePrediction pred = predictor.predictTradeStyle("TEST", series, config, null, scalers);
        assertTrue(pred.volumeFiltered);
        // Simule RSI extrême
        config.setRsiOverboughtLimit(10);
        TradeStylePrediction pred2 = predictor.predictTradeStyle("TEST", series, config, null, scalers);
        assertTrue(pred2.rsiFiltered);
    }

    @Test
    void testModelReconstructionComment() {
        // Cas où le modèle ou les scalers sont nuls
        BarSeries series = createSeries(
                new double[]{100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110},
                new double[]{1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000}
        );
        TradeStylePrediction pred = predictor.predictTradeStyle("TEST", series, config, null, null);
        String comment = pred.comment != null ? pred.comment.toLowerCase() : "";
        assertTrue(
            comment.contains("modèle") &&
            (comment.contains("reconstruit") || comment.contains("initialisation") || comment.contains("null"))
        );
    }
}
