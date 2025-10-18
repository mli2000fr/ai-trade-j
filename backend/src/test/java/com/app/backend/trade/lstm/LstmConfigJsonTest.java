// ...existing code...
package com.app.backend.trade.lstm;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class LstmConfigJsonTest {

    @Test
    public void testJsonRoundtripAllFields() throws Exception {
        // Création et alimentation d'une instance source avec des valeurs factices
        LstmConfig src = new LstmConfig();
        src.setIndexTop(7);
        src.setWindowSize(42);
        src.setLstmNeurons(123);
        src.setDropoutRate(0.33);
        src.setEnableGradientClipping(true);
        src.setGradientClippingThreshold(0.9);
        src.setLearningRate(0.005);
        src.setNumEpochs(11);
        src.setPatience(6);
        src.setMinDelta(0.00005);
        src.setKFolds(4);
        src.setOptimizer("rmsprop");
        src.setL1(0.001);
        src.setL2(0.002);
        src.setNormalizationScope("global");
        // éviter le mode 'auto' pour ne pas être affecté par la méthode getNormalizationMethod()
        src.setNormalizationMethod("minmax");
        src.setSwingTradeType("range");
        List<String> features = Arrays.asList("f_close","f_vol","f_open");
        src.setFeatures(features);
        src.setNumLstmLayers(4);
        src.setBidirectional(true);
        src.setAttention(true);
        src.setUseMultiHorizonAvg(false);
        src.setHorizonBars(10);
        src.setThresholdType("returns");
        src.setThresholdK(2.5);
        src.setLimitPredictionPct(0.12);
        src.setBatchSize(16);
        src.setCvMode("timeseries");
        src.setUseScalarV2(true);
        src.setUseLogReturnTarget(false);
        src.setUseWalkForwardV2(false);
        src.setWalkForwardSplits(5);
        src.setEmbargoBars(2);
        src.setSeed(424242L);
        src.setBusinessProfitFactorCap(4.5);
        src.setBusinessDrawdownGamma(2.2);
        src.setCapital(50000.0);
        src.setStopLossPct(0.07);
        src.setTakeProfitPct(0.15);
        src.setRiskPct(0.02);
        src.setSizingK(1.5);
        src.setFeePct(0.001);
        src.setSlippagePct(0.0003);
        src.setKlDriftThreshold(0.2);
        src.setMeanShiftSigmaThreshold(1.5);
        src.setEntryThresholdFactor(1.7);
        src.setPatienceVal(9);
        src.setPredictionResidualVarianceMin(1e-5);
        src.setUseAsyncIterator(false);
        src.setAsyncQueueSize(3);
        src.setBaselineReplica(false);
        src.setEntryOrLogic(false);
        src.setEntryPercentileQuantile(0.75);
        src.setEntryDeltaFloor(0.0008);
        src.setVolumeMinRatio(0.7);
        src.setRsiOverboughtLimit(78.0);
        src.setDeadzoneFactor(0.11);
        src.setDisableDeadzone(false);
        src.setThresholdAtrMin(0.0006);
        src.setThresholdAtrMax(0.05);
        src.setAggressivenessBoost(1.4);
        src.setAggressiveFallbackEnabled(false);
        src.setFallbackNoTradeBars(55);
        src.setFallbackMaxExtraBars(120);
        src.setFallbackMinPercentileQuantile(0.25);
        src.setFallbackMinDeltaFloor(0.00005);

        // Sérialisation en JSON
        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writeValueAsString(src);

        // Désérialisation en nouvelle instance
        LstmConfig dst = mapper.readValue(json, LstmConfig.class);

        // Comparaison champ à champ
        assertEquals(src.getIndexTop(), dst.getIndexTop());
        assertEquals(src.getWindowSize(), dst.getWindowSize());
        assertEquals(src.getLstmNeurons(), dst.getLstmNeurons());
        assertEquals(src.getDropoutRate(), dst.getDropoutRate(), 1e-12);
        assertEquals(src.isEnableGradientClipping(), dst.isEnableGradientClipping());
        assertEquals(src.getGradientClippingThreshold(), dst.getGradientClippingThreshold(), 1e-12);
        assertEquals(src.getLearningRate(), dst.getLearningRate(), 1e-12);
        assertEquals(src.getNumEpochs(), dst.getNumEpochs());
        assertEquals(src.getPatience(), dst.getPatience());
        assertEquals(src.getMinDelta(), dst.getMinDelta(), 1e-12);
        assertEquals(src.getKFolds(), dst.getKFolds());
        assertEquals(src.getOptimizer(), dst.getOptimizer());
        assertEquals(src.getL1(), dst.getL1(), 1e-12);
        assertEquals(src.getL2(), dst.getL2(), 1e-12);
        assertEquals(src.getNormalizationScope(), dst.getNormalizationScope());
        // Ici la méthode getNormalizationMethod() est customisée : vérifier la valeur brute du champ
        assertEquals(src.getNormalizationMethod(), dst.getNormalizationMethod());
        assertEquals(src.getSwingTradeType(), dst.getSwingTradeType());
        assertEquals(src.getFeatures(), dst.getFeatures());
        assertEquals(src.getNumLstmLayers(), dst.getNumLstmLayers());
        assertEquals(src.isBidirectional(), dst.isBidirectional());
        assertEquals(src.isAttention(), dst.isAttention());
        assertEquals(src.isUseMultiHorizonAvg(), dst.isUseMultiHorizonAvg());
        assertEquals(src.getHorizonBars(), dst.getHorizonBars());
        assertEquals(src.getThresholdType(), dst.getThresholdType());
        assertEquals(src.getThresholdK(), dst.getThresholdK(), 1e-12);
        assertEquals(src.getLimitPredictionPct(), dst.getLimitPredictionPct(), 1e-12);
        assertEquals(src.getBatchSize(), dst.getBatchSize());
        assertEquals(src.getCvMode(), dst.getCvMode());
        assertEquals(src.isUseScalarV2(), dst.isUseScalarV2());
        assertEquals(src.isUseLogReturnTarget(), dst.isUseLogReturnTarget());
        assertEquals(src.isUseWalkForwardV2(), dst.isUseWalkForwardV2());
        assertEquals(src.getWalkForwardSplits(), dst.getWalkForwardSplits());
        assertEquals(src.getEmbargoBars(), dst.getEmbargoBars());
        assertEquals(src.getSeed(), dst.getSeed());
        assertEquals(src.getBusinessProfitFactorCap(), dst.getBusinessProfitFactorCap(), 1e-12);
        assertEquals(src.getBusinessDrawdownGamma(), dst.getBusinessDrawdownGamma(), 1e-12);
        assertEquals(src.getCapital(), dst.getCapital(), 1e-12);
        assertEquals(src.getStopLossPct(), dst.getStopLossPct(), 1e-12);
        assertEquals(src.getTakeProfitPct(), dst.getTakeProfitPct(), 1e-12);
        assertEquals(src.getRiskPct(), dst.getRiskPct(), 1e-12);
        assertEquals(src.getSizingK(), dst.getSizingK(), 1e-12);
        assertEquals(src.getFeePct(), dst.getFeePct(), 1e-12);
        assertEquals(src.getSlippagePct(), dst.getSlippagePct(), 1e-12);
        assertEquals(src.getKlDriftThreshold(), dst.getKlDriftThreshold(), 1e-12);
        assertEquals(src.getMeanShiftSigmaThreshold(), dst.getMeanShiftSigmaThreshold(), 1e-12);
        assertEquals(src.getEntryThresholdFactor(), dst.getEntryThresholdFactor(), 1e-12);
        assertEquals(src.getPatienceVal(), dst.getPatienceVal());
        assertEquals(src.getPredictionResidualVarianceMin(), dst.getPredictionResidualVarianceMin(), 1e-12);
        assertEquals(src.isUseAsyncIterator(), dst.isUseAsyncIterator());
        assertEquals(src.getAsyncQueueSize(), dst.getAsyncQueueSize());
        assertEquals(src.isBaselineReplica(), dst.isBaselineReplica());
        assertEquals(src.isEntryOrLogic(), dst.isEntryOrLogic());
        assertEquals(src.getEntryPercentileQuantile(), dst.getEntryPercentileQuantile(), 1e-12);
        assertEquals(src.getEntryDeltaFloor(), dst.getEntryDeltaFloor(), 1e-12);
        assertEquals(src.getVolumeMinRatio(), dst.getVolumeMinRatio(), 1e-12);
        assertEquals(src.getRsiOverboughtLimit(), dst.getRsiOverboughtLimit(), 1e-12);
        assertEquals(src.getDeadzoneFactor(), dst.getDeadzoneFactor(), 1e-12);
        assertEquals(src.isDisableDeadzone(), dst.isDisableDeadzone());
        assertEquals(src.getThresholdAtrMin(), dst.getThresholdAtrMin(), 1e-12);
        assertEquals(src.getThresholdAtrMax(), dst.getThresholdAtrMax(), 1e-12);
        assertEquals(src.getAggressivenessBoost(), dst.getAggressivenessBoost(), 1e-12);
        assertEquals(src.isAggressiveFallbackEnabled(), dst.isAggressiveFallbackEnabled());
        assertEquals(src.getFallbackNoTradeBars(), dst.getFallbackNoTradeBars());
        assertEquals(src.getFallbackMaxExtraBars(), dst.getFallbackMaxExtraBars());
        assertEquals(src.getFallbackMinPercentileQuantile(), dst.getFallbackMinPercentileQuantile(), 1e-12);
        assertEquals(src.getFallbackMinDeltaFloor(), dst.getFallbackMinDeltaFloor(), 1e-12);
    }
}

