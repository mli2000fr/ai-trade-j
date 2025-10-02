package com.app.backend.trade.lstm;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.ta4j.core.BarSeries;
import org.ta4j.core.BaseBarSeriesBuilder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.time.ZonedDateTime;
import java.util.Collections;
import static org.junit.jupiter.api.Assertions.*;

public class LstmTradePredictorTest {
    @Test
    public void testPredictNextCloseScalarV2_LogReturn() {
        // Création d'une série temporelle simple
        BarSeries series = new BaseBarSeriesBuilder().withName("test").build();
        ZonedDateTime now = ZonedDateTime.now();
        series.addBar(now.minusMinutes(5), 100d, 105d, 95d, 100d, 1000d);
        series.addBar(now.minusMinutes(4), 102d, 107d, 101d, 102d, 1000d);
        series.addBar(now.minusMinutes(3), 104d, 108d, 103d, 104d, 1000d);
        series.addBar(now.minusMinutes(2), 106d, 110d, 105d, 106d, 1000d);
        series.addBar(now.minusMinutes(1), 108d, 112d, 107d, 108d, 1000d);
        series.addBar(now, 110d, 115d, 109d, 110d, 1000d);

        // Mock du modèle LSTM pour retourner une valeur fixe (log-return normalisé = 0.05)
        MultiLayerNetwork model = Mockito.mock(MultiLayerNetwork.class);
        INDArray fakeOutput = Nd4j.create(new double[]{0.05});
        Mockito.when(model.output(Mockito.any(INDArray.class))).thenReturn(fakeOutput);

        // Scalers: feature identité + label ZScore identité (mean=0,std=1) => pas de migration
        LstmTradePredictor.ScalerSet scalers = new LstmTradePredictor.ScalerSet();
        scalers.featureScalers = Collections.singletonMap("close", new IdentityScaler());
        scalers.labelScaler = new ZScoreIdentityScaler();

        // Configuration LSTM
        LstmConfig config = new LstmConfig();
        config.setFeatures(Collections.singletonList("close"));
        config.setWindowSize(3);
        config.setUseLogReturnTarget(true);
        config.setLimitPredictionPct(0.2); // Limite à ±20%

        // Mock des dépendances du constructeur
        LstmHyperparamsRepository repo = Mockito.mock(LstmHyperparamsRepository.class);
        org.springframework.jdbc.core.JdbcTemplate jdbc = Mockito.mock(org.springframework.jdbc.core.JdbcTemplate.class);
        LstmTradePredictor predictor = new LstmTradePredictor(repo, jdbc);
        double predicted = predictor.predictNextCloseScalarV2(series, config, model, scalers);

        double expected = 110 * Math.exp(0.05); // log-return direct car mean=0,std=1
        assertEquals(expected, predicted, 1e-6);
    }

    @Test
    public void testLabelScalerMigrationMinMaxToZScore() {
        // Série avec variations de log-return pour éviter std=0
        BarSeries series = new BaseBarSeriesBuilder().withName("test-mig").build();
        ZonedDateTime now = ZonedDateTime.now();
        // 10 barres avec variations non uniformes
        series.addBar(now.minusMinutes(9), 100d, 101d, 99d, 100d, 1000d);
        series.addBar(now.minusMinutes(8), 101d, 102d, 100d, 101d, 1000d);
        series.addBar(now.minusMinutes(7), 102d, 103d, 101d, 102d, 1000d);
        series.addBar(now.minusMinutes(6), 104d, 105d, 103d, 104d, 1000d);
        series.addBar(now.minusMinutes(5), 103d, 104d, 102d, 103d, 1000d);
        series.addBar(now.minusMinutes(4), 107d, 108d, 106d, 107d, 1000d);
        series.addBar(now.minusMinutes(3), 108d, 109d, 107d, 108d, 1000d);
        series.addBar(now.minusMinutes(2), 109d, 110d, 108d, 109d, 1000d);
        series.addBar(now.minusMinutes(1), 113d, 114d, 112d, 113d, 1000d);
        series.addBar(now,                112d, 113d, 111d, 112d, 1000d);

        // Config log-return + fenêtre
        LstmConfig config = new LstmConfig();
        config.setFeatures(Collections.singletonList("close"));
        config.setWindowSize(3);
        config.setUseLogReturnTarget(true);
        config.setUseMultiHorizonAvg(false);

        // Scalers factices avec labelScaler MINMAX (ancien format à migrer)
        LstmTradePredictor.ScalerSet scalers = new LstmTradePredictor.ScalerSet();
        IdentityScaler id = new IdentityScaler();
        scalers.featureScalers = Collections.singletonMap("close", id);
        LstmTradePredictor.FeatureScaler oldLabel = new LstmTradePredictor.FeatureScaler(LstmTradePredictor.FeatureScaler.Type.MINMAX);
        // Simule ancien fit
        oldLabel.min = -0.01; oldLabel.max = 0.02; // plage étroite => compression
        scalers.labelScaler = oldLabel;

        // Modèle mock
        MultiLayerNetwork model = Mockito.mock(MultiLayerNetwork.class);
        // Sortie normalisée arbitraire (ne sert pas au test de migration lui-même)
        Mockito.when(model.output(Mockito.any(INDArray.class))).thenReturn(Nd4j.create(new double[]{0.0}));

        LstmHyperparamsRepository repo = Mockito.mock(LstmHyperparamsRepository.class);
        org.springframework.jdbc.core.JdbcTemplate jdbc = Mockito.mock(org.springframework.jdbc.core.JdbcTemplate.class);
        LstmTradePredictor predictor = new LstmTradePredictor(repo, jdbc);

        // Appel prédiction -> doit déclencher migration vers ZSCORE
        predictor.predictNextCloseScalarV2(series, config, model, scalers);

        assertNotNull(scalers.labelScaler, "Label scaler doit être présent après migration");
        assertEquals(LstmTradePredictor.FeatureScaler.Type.ZSCORE, scalers.labelScaler.type, "Le label scaler doit être converti en ZSCORE");

        // Recalcule manuel des log-returns pour vérifier std ≈ 1 après transform
        double[] closes = new double[series.getBarCount()];
        for (int i = 0; i < closes.length; i++) closes[i] = series.getBar(i).getClosePrice().doubleValue();
        int windowSize = config.getWindowSize();
        int numSeq = closes.length - windowSize - 1; if (numSeq < 1) numSeq = closes.length - 1;
        double[] labelSeq = new double[Math.max(0, numSeq)];
        for (int i = 0; i < labelSeq.length; i++) {
            double prev = closes[i + windowSize - 1];
            double next = closes[i + windowSize];
            labelSeq[i] = Math.log(next / prev);
        }
        double[] norm = scalers.labelScaler.transform(labelSeq);
        double mean = 0, var = 0; int n = norm.length; for(double v: norm) mean += v; mean = n>0? mean/n:0; for(double v: norm) var += (v-mean)*(v-mean); var = n>0? var/n:0; double std = Math.sqrt(var);
        assertTrue(std > 1e-3, "Std normalisée ne doit pas être < 1e-3 (plateau)");
        assertTrue(Math.abs(std - 1.0) < 1e-3, "Std après ZScore doit être ≈ 1 (std="+std+")");
    }

    // Implémentation d'un scaler identité pour le test
    static class IdentityScaler extends LstmTradePredictor.FeatureScaler {
        public IdentityScaler() { super(Type.MINMAX); }
        @Override public double[] transform(double[] values) { return values; }
        @Override public double inverse(double value) { return value; }
    }
    static class ZScoreIdentityScaler extends LstmTradePredictor.FeatureScaler {
        public ZScoreIdentityScaler() { super(Type.ZSCORE); this.mean = 0.0; this.std = 1.0; }
        @Override public void fit(double[] data) { /* no-op */ }
        @Override public double[] transform(double[] values) { return values; }
        @Override public double inverse(double value) { return value; }
    }
}
