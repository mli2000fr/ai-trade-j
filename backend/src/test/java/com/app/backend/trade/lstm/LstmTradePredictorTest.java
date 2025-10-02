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

        // Mock du modèle LSTM pour retourner une valeur fixe (log-return = 0.05)
        MultiLayerNetwork model = Mockito.mock(MultiLayerNetwork.class);
        INDArray fakeOutput = Nd4j.create(new double[]{0.05});
        Mockito.when(model.output(Mockito.any(INDArray.class))).thenReturn(fakeOutput);

        // Création d'un scaler identité (pas de normalisation)
        LstmTradePredictor.ScalerSet scalers = new LstmTradePredictor.ScalerSet();
        scalers.featureScalers = Collections.singletonMap("close", new IdentityScaler());
        scalers.labelScaler = new IdentityScaler();

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

        // Calcul attendu : dernier close = 110, log-return = 0.05 => 110 * exp(0.05)
        double expected = 110 * Math.exp(0.05);
        // Limite supérieure = 110 * 1.2 = 132, donc pas de clipping
        assertEquals(expected, predicted, 1e-6);
    }

    // Implémentation d'un scaler identité pour le test
    static class IdentityScaler extends LstmTradePredictor.FeatureScaler {
        public IdentityScaler() { super(Type.MINMAX); }
        @Override
        public double[] transform(double[] values) { return values; }
        @Override
        public double inverse(double value) { return value; }
    }
}
