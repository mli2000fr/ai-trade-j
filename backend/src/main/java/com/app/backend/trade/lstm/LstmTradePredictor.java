/**
 * Service de prédiction LSTM pour le trading.
 * <p>
 * Permet d'entraîner, valider, sauvegarder, charger et utiliser un modèle LSTM
 * pour prédire la prochaine clôture, le delta ou la classe (hausse/baisse/stable)
 * d'une série de données financières.
 * Les hyperparamètres sont configurés dynamiquement via {@link LstmConfig}.
 * </p>
 */

package com.app.backend.trade.lstm;

import com.app.backend.trade.model.PreditLsdm;
import com.app.backend.trade.model.SignalType;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.dao.EmptyResultDataAccessException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.ta4j.core.*;
import org.ta4j.core.indicators.*;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.indicators.helpers.HighPriceIndicator;
import org.ta4j.core.indicators.helpers.LowPriceIndicator;
import org.ta4j.core.indicators.helpers.VolumeIndicator;
import org.ta4j.core.indicators.statistics.StandardDeviationIndicator;

import java.io.*;
import java.time.DayOfWeek;
import java.time.ZonedDateTime;
import java.util.*;
import java.util.Arrays;

@Service
public class LstmTradePredictor {
    private static final Logger logger = LoggerFactory.getLogger(LstmTradePredictor.class);
    private final LstmHyperparamsRepository hyperparamsRepository;
    private final JdbcTemplate jdbcTemplate;

    public LstmTradePredictor(LstmHyperparamsRepository hyperparamsRepository, JdbcTemplate jdbcTemplate) {
        this.hyperparamsRepository = hyperparamsRepository;
        this.jdbcTemplate = jdbcTemplate;
    }

    /* ===================== INIT MODELE ===================== */
    public MultiLayerNetwork initModel(int inputSize, int outputSize, int lstmNeurons, double dropoutRate, double learningRate, String optimizer, double l1, double l2, LstmConfig config, boolean classification) {
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.updater(
            "adam".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.Adam(learningRate)
                : "rmsprop".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.RmsProp(learningRate)
                : new org.nd4j.linalg.learning.config.Sgd(learningRate));
        builder.l1(l1).l2(l2);
        builder.trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED);

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();
        int nLayers = config != null ? config.getNumLstmLayers() : 1;
        boolean bidir = config != null && config.isBidirectional();
        boolean attention = config != null && config.isAttention();

        for (int i = 0; i < nLayers; i++) {
            int inSize = (i == 0) ? inputSize : (bidir ? lstmNeurons * 2 : lstmNeurons);
            LSTM.Builder lstmBuilder = new LSTM.Builder()
                .nOut(lstmNeurons)
                .activation(Activation.TANH);
            org.deeplearning4j.nn.conf.layers.Layer recurrent = bidir ? new org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional(lstmBuilder.build()) : lstmBuilder.build();
            if (i == nLayers - 1) {
                // LastTimeStep : remplace le pooling temporel (équivalent à prendre le dernier pas)
                listBuilder.layer(new LastTimeStep(recurrent));
            } else {
                listBuilder.layer(recurrent);
                if (dropoutRate > 0.0) {
                    listBuilder.layer(new DropoutLayer.Builder().dropOut(dropoutRate).build());
                }
            }
        }
        if (attention) {
            listBuilder.layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                .nIn(lstmNeurons * (bidir ? 2 : 1))
                .nOut(lstmNeurons * (bidir ? 2 : 1))
                .activation(Activation.SOFTMAX)
                .build());
        }
        int finalRecurrentSize = lstmNeurons * (bidir ? 2 : 1);
        int denseOut = Math.max(16, lstmNeurons / 4);
        listBuilder.layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
            .nIn(finalRecurrentSize)
            .nOut(denseOut)
            .activation(Activation.RELU)
            .build());


        Activation outAct; LossFunctions.LossFunction outLoss;
        if (outputSize == 1 && !classification) { outAct = Activation.IDENTITY; outLoss = LossFunctions.LossFunction.MSE; }
        else if (classification) { outAct = Activation.SOFTMAX; outLoss = LossFunctions.LossFunction.MCXENT; }
        else { outAct = Activation.IDENTITY; outLoss = LossFunctions.LossFunction.MSE; }

        listBuilder.layer(new OutputLayer.Builder(outLoss)
            .nIn(denseOut)
            .nOut(outputSize)
            .activation(outAct)
            .build());

        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    private MultiLayerNetwork ensureModelWindowSize(MultiLayerNetwork model, int numFeatures, LstmConfig config) {
        if (model == null) {
            logger.info("Initialisation modèle LSTM nIn={}", numFeatures);
            return initModel(numFeatures, 1, config.getLstmNeurons(), config.getDropoutRate(), config.getLearningRate(), config.getOptimizer(), config.getL1(), config.getL2(), config, false);
        }
        return model;
    }

    /* ===================== EXTRACTION FEATURES ===================== */
    public double[][] extractFeatureMatrix(BarSeries series, List<String> features) {
        int n = series.getBarCount();
        int fCount = features.size();
        double[][] M = new double[n][fCount];
        if (n == 0) return M;
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        HighPriceIndicator high = new HighPriceIndicator(series);
        LowPriceIndicator low = new LowPriceIndicator(series);
        VolumeIndicator vol = new VolumeIndicator(series);

        // Pré-instanciation conditionnelle
        RSIIndicator rsi = features.contains("rsi") ? new RSIIndicator(close, 14) : null;
        SMAIndicator sma14 = features.contains("sma") ? new SMAIndicator(close, 14) : null;
        EMAIndicator ema14 = features.contains("ema") ? new EMAIndicator(close, 14) : null;
        MACDIndicator macd = features.contains("macd") ? new MACDIndicator(close, 12, 26) : null;
        ATRIndicator atr = features.contains("atr") ? new ATRIndicator(series, 14) : null;
        StochasticOscillatorKIndicator stoch = features.contains("stochastic") ? new StochasticOscillatorKIndicator(series, 14) : null;
        CCIIndicator cci = features.contains("cci") ? new CCIIndicator(series, 20) : null;
        StandardDeviationIndicator sd20 = (features.contains("bollinger_high") || features.contains("bollinger_low")) ? new StandardDeviationIndicator(close, 20) : null;
        SMAIndicator sma20 = (features.contains("bollinger_high") || features.contains("bollinger_low")) ? new SMAIndicator(close, 20) : null;
        boolean needMomentum = features.contains("momentum");

        for (int i = 0; i < n; i++) {
            double closeVal = close.getValue(i).doubleValue();
            double highVal = high.getValue(i).doubleValue();
            double lowVal = low.getValue(i).doubleValue();
            java.time.ZonedDateTime t = series.getBar(i).getEndTime();
            for (int f = 0; f < fCount; f++) {
                String feat = features.get(f);
                double val = 0.0;
                switch (feat) {
                    case "close" -> val = closeVal;
                    case "volume" -> val = vol.getValue(i).doubleValue();
                    case "rsi" -> val = rsi != null ? rsi.getValue(i).doubleValue() : 0.0;
                    case "sma" -> val = sma14 != null ? sma14.getValue(i).doubleValue() : 0.0;
                    case "ema" -> val = ema14 != null ? ema14.getValue(i).doubleValue() : 0.0;
                    case "macd" -> val = macd != null ? macd.getValue(i).doubleValue() : 0.0;
                    case "atr" -> val = atr != null ? atr.getValue(i).doubleValue() : 0.0;
                    case "stochastic" -> val = stoch != null ? stoch.getValue(i).doubleValue() : 0.0;
                    case "cci" -> val = cci != null ? cci.getValue(i).doubleValue() : 0.0;
                    case "momentum" -> val = (needMomentum && i >= 10) ? (close.getValue(i).doubleValue() - close.getValue(i-10).doubleValue()) : 0.0;
                    case "bollinger_high" -> {
                        if (sma20 != null && sd20 != null) {
                            double mid = sma20.getValue(i).doubleValue();
                            double sdv = sd20.getValue(i).doubleValue();
                            val = mid + 2 * sdv;
                        } else val = 0.0;
                    }
                    case "bollinger_low" -> {
                        if (sma20 != null && sd20 != null) {
                            double mid = sma20.getValue(i).doubleValue();
                            double sdv = sd20.getValue(i).doubleValue();
                            val = mid - 2 * sdv;
                        } else val = 0.0;
                    }
                    case "day_of_week" -> val = t.getDayOfWeek().getValue();
                    case "month" -> val = t.getMonthValue();
                    default -> val = closeVal;
                }
                if (Double.isNaN(val) || Double.isInfinite(val)) val = 0.0;
                M[i][f] = val;
            }
        }
        return M;
    }

    /* ===================== UTIL SHAPES ===================== */
    public double[][][] transposeTimeFeature(double[][][] seq) {
        // input: [batch][time][features] => output [batch][features][time]
        int b = seq.length;
        if (b == 0) return seq;
        int t = seq[0].length;
        int f = seq[0][0].length;
        double[][][] out = new double[b][f][t];
        for (int i = 0; i < b; i++) {
            for (int ti = 0; ti < t; ti++) {
                for (int fi = 0; fi < f; fi++) {
                    out[i][fi][ti] = seq[i][ti][fi];
                }
            }
        }
        return out;
    }

    public org.nd4j.linalg.api.ndarray.INDArray toINDArray(double[][][] sequences) {
        return Nd4j.create(sequences);
    }

    /* ===================== NORMALISATION / SCALERS ===================== */
    public static class FeatureScaler implements Serializable {
        public enum Type { MINMAX, ZSCORE }
        public double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY, mean = 0.0, std = 0.0; public Type type;
        public FeatureScaler(Type type){ this.type = type; }
        public void fit(double[] data){
            if (type == Type.MINMAX) {
                for (double v : data) { if (v < min) min = v; if (v > max) max = v; }
                if (min == Double.POSITIVE_INFINITY) { min = 0; max = 1; }
            } else {
                double s = 0; for (double v : data) s += v; mean = data.length > 0 ? s / data.length : 0;
                double var = 0; for (double v : data) var += (v - mean) * (v - mean); std = data.length > 0 ? Math.sqrt(var / data.length) : 1.0;
                if (std == 0) std = 1.0;
            }
        }
        public double[] transform(double[] data){
            double[] out = new double[data.length];
            if (type == Type.MINMAX) {
                double range = (max - min) == 0 ? 1e-9 : (max - min);
                for (int i = 0; i < data.length; i++) out[i] = (data[i] - min) / range;
            } else {
                for (int i = 0; i < data.length; i++) out[i] = (data[i] - mean) / (std == 0 ? 1e-9 : std);
            }
            return out;
        }
        public double inverse(double v){ return type == Type.MINMAX ? min + v * (max - min) : mean + v * std; }
    }
    public static class ScalerSet implements Serializable {
        public Map<String, FeatureScaler> featureScalers = new HashMap<>();
        public FeatureScaler labelScaler;
    }

    /* ===================== ENTRAINEMENT ===================== */
    public static class TrainResult { public MultiLayerNetwork model; public ScalerSet scalers; public TrainResult(MultiLayerNetwork m, ScalerSet s){this.model=m;this.scalers=s;} }

    public double[] extractCloseValues(BarSeries series) {
        double[] closes = new double[series.getBarCount()];
        for (int i = 0; i < series.getBarCount(); i++) closes[i] = series.getBar(i).getClosePrice().doubleValue();
        return closes;
    }

    // Détermine le type de normalisation adapté à une feature
    public String getFeatureNormalizationType(String feature) {
        return switch (feature) {
            case "rsi", "momentum", "stochastic", "cci", "macd" -> "zscore";
            default -> "minmax";
        };
    }

    public TrainResult trainLstmScalarV2(BarSeries series, LstmConfig config, Object unused) {
        List<String> features = config.getFeatures();
        int windowSize = config.getWindowSize();
        int numFeatures = features.size();
        int barCount = series.getBarCount();
        if (barCount <= windowSize + 1) return new TrainResult(null, null);
        double[][] matrix = extractFeatureMatrix(series, features);
        double[] closes = extractCloseValues(series);
        int numSeq = barCount - windowSize - 1;
        double[][][] inputSeq = new double[numSeq][windowSize][numFeatures]; // [batch][time][features]
        double[] labelSeq = new double[numSeq];
        for (int i = 0; i < numSeq; i++) {
            for (int j = 0; j < windowSize; j++) {
                System.arraycopy(matrix[i + j], 0, inputSeq[i][j], 0, numFeatures);
            }
            if (config.isUseLogReturnTarget()) {
                double prev = closes[i + windowSize - 1];
                double next = closes[i + windowSize];
                labelSeq[i] = Math.log(next / prev);
            } else {
                labelSeq[i] = closes[i + windowSize];
            }
        }
        ScalerSet scalers = new ScalerSet();
        for (int f = 0; f < numFeatures; f++) {
            double[] col = new double[numSeq + windowSize];
            for (int i = 0; i < numSeq + windowSize; i++) col[i] = matrix[i][f];
            FeatureScaler.Type type = getFeatureNormalizationType(features.get(f)).equals("zscore") ? FeatureScaler.Type.ZSCORE : FeatureScaler.Type.MINMAX;
            FeatureScaler scaler = new FeatureScaler(type); scaler.fit(col); scalers.featureScalers.put(features.get(f), scaler);
        }
        FeatureScaler labelScaler = new FeatureScaler(FeatureScaler.Type.MINMAX); labelScaler.fit(labelSeq); scalers.labelScaler = labelScaler;
        double[][][] normSeq = new double[numSeq][windowSize][numFeatures];
        for (int i = 0; i < numSeq; i++) {
            for (int j = 0; j < windowSize; j++) {
                for (int f = 0; f < numFeatures; f++) {
                    normSeq[i][j][f] = scalers.featureScalers.get(features.get(f)).transform(new double[]{inputSeq[i][j][f]})[0];
                }
            }
        }
        // Création INDArray puis permutation vers [batch, features, time]
        org.nd4j.linalg.api.ndarray.INDArray X = Nd4j.create(normSeq); // [batch, time, features]
        X = X.permute(0, 2, 1).dup('c'); // maintenant [batch, features, time]
        if (X.size(1) != numFeatures || X.size(2) != windowSize) {
            logger.warn("[SHAPE][TRAIN] Incohérence shape après permute: expected features={} time={} got features={} time={}", numFeatures, windowSize, X.size(1), X.size(2));
        }
        double[] normLabels = scalers.labelScaler.transform(labelSeq);
        org.nd4j.linalg.api.ndarray.INDArray y = Nd4j.create(normLabels, new long[]{numSeq, 1});

        int effectiveFeatures = (int) X.size(1);
        if (effectiveFeatures != numFeatures) {
            logger.warn("[INIT][ADAPT] numFeatures déclaré={} mais tensor features={} => reconstruction modèle", numFeatures, effectiveFeatures);
        }
        MultiLayerNetwork model = initModel(effectiveFeatures, 1, config.getLstmNeurons(), config.getDropoutRate(), config.getLearningRate(), config.getOptimizer(), config.getL1(), config.getL2(), config, false);
        logger.debug("[TRAIN] X shape={} (batch={} features={} time={}) y shape={} expectedFeatures={} lstmNeurons={}", Arrays.toString(X.shape()), X.size(0), X.size(1), X.size(2), Arrays.toString(y.shape()), numFeatures, config.getLstmNeurons());
        org.nd4j.linalg.dataset.DataSet ds = new org.nd4j.linalg.dataset.DataSet(X, y);
        org.nd4j.linalg.dataset.api.iterator.DataSetIterator iterator = new ListDataSetIterator<>(ds.asList(), config.getBatchSize());
        model.fit(iterator, config.getNumEpochs());
        return new TrainResult(model, scalers);
    }

    /* ===================== PREDICTION ===================== */
    public double predictNextCloseScalarV2(BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers) {
        List<String> features = config.getFeatures();
        int windowSize = config.getWindowSize();
        if (series.getBarCount() <= windowSize) throw new IllegalArgumentException("Pas assez de barres");
        double[][] matrix = extractFeatureMatrix(series, features);
        int numFeatures = features.size();
        double[][] normMatrix = new double[matrix.length][numFeatures];
        for (int f = 0; f < numFeatures; f++) {
            double[] col = new double[matrix.length];
            for (int i = 0; i < matrix.length; i++) col[i] = matrix[i][f];
            double[] normCol = scalers.featureScalers.get(features.get(f)).transform(col);
            for (int i = 0; i < matrix.length; i++) normMatrix[i][f] = normCol[i];
        }
        double[][][] seq = new double[1][windowSize][numFeatures]; // [1][time][features]
        for (int j = 0; j < windowSize; j++) System.arraycopy(normMatrix[normMatrix.length - windowSize + j], 0, seq[0][j], 0, numFeatures);
        org.nd4j.linalg.api.ndarray.INDArray input = Nd4j.create(seq).permute(0, 2, 1).dup('c'); // [1, features, time]
        if (input.size(1) != numFeatures || input.size(2) != windowSize) {
            logger.warn("[SHAPE][PRED] Incohérence shape input: expected features={} time={} got features={} time={}", numFeatures, windowSize, input.size(1), input.size(2));
        }
        double predNorm = model.output(input).getDouble(0);
        double predTarget = scalers.labelScaler.inverse(predNorm);
        double lastClose = series.getLastBar().getClosePrice().doubleValue();
        double predicted = config.isUseLogReturnTarget() ? lastClose * Math.exp(predTarget) : predTarget;
        double limitPct = config.getLimitPredictionPct();
        if (limitPct > 0) {
            double min = lastClose * (1 - limitPct), max = lastClose * (1 + limitPct);
            if (predicted < min) predicted = min; else if (predicted > max) predicted = max;
        }
        return predicted;
    }

    public double predictNextCloseWithScalerSet(String symbol, BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers){
        return predictNextCloseScalarV2(series, config, model, scalers);
    }

    /* ===================== WALK FORWARD ===================== */
    public static class TradingMetricsV2 implements Serializable { public double totalProfit, profitFactor, winRate, maxDrawdownPct, expectancy, sharpe, sortino, exposure, turnover, avgBarsInPosition, mse, businessScore, calmar; public int numTrades; }
    public static class WalkForwardResultV2 implements Serializable { public List<TradingMetricsV2> splits = new ArrayList<>(); public double meanMse, meanBusinessScore, mseVariance, mseInterModelVariance; }

    public WalkForwardResultV2 walkForwardEvaluate(BarSeries series, LstmConfig config) {
        WalkForwardResultV2 result = new WalkForwardResultV2();
        int splits = Math.max(2, config.getWalkForwardSplits());
        int windowSize = config.getWindowSize();
        int totalBars = series.getBarCount();
        if (totalBars < windowSize + 50) return result;
        int splitSize = (totalBars - windowSize) / splits;
        double sumMse = 0, sumBusiness = 0; int mseCount = 0, businessCount = 0; List<Double> mseList = new ArrayList<>();
        for (int s = 1; s <= splits; s++) {
            int testEndBar = (s == splits) ? totalBars : windowSize + s * splitSize;
            int testStartBar = windowSize + (s - 1) * splitSize + config.getEmbargoBars();
            if (testStartBar + windowSize + 5 >= testEndBar) continue;
            BarSeries trainSeries = series.getSubSeries(0, testStartBar);
            TrainResult tr = trainLstmScalarV2(trainSeries, config, null); if (tr.model == null) continue;
            TradingMetricsV2 metrics = simulateTradingWalkForward(series, trainSeries.getBarCount(), testStartBar, testEndBar, tr.model, tr.scalers, config);
            metrics.mse = computeSplitMse(series, testStartBar, testEndBar, tr.model, tr.scalers, config);
            double pfAdj = Math.min(metrics.profitFactor, config.getBusinessProfitFactorCap());
            double expPos = Math.max(metrics.expectancy, 0.0);
            double denom = 1.0 + Math.pow(Math.max(metrics.maxDrawdownPct, 0.0), config.getBusinessDrawdownGamma());
            metrics.businessScore = (expPos * pfAdj * metrics.winRate) / (denom + 1e-9);
            result.splits.add(metrics);
            if (Double.isFinite(metrics.mse)) { sumMse += metrics.mse; mseCount++; mseList.add(metrics.mse); }
            if (Double.isFinite(metrics.businessScore)) { sumBusiness += metrics.businessScore; businessCount++; }
        }
        result.meanMse = mseCount > 0 ? sumMse / mseCount : Double.NaN;
        result.meanBusinessScore = businessCount > 0 ? sumBusiness / businessCount : Double.NaN;
        if (mseList.size() > 1) {
            double mean = result.meanMse; double var = mseList.stream().mapToDouble(m -> (m - mean) * (m - mean)).sum() / mseList.size();
            result.mseVariance = var; result.mseInterModelVariance = var; // simple alias
        }
        return result;
    }

    public TradingMetricsV2 simulateTradingWalkForward(BarSeries fullSeries, int trainBarCount, int testStartBar, int testEndBar, MultiLayerNetwork model, ScalerSet scalers, LstmConfig config) {
        TradingMetricsV2 tm = new TradingMetricsV2();
        double equity = 0, peak = 0, trough = 0; boolean inPos = false, longPos = false; double entry = 0; int barsInPos = 0; int horizon = Math.max(3, config.getHorizonBars());
        List<Double> tradePnL = new ArrayList<>(); List<Double> tradeReturns = new ArrayList<>(); List<Integer> barsInPosList = new ArrayList<>();
        double timeInPos = 0; int positionChanges = 0; double capital = config.getCapital(); double riskPct = config.getRiskPct(); double sizingK = config.getSizingK();
        double feePct = config.getFeePct(); double slippagePct = config.getSlippagePct(); double positionSize = 0; double entrySpread = 0;
        for (int bar = testStartBar; bar < testEndBar - 1; bar++) {
            BarSeries sub = fullSeries.getSubSeries(0, bar + 1);
            if (sub.getBarCount() <= config.getWindowSize()) continue;
            double threshold = computeSwingTradeThreshold(sub, config);
            ATRIndicator atrInd = new ATRIndicator(sub, 14);
            double atr = atrInd.getValue(sub.getEndIndex()).doubleValue();
            if (!inPos) {
                double predicted = predictNextCloseScalarV2(sub, config, model, scalers);
                double lastClose = sub.getLastBar().getClosePrice().doubleValue();
                double up = lastClose * (1 + threshold); double down = lastClose * (1 - threshold);
                if (predicted > up || predicted < down) {
                    inPos = true; longPos = predicted > up; entry = lastClose; barsInPos = 0; positionChanges++;
                    positionSize = atr > 0 ? capital * riskPct / (atr * sizingK) : 0.0;
                    entrySpread = computeMeanSpread(sub);
                }
            } else {
                barsInPos++; timeInPos++;
                double current = fullSeries.getBar(bar).getClosePrice().doubleValue();
                double stop = entry * (1 - threshold * (longPos ? 1 : -1));
                double target = entry * (1 + 2 * threshold * (longPos ? 1 : -1));
                boolean exit = false; double pnl = 0;
                if (longPos) {
                    if (current <= stop || current >= target) { pnl = (current - entry) * positionSize; exit = true; }
                } else {
                    if (current >= stop || current <= target) { pnl = (entry - current) * positionSize; exit = true; }
                }
                if (!exit && barsInPos >= horizon) { pnl = longPos ? (current - entry) * positionSize : (entry - current) * positionSize; exit = true; }
                if (exit) {
                    double cost = entrySpread * positionSize + slippagePct * entry * positionSize + feePct * entry * positionSize;
                    pnl -= cost; tradePnL.add(pnl); tradeReturns.add(pnl / (entry * Math.max(positionSize, 1e-9))); barsInPosList.add(barsInPos);
                    equity += pnl; if (equity > peak) { peak = equity; trough = equity; } else if (equity < trough) trough = equity;
                    inPos = false; longPos = false; entry = 0; barsInPos = 0; positionSize = 0; entrySpread = 0;
                }
            }
        }
        double gains = 0, losses = 0; int win = 0, loss = 0; for (double p : tradePnL) { if (p > 0) { gains += p; win++; } else if (p < 0) { losses += p; loss++; } }
        tm.totalProfit = gains + losses; tm.numTrades = tradePnL.size(); tm.profitFactor = losses != 0 ? gains / Math.abs(losses) : (gains > 0 ? Double.POSITIVE_INFINITY : 0);
        tm.winRate = tm.numTrades > 0 ? (double) win / tm.numTrades : 0.0; double ddAbs = peak - trough; tm.maxDrawdownPct = peak != 0 ? ddAbs / Math.abs(peak) : 0.0;
        double avgGain = win > 0 ? gains / win : 0, avgLoss = loss > 0 ? Math.abs(losses) / loss : 0; tm.expectancy = (win + loss) > 0 ? (tm.winRate * avgGain - (1 - tm.winRate) * avgLoss) : 0;
        double meanRet = tradeReturns.stream().mapToDouble(d -> d).average().orElse(0); double stdRet = Math.sqrt(tradeReturns.stream().mapToDouble(d -> { double m = d - meanRet; return m * m; }).average().orElse(0));
        tm.sharpe = stdRet > 0 ? meanRet / stdRet * Math.sqrt(Math.max(1, tradeReturns.size())) : 0; double downsideStd = Math.sqrt(tradeReturns.stream().filter(r -> r < 0).mapToDouble(r -> { double dr = r - meanRet; return dr * dr; }).average().orElse(0));
        tm.sortino = downsideStd > 0 ? meanRet / downsideStd : 0; tm.exposure = 0; tm.turnover = 0; tm.avgBarsInPosition = barsInPosList.stream().mapToInt(i -> i).average().orElse(0);
        double capitalBase = config.getCapital() > 0 ? config.getCapital() : 1.0; tm.calmar = tm.maxDrawdownPct > 0 ? ((tm.totalProfit / capitalBase) / tm.maxDrawdownPct) : 0.0;
        return tm;
    }

    public double computeSplitMse(BarSeries series, int testStartBar, int testEndBar, MultiLayerNetwork model, ScalerSet scalers, LstmConfig config) {
        int window = config.getWindowSize();
        double se = 0; int count = 0;
        double[] closes = extractCloseValues(series);
        for (int t = testStartBar; t < testEndBar; t++) {
            if (t - window < 1) continue; // besoin t-1 pour log-return
            BarSeries sub = series.getSubSeries(0, t); // exclut t (label à t)
            double pred = predictNextCloseScalarV2(sub, config, model, scalers);
            double actual = config.isUseLogReturnTarget() ? Math.log(closes[t] / closes[t - 1]) : closes[t];
            double err = pred - actual; se += err * err; count++;
        }
        return count > 0 ? se / count : Double.NaN;
    }

    /* ===================== DRIFT DETECTION ===================== */
    public static class DriftDetectionResult { public boolean drift; public String driftType; public double kl; public double meanShift; }
    public static class DriftReportEntry { public java.time.Instant eventDate; public String symbol; public String feature; public String driftType; public double kl; public double meanShift; public double mseBefore; public double mseAfter; public boolean retrained; }

    public boolean checkDriftForFeature(String feat, double[] values, FeatureScaler scaler, double klThreshold, double meanShiftSigma) {
        return checkDriftForFeatureDetailed(feat, values, scaler, klThreshold, meanShiftSigma).drift;
    }

    public DriftDetectionResult checkDriftForFeatureDetailed(String feat, double[] values, FeatureScaler scaler, double klThreshold, double meanShiftSigma) {
        DriftDetectionResult r = new DriftDetectionResult();
        int n = values.length; if (n < 40) return r;
        int half = n / 2; double[] past = Arrays.copyOfRange(values, 0, half); double[] recent = Arrays.copyOfRange(values, half, n);
        double meanPast = Arrays.stream(past).average().orElse(0); double meanRecent = Arrays.stream(recent).average().orElse(0);
        double varPast = Arrays.stream(past).map(v -> (v - meanPast) * (v - meanPast)).sum() / past.length;
        double varRecent = Arrays.stream(recent).map(v -> (v - meanRecent) * (v - meanRecent)).sum() / recent.length;
        double stdPast = Math.sqrt(varPast + 1e-9); r.meanShift = (meanRecent - meanPast) / stdPast;
        r.kl = approximateSymmetricKl(past, recent, 20);
        if (Math.abs(r.meanShift) > meanShiftSigma) { r.drift = true; r.driftType = "mean_shift"; }
        if (r.kl > klThreshold) { r.drift = true; r.driftType = (r.driftType == null? "kl" : r.driftType+"+kl"); }
        return r;
    }

    private double approximateSymmetricKl(double[] a, double[] b, int bins) {
        double min = Math.min(Arrays.stream(a).min().orElse(0), Arrays.stream(b).min().orElse(0));
        double max = Math.max(Arrays.stream(a).max().orElse(1), Arrays.stream(b).max().orElse(1));
        if (max - min == 0) return 0;
        double[] ha = new double[bins]; double[] hb = new double[bins]; double w = (max - min) / bins;
        for (double v : a) { int idx = (int) Math.floor((v - min) / w); if (idx < 0) idx = 0; else if (idx >= bins) idx = bins - 1; ha[idx]++; }
        for (double v : b) { int idx = (int) Math.floor((v - min) / w); if (idx < 0) idx = 0; else if (idx >= bins) idx = bins - 1; hb[idx]++; }
        double sumA = Arrays.stream(ha).sum(); double sumB = Arrays.stream(hb).sum();
        double kl1 = 0, kl2 = 0; for (int i = 0; i < bins; i++) { double pa = (ha[i] + 1e-9) / sumA; double pb = (hb[i] + 1e-9) / sumB; kl1 += pa * Math.log(pa / pb); kl2 += pb * Math.log(pb / pa); }
        return 0.5 * (kl1 + kl2);
    }

    public java.util.List<DriftReportEntry> checkDriftAndRetrainWithReport(BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers, String symbol) {
        List<DriftReportEntry> reports = new ArrayList<>();
        if (model == null || scalers == null) return reports;
        double mseBefore = Double.NaN; double mseAfter = Double.NaN;
        try { int total = series.getBarCount(); int testStart = Math.max(0, total - (config.getWindowSize() * 3)); mseBefore = computeSplitMse(series, testStart, total, model, scalers, config); } catch (Exception ignored) {}
        boolean retrain = false;
        for (String feat : config.getFeatures()) {
            FeatureScaler sc = scalers.featureScalers.get(feat); if (sc == null) continue;
            double[] vals = new double[series.getBarCount()];
            for (int i = 0; i < series.getBarCount(); i++) vals[i] = extractFeatureMatrix(series, Collections.singletonList(feat))[i][0];
            DriftDetectionResult res = checkDriftForFeatureDetailed(feat, vals, sc, config.getKlDriftThreshold(), config.getMeanShiftSigmaThreshold());
            if (res.drift) retrain = true;
            DriftReportEntry entry = new DriftReportEntry(); entry.eventDate = java.time.Instant.now(); entry.symbol = symbol; entry.feature = feat; entry.driftType = res.driftType; entry.kl = res.kl; entry.meanShift = res.meanShift; entry.mseBefore = mseBefore; entry.retrained = false; reports.add(entry);
        }
        if (retrain) {
            TrainResult tr = trainLstmScalarV2(series, config, null);
            if (tr.model != null) {
                model.setParams(tr.model.params()); scalers.featureScalers = tr.scalers.featureScalers; scalers.labelScaler = tr.scalers.labelScaler;
                try { int total = series.getBarCount(); int testStart = Math.max(0, total - (config.getWindowSize() * 3)); mseAfter = computeSplitMse(series, testStart, total, model, scalers, config); } catch (Exception ignored) {}
                for (DriftReportEntry r : reports) { r.retrained = true; r.mseAfter = mseAfter; }
            }
        }
        return reports;
    }

    public boolean checkDriftAndRetrain(BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers) {
        return !checkDriftAndRetrainWithReport(series, config, model, scalers, "").isEmpty();
    }

    /* ===================== THRESHOLD & SPREAD ===================== */
    public double computeSwingTradeThreshold(BarSeries series, LstmConfig config) {
        double k = config.getThresholdK(); String type = config.getThresholdType();
        if ("ATR".equalsIgnoreCase(type)) {
            ATRIndicator atr = new ATRIndicator(series, 14);
            double lastATR = atr.getValue(series.getEndIndex()).doubleValue();
            double lastClose = series.getLastBar().getClosePrice().doubleValue();
            double th = k * lastATR / (lastClose == 0 ? 1 : lastClose);
            logger.info("[SEUIL SWING] ATR%={}", th); return th;
        } else if ("returns".equalsIgnoreCase(type)) {
            double[] closes = extractCloseValues(series); if (closes.length < 3) return 0;
            double[] logRet = new double[closes.length - 1]; for (int i = 1; i < closes.length; i++) logRet[i - 1] = Math.log(closes[i] / closes[i - 1]);
            double mean = Arrays.stream(logRet).average().orElse(0); double std = Math.sqrt(Arrays.stream(logRet).map(r -> (r - mean) * (r - mean)).sum() / logRet.length);
            return k * std;
        }
        return 0.01 * k; // fallback simple
    }

    public double computeMeanSpread(BarSeries series) {
        int n = series.getBarCount(); if (n == 0) return 0;
        double sum = 0; for (int i = 0; i < n; i++) sum += (series.getBar(i).getHighPrice().doubleValue() - series.getBar(i).getLowPrice().doubleValue());
        return sum / n;
    }

    /* ===================== PREDIT ===================== */
    public PreditLsdm getPredit(String symbol, BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers) {
        model = ensureModelWindowSize(model, config.getFeatures().size(), config);
        double th = computeSwingTradeThreshold(series, config);
        double predicted = predictNextCloseWithScalerSet(symbol, series, config, model, scalers);
        predicted = Math.round(predicted * 1000.0) / 1000.0;
        double[] closes = extractCloseValues(series); double lastClose = closes[closes.length - 1];
        double delta = predicted - lastClose; SignalType signal = delta > th ? SignalType.UP : (delta < -th ? SignalType.DOWN : SignalType.STABLE);
        String position = analyzePredictionPosition(Arrays.copyOfRange(closes, closes.length - config.getWindowSize(), closes.length), predicted);
        String formattedDate = series.getLastBar().getEndTime().format(java.time.format.DateTimeFormatter.ofPattern("dd/MM"));
        logger.info("PREDICT win={} last={} pred={} delta={} thr={} signal={}", config.getWindowSize(), lastClose, predicted, delta, th, signal);
        return PreditLsdm.builder().lastClose(lastClose).predictedClose(predicted).signal(signal).lastDate(formattedDate).position(position).build();
    }

    public String analyzePredictionPosition(double[] lastWindow, double predicted) {
        double min = Arrays.stream(lastWindow).min().orElse(Double.NaN); double max = Arrays.stream(lastWindow).max().orElse(Double.NaN);
        if (predicted > max) return "au-dessus"; if (predicted < min) return "en-dessous"; return "dans la plage"; }

    /* ===================== PERSISTENCE ===================== */
    public void saveModelToDb(String symbol, MultiLayerNetwork model, JdbcTemplate jdbcTemplate, LstmConfig config, ScalerSet scalers) throws IOException {
        if (model == null) return;
        ByteArrayOutputStream baos = new ByteArrayOutputStream(); ModelSerializer.writeModel(model, baos, true); byte[] modelBytes = baos.toByteArray();
        hyperparamsRepository.saveHyperparams(symbol, config);
        LstmHyperparameters params = new LstmHyperparameters(config.getWindowSize(), config.getLstmNeurons(), config.getDropoutRate(), config.getLearningRate(), config.getNumEpochs(), config.getPatience(), config.getMinDelta(), config.getOptimizer(), config.getL1(), config.getL2(), config.getNormalizationScope(), config.getNormalizationMethod());
        com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        String hyperparamsJson = mapper.writeValueAsString(params); String scalersJson = mapper.writeValueAsString(scalers);
        String sql = "REPLACE INTO lstm_models (symbol, model_blob, hyperparams_json, normalization_scope, scalers_json, updated_date) VALUES (?,?,?,?,?, CURRENT_TIMESTAMP)";
        jdbcTemplate.update(sql, symbol, modelBytes, hyperparamsJson, config.getNormalizationScope(), scalersJson);
    }

    public MultiLayerNetwork loadModelFromDb(String symbol, JdbcTemplate jdbcTemplate) throws IOException {
        LstmConfig config = hyperparamsRepository.loadHyperparams(symbol);
        if (config == null) throw new IOException("Aucun hyperparamètre pour " + symbol);
        String sql = "SELECT model_blob, normalization_scope FROM lstm_models WHERE symbol = ?";
        try {
            Map<String, Object> result = jdbcTemplate.queryForMap(sql, symbol);
            byte[] modelBytes = (byte[]) result.get("model_blob");
            if (modelBytes != null) {
                ByteArrayInputStream bais = new ByteArrayInputStream(modelBytes);
                MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(bais);
                logger.info("Modèle chargé {}", symbol); return model;
            }
        } catch (EmptyResultDataAccessException e) { throw new IOException("Modèle non trouvé"); }
        return null;
    }

    // Chargement modèle + scalers (compatibilité ancienne API)
    public LoadedModel loadModelAndScalersFromDb(String symbol, JdbcTemplate jdbcTemplate) throws IOException {
        LstmConfig config = hyperparamsRepository.loadHyperparams(symbol);
        if (config == null) throw new IOException("Aucun hyperparamètre pour " + symbol);
        String sql = "SELECT model_blob, normalization_scope, scalers_json FROM lstm_models WHERE symbol = ?";
        MultiLayerNetwork model = null; ScalerSet scalers = null;
        try {
            Map<String,Object> result = jdbcTemplate.queryForMap(sql, symbol);
            byte[] modelBlob = (byte[]) result.get("model_blob");
            String scalersJson = (String) result.get("scalers_json");
            if (modelBlob != null) {
                try (ByteArrayInputStream bais = new ByteArrayInputStream(modelBlob)) {
                    model = ModelSerializer.restoreMultiLayerNetwork(bais);
                }
            }
            if (scalersJson != null && !scalersJson.isBlank()) {
                try {
                    com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                    scalers = mapper.readValue(scalersJson, ScalerSet.class);
                } catch (Exception e) {
                    logger.warn("Impossible de parser scalers_json : {}", e.getMessage());
                }
            }
            logger.info("Chargé modèle+scalers pour {} (scalers={})", symbol, scalers!=null);
        } catch (EmptyResultDataAccessException e) {
            throw new IOException("Modèle non trouvé");
        }
        return new LoadedModel(model, scalers);
    }

    public static class LoadedModel { public MultiLayerNetwork model; public ScalerSet scalers; public LoadedModel(MultiLayerNetwork m, ScalerSet s){this.model=m;this.scalers=s;} }

    /* ===================== SEEDS ===================== */
    public void setGlobalSeeds(long seed){
        Nd4j.getRandom().setSeed(seed);
        org.deeplearning4j.nn.api.OptimizationAlgorithm.valueOf("STOCHASTIC_GRADIENT_DESCENT"); // force load classes
        logger.debug("Seeds fixés seed={}", seed);
    }
}
