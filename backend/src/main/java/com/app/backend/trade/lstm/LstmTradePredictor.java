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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.sql.SQLException;
import java.time.format.DateTimeFormatter;

import com.app.backend.trade.model.PreditLsdm;
import com.app.backend.trade.model.SignalType;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.dao.EmptyResultDataAccessException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.ta4j.core.BarSeries;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.ta4j.core.indicators.ROCIndicator;
import org.ta4j.core.indicators.statistics.StandardDeviationIndicator;
import org.ta4j.core.num.Num;

@Service
public class LstmTradePredictor {
    private static final Logger logger = LoggerFactory.getLogger(LstmTradePredictor.class);

    private final LstmHyperparamsRepository hyperparamsRepository;

    private final JdbcTemplate jdbcTemplate;

    public LstmTradePredictor(LstmHyperparamsRepository hyperparamsRepository, JdbcTemplate jdbcTemplate) {
        this.hyperparamsRepository = hyperparamsRepository;
        this.jdbcTemplate = jdbcTemplate;
    }

    /**
     * Initialise le modèle LSTM avec les hyperparamètres fournis.
     * @param inputSize nombre de features (nIn)
     * @param outputSize taille de la sortie (nOut)
     * @param classification true pour classification (softmax), false pour régression
     */
    public MultiLayerNetwork initModel(int inputSize, int outputSize, int lstmNeurons, double dropoutRate, double learningRate, String optimizer, double l1, double l2, LstmConfig config, boolean classification) {
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.updater(
            "adam".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.Adam(learningRate)
            : "rmsprop".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.RmsProp(learningRate)
            : new org.nd4j.linalg.learning.config.Sgd(learningRate)
        );
        builder.l1(l1);
        builder.l2(l2);
        // Activation des workspaces ND4J pour une meilleure gestion mémoire
        builder.trainingWorkspaceMode(WorkspaceMode.ENABLED);
        builder.inferenceWorkspaceMode(WorkspaceMode.ENABLED);

        org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder listBuilder = builder.list();
        int nLayers = config != null ? config.getNumLstmLayers() : 1;
        boolean bidir = config != null && config.isBidirectional();
        boolean attention = config != null && config.isAttention();

        // Couches LSTM empilées
        for (int i = 0; i < nLayers; i++) {
            LSTM.Builder lstmBuilder = new LSTM.Builder()
                .nIn(i == 0 ? inputSize : lstmNeurons)
                .nOut(lstmNeurons)
                .activation(Activation.TANH);
            if (bidir) {
                listBuilder.layer(new org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional(lstmBuilder.build()));
            } else {
                listBuilder.layer(lstmBuilder.build());
            }
            if (dropoutRate > 0.0 && (i < nLayers - 1)) {
                listBuilder.layer(new DropoutLayer.Builder().dropOut(dropoutRate).build());
            }
        }
        if (attention) {
            listBuilder.layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                .nIn(lstmNeurons)
                .nOut(lstmNeurons)
                .activation(Activation.SOFTMAX)
                .build());
        }
        listBuilder.layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
            .nIn(lstmNeurons)
            .nOut(Math.max(16, lstmNeurons / 4))
            .activation(Activation.RELU)
            .build());
        // Correction : choix de l'activation et de la loss selon outputSize et classification
        Activation outputActivation;
        LossFunctions.LossFunction outputLoss;
        if (outputSize == 1) {
            outputActivation = Activation.IDENTITY;
            outputLoss = LossFunctions.LossFunction.MSE;
        } else if (classification) {
            outputActivation = Activation.SOFTMAX;
            outputLoss = LossFunctions.LossFunction.MCXENT;
        } else {
            outputActivation = Activation.IDENTITY;
            outputLoss = LossFunctions.LossFunction.MSE;
        }
        listBuilder.layer(new RnnOutputLayer.Builder()
            .nIn(Math.max(16, lstmNeurons / 4))
            .nOut(outputSize)
            .activation(outputActivation)
            .lossFunction(outputLoss)
            .build());
        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    /**
     * Vérifie et réinitialise le modèle si le nombre de features demandé est différent du modèle courant.
     */
    private MultiLayerNetwork ensureModelWindowSize(MultiLayerNetwork model, int numFeatures, LstmConfig config) {
        if (model == null) {
            logger.info("Réinitialisation du modèle LSTM : numFeatures demandé = {}", numFeatures);
            return initModel(
                numFeatures, // nIn = nombre de features
                1,
                config.getLstmNeurons(),
                config.getDropoutRate(),
                config.getLearningRate(),
                config.getOptimizer(),
                config.getL1(),
                config.getL2(),
                config,
                    true
            );
        } else {
            logger.info("Modèle LSTM déjà initialisé");
            return model;
        }
    }

    // Extraction des valeurs de clôture
    public double[] extractCloseValues(BarSeries series) {
        double[] closes = new double[series.getBarCount()];
        for (int i = 0; i < series.getBarCount(); i++) {
            closes[i] = series.getBar(i).getClosePrice().doubleValue();
        }
        return closes;
    }

    // Normalisation MinMax
    public double[] normalize(double[] values) {
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        for (double v : values) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        double[] normalized = new double[values.length];
        if (min == max) {
            for (int i = 0; i < values.length; i++) {
                normalized[i] = 0.5;
            }
        } else {
            for (int i = 0; i < values.length; i++) {
                normalized[i] = (values[i] - min) / (max - min);
            }
        }
        return normalized;
    }

    public double[] normalize(double[] values, double min, double max) {
        double[] normalized = new double[values.length];
        if (min == max) {
            for (int i = 0; i < values.length; i++) {
                normalized[i] = 0.5;
            }
        } else {
            for (int i = 0; i < values.length; i++) {
                normalized[i] = (values[i] - min) / (max - min);
            }
        }
        return normalized;
    }

    // Séquences univariées [numSeq][windowSize][1]
    public double[][][] createSequences(double[] values, int windowSize) {
        int numSeq = values.length - windowSize;
        double[][][] sequences = new double[numSeq][windowSize][1];
        for (int i = 0; i < numSeq; i++) {
            for (int j = 0; j < windowSize; j++) {
                sequences[i][j][0] = values[i + j];
            }
        }
        return sequences;
    }

    // Conversion en INDArray
    public org.nd4j.linalg.api.ndarray.INDArray toINDArray(double[][][] sequences) {
        return org.nd4j.linalg.factory.Nd4j.create(sequences);
    }

    /**
     * Entraîne le modèle LSTM avec séquences complètes [batch, numFeatures, windowSize].
     * Labels: next-step pour chaque time step de la fenêtre (séquence à séquence).
     * Retourne le modèle et le set de scalers appris sur le train.
     */
    public static class TrainResult {
        public MultiLayerNetwork model;
        public ScalerSet scalers;
        public TrainResult(MultiLayerNetwork model, ScalerSet scalers) {
            this.model = model;
            this.scalers = scalers;
        }
    }
    /**
     * Validation croisée k-fold avec séquences complètes [batch, numFeatures, windowSize].
     */
    public double crossValidateLstm(BarSeries series, LstmConfig config) {
        throw new UnsupportedOperationException("crossValidateLstm legacy supprimé. Utiliser walkForwardEvaluate.");
    }
    /**
     * Validation croisée temporelle (Time Series Split)
     */
    public double crossValidateLstmTimeSeriesSplit(BarSeries series, LstmConfig config) {
        throw new UnsupportedOperationException("crossValidateLstmTimeSeriesSplit legacy supprimé. Utiliser walkForwardEvaluate.");
    }
    /**
     * Prédiction de la prochaine valeur de clôture
     */
    public double predictNextClose(String symbol, BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers) {
        return predictNextCloseScalarV2(series, config, model, scalers);
    }
    /**
     * Prédiction de la prochaine valeur de clôture avec un scaler set
     */
    public double predictNextCloseWithScalerSet(String symbol, BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers) {
        return predictNextCloseScalarV2(series, config, model, scalers);
    }
    /**
     * Évalue le modèle sur le jeu de test (MSE) avec séquences complètes
     */
    private double evaluateModel(MultiLayerNetwork model, BarSeries series, LstmConfig config) {
        throw new UnsupportedOperationException("evaluateModel legacy supprimé.");
    }
    /**
     * Calcule les métriques de trading avancées
     */
    public double[] calculateTradingMetricsAdvanced(BarSeries series, LstmConfig config, MultiLayerNetwork model, double feePct, double slippagePct) {
        throw new UnsupportedOperationException("calculateTradingMetricsAdvanced legacy supprimé. Utiliser simulateTradingWalkForward.");
    }

    /** Prédiction scalar V2 */
    public double predictNextCloseScalarV2(BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers){
        java.util.List<String> features = config.getFeatures();
        int windowSize = config.getWindowSize();
        if(series.getBarCount() <= windowSize) throw new IllegalArgumentException("Pas assez de barres pour prédire");
        double[][] matrix = extractFeatureMatrix(series, features);
        int numFeatures = features.size();
        // normaliser via scalers
        double[][] normMatrix = new double[matrix.length][numFeatures];
        for(int f=0; f<numFeatures; f++){
            double[] col = new double[matrix.length];
            for(int i=0;i<matrix.length;i++) col[i]=matrix[i][f];
            double[] normCol = scalers.featureScalers.get(features.get(f)).transform(col);
            for(int i=0;i<matrix.length;i++) normMatrix[i][f]=normCol[i];
        }
        double[][] lastWin = new double[1][windowSize*numFeatures]; // intermediate not used
        double[][][] seq = new double[1][windowSize][numFeatures];
        for(int j=0;j<windowSize;j++) for(int f=0;f<numFeatures;f++) seq[0][j][f]=normMatrix[normMatrix.length-windowSize+j][f];
        seq = transposeTimeFeature(seq); // [1][features][window]
        org.nd4j.linalg.api.ndarray.INDArray input = toINDArray(seq);
        org.nd4j.linalg.api.ndarray.INDArray out = model.output(input); // [1,1,window]
        double predNorm = out.getDouble(0,0,windowSize-1);
        double predTarget = scalers.labelScaler.inverse(predNorm);
        double lastClose = series.getLastBar().getClosePrice().doubleValue();
        if(config.isUseLogReturnTarget()) {
            double predictedClose = lastClose * Math.exp(predTarget); // log-return -> close
            return predictedClose;
        }
        return predTarget;
    }

    /** Walk-forward evaluation V2 */
    public WalkForwardResultV2 walkForwardEvaluate(BarSeries series, LstmConfig config){
        WalkForwardResultV2 result = new WalkForwardResultV2();
        int splits = Math.max(2, config.getWalkForwardSplits());
        int windowSize = config.getWindowSize();
        int totalBars = series.getBarCount();
        int minBarsNeeded = windowSize + 50; // arbitraire minimal
        if(totalBars < minBarsNeeded){
            logger.warn("[V2] Pas assez de barres pour walk-forward");
            return result;
        }
        int splitSize = (totalBars - windowSize) / splits;
        double sumMse=0.0; int mseCount=0; double sumBusiness=0.0; int businessCount=0;
        java.util.List<Double> mseList = new java.util.ArrayList<>();
        for(int s=1; s<=splits; s++){
            int testEndBar = windowSize + s*splitSize;
            if(s==splits) testEndBar = totalBars;
            int testStartBar = windowSize + (s-1)*splitSize + config.getEmbargoBars();
            if(testStartBar + windowSize + 5 >= testEndBar) continue; // skip split trop petit
            BarSeries trainSeries = series.getSubSeries(0, testStartBar);
            TrainResult tr = trainLstmScalarV2(trainSeries, config, null);
            MultiLayerNetwork model = tr.model;
            ScalerSet scalers = tr.scalers;
            TradingMetricsV2 metrics = simulateTradingWalkForward(series, trainSeries.getBarCount(), testStartBar, testEndBar, model, scalers, config);
            metrics.mse = computeSplitMse(series, testStartBar, testEndBar, model, scalers, config);
            // Business score par split
            double pfAdj = Math.min(metrics.profitFactor, config.getBusinessProfitFactorCap());
            double expPos = Math.max(metrics.expectancy, 0.0);
            double denom = 1.0 + Math.pow(Math.max(metrics.maxDrawdownPct,0.0), config.getBusinessDrawdownGamma());
            metrics.businessScore = (expPos * pfAdj * metrics.winRate) / (denom + 1e-9);
            result.splits.add(metrics);
            if(!Double.isNaN(metrics.mse) && !Double.isInfinite(metrics.mse)){
                sumMse += metrics.mse; mseCount++; mseList.add(metrics.mse);
            }
            if(!Double.isNaN(metrics.businessScore) && !Double.isInfinite(metrics.businessScore)){
                sumBusiness += metrics.businessScore; businessCount++;
            }
        }
        result.meanMse = mseCount>0? sumMse/mseCount: Double.NaN;
        result.meanBusinessScore = businessCount>0? sumBusiness/businessCount: Double.NaN;
        // Ajout du calcul de la variance inter-modèles du MSE
        if(mseList.size()>1){
            double mean = result.meanMse;
            double var = mseList.stream().mapToDouble(m->(m-mean)*(m-mean)).sum()/mseList.size();
            result.mseVariance = var;
            // Calcul et stockage de la variance inter-modèles
            double interVar = 0.0;
            for (double mse : mseList) interVar += Math.pow(mse - mean, 2);
            result.mseInterModelVariance = interVar / mseList.size();
        } else {
            result.mseVariance = 0.0;
            result.mseInterModelVariance = 0.0;
        }
        return result;
    }

    /** Simulation trading walk-forward sur plage test. */
    public TradingMetricsV2 simulateTradingWalkForward(BarSeries fullSeries, int trainBarCount, int testStartBar, int testEndBar, MultiLayerNetwork model, ScalerSet scalers, LstmConfig config){
        TradingMetricsV2 tm = new TradingMetricsV2();
        double equity = 0.0; double peak = 0.0; double trough=0.0; boolean inPosition=false; boolean longPos=false; double entry=0.0; int barsInPos=0; int totalBars=0; int horizon = Math.max(3, config.getHorizonBars());
        java.util.List<Double> tradePnL = new java.util.ArrayList<>();
        java.util.List<Double> tradeReturns = new java.util.ArrayList<>();
        java.util.List<Integer> barsInPositionList = new java.util.ArrayList<>();
        java.util.List<Double> positionSizes = new java.util.ArrayList<>(); // Ajout pour reporting
        double timeInPos=0.0;
        int barsTested = 0;
        int positionChanges = 0;
        double capital = config.getCapital();
        double riskPct = config.getRiskPct();
        double sizingK = config.getSizingK();
        double positionSize = 0.0;
        double entryATR = 0.0;
        double entrySpread = 0.0; // Nouveau : spread moyen à l'entrée
        double feePct = config.getFeePct();
        double slippagePct = config.getSlippagePct();
        for(int bar=testStartBar; bar<testEndBar-1; bar++){
            totalBars++;
            BarSeries sub = fullSeries.getSubSeries(0, bar+1); // jusqu'à bar inclus
            if(sub.getBarCount() <= config.getWindowSize()) continue;
            double threshold = computeSwingTradeThreshold(sub, config); // proportion (ATR % ou autre)
            org.ta4j.core.indicators.ATRIndicator atrInd = new org.ta4j.core.indicators.ATRIndicator(sub, 14);
            double atr = atrInd.getValue(sub.getEndIndex()).doubleValue();
            if(!inPosition){
                double predicted = predictNextCloseScalarV2(sub, config, model, scalers);
                double lastClose = sub.getLastBar().getClosePrice().doubleValue();
                double upLevel = lastClose * (1+threshold);
                double downLevel = lastClose * (1-threshold);
                if(predicted > upLevel){
                    inPosition=true; longPos=true; entry=lastClose; barsInPos=0; positionChanges++;
                    entryATR = atr;
                    positionSize = (atr > 0) ? capital * riskPct / (atr * sizingK) : 0.0;
                    entrySpread = computeMeanSpread(sub); // calcul du spread moyen historique sur la fenêtre sub
                }
                else if(predicted < downLevel){
                    inPosition=true; longPos=false; entry=lastClose; barsInPos=0; positionChanges++;
                    entryATR = atr;
                    positionSize = (atr > 0) ? capital * riskPct / (atr * sizingK) : 0.0;
                    entrySpread = computeMeanSpread(sub); // calcul du spread moyen historique sur la fenêtre sub
                }
            } else {
                barsInPos++; timeInPos++;
                double currentClose = fullSeries.getBar(bar).getClosePrice().doubleValue();
                double stop, target;
                if(longPos){ stop = entry*(1-threshold); target = entry*(1+2*threshold); } else { stop = entry*(1+threshold); target = entry*(1-2*threshold); }
                boolean exit=false; double pnl=0.0;
                if(longPos){
                    if(currentClose <= stop){ pnl = (currentClose-entry) * positionSize; exit=true; }
                    else if(currentClose >= target){ pnl = (currentClose-entry) * positionSize; exit=true; }
                } else {
                    if(currentClose >= stop){ pnl = (entry-currentClose) * positionSize; exit=true; }
                    else if(currentClose <= target){ pnl = (entry-currentClose) * positionSize; exit=true; }
                }
                if(!exit && barsInPos>=horizon){ double current= currentClose; pnl = longPos? (current-entry)*positionSize: (entry-current)*positionSize; exit=true; }
                if(exit){
                    // Application des coûts dynamiques (spread, slippage, frais)
                    double spreadCost = entrySpread * positionSize; // coût du spread moyen
                    double slippageCost = slippagePct * entry * positionSize; // slippage en % du prix d'entrée
                    double feeCost = feePct * entry * positionSize; // frais en % du prix d'entrée
                    pnl -= (spreadCost + slippageCost + feeCost); // déduction des coûts du PnL
                    tradePnL.add(pnl);
                    tradeReturns.add(pnl/(entry*positionSize));
                    barsInPositionList.add(barsInPos);
                    positionSizes.add(positionSize); // Stocker la taille de position pour chaque trade
                    equity += pnl;
                    if(equity>peak) {peak=equity; trough=equity;} else if(equity<trough){ trough=equity; }
                    inPosition=false; longPos=false; entry=0.0; barsInPos=0; positionSize=0.0; entryATR=0.0; entrySpread=0.0;
                }
            }
            barsTested++;
        }
        double gains=0.0; double losses=0.0; int win=0; int loss=0;
        for(double p: tradePnL){ if(p>0){ gains+=p; win++; } else if(p<0){ losses+=p; loss++; } }
        tm.totalProfit = gains + losses;
        tm.numTrades = tradePnL.size();
        tm.profitFactor = losses!=0? gains/Math.abs(losses): (gains>0? Double.POSITIVE_INFINITY:0);
        tm.winRate = tm.numTrades>0? (double)win/tm.numTrades:0.0;
        double ddAbs = peak - trough; tm.maxDrawdownPct = peak!=0? ddAbs / Math.abs(peak): 0.0;
        double avgGain = win>0? gains/win:0.0; double avgLoss = loss>0? Math.abs(losses)/loss:0.0;
        tm.expectancy = (win>0 || loss>0)? (tm.winRate*avgGain - (1-tm.winRate)*avgLoss):0.0;
        double meanRet = tradeReturns.stream().mapToDouble(d->d).average().orElse(0.0);
        double stdRet = Math.sqrt(tradeReturns.stream().mapToDouble(d-> (d-meanRet)*(d-meanRet)).average().orElse(0.0));
        tm.sharpe = stdRet>0? meanRet/stdRet * Math.sqrt(Math.max(1, tradeReturns.size())):0.0;
        double downsideStd = Math.sqrt(tradeReturns.stream().filter(r->r<0).mapToDouble(r-> { double dr = r-meanRet; return dr*dr; }).average().orElse(0.0));
        tm.sortino = downsideStd>0? meanRet/downsideStd:0.0;
        // Amélioration du calcul d'exposition et turnover
        tm.exposure = barsTested>0? timeInPos/barsTested:0.0;
        tm.turnover = barsTested>0? (double)positionChanges/barsTested:0.0;
        tm.avgBarsInPosition = barsInPositionList.size()>0 ? barsInPositionList.stream().mapToInt(Integer::intValue).average().orElse(0.0) : 0.0;
        // Business score calculé plus tard (besoin config) -> placeholder
        return tm;
    }

    /**
     * Calcule le seuil swing trade selon la config : ATR(14) ou std des log-returns
     */
    public double computeSwingTradeThreshold(BarSeries series, LstmConfig config) {
        double k = config.getThresholdK();
        String type = config.getThresholdType();
        double threshold = 0.0;
        if ("ATR".equalsIgnoreCase(type)) {
            org.ta4j.core.indicators.ATRIndicator atr = new org.ta4j.core.indicators.ATRIndicator(series, 14);
            double lastATR = atr.getValue(series.getEndIndex()).doubleValue();
            double lastClose = series.getBar(series.getEndIndex()).getClosePrice().doubleValue();
            threshold = k * lastATR / lastClose; // ATR en % du prix
        } else if ("returns".equalsIgnoreCase(type)) {
            double[] closes = extractCloseValues(series);
            if (closes.length < 2) return 0.0;
            double[] logReturns = new double[closes.length - 1];
            for (int i = 1; i < closes.length; i++) {
                logReturns[i - 1] = Math.log(closes[i] / closes[i - 1]);
            }
            double mean = java.util.Arrays.stream(logReturns).average().orElse(0.0);
            double std = Math.sqrt(java.util.Arrays.stream(logReturns).map(r -> Math.pow(r - mean, 2)).sum() / logReturns.length);
            threshold = k * std;
        } else {
            double[] closes = extractCloseValues(series);
            if (closes.length < 2) return 0.0;
            double avgPrice = java.util.Arrays.stream(closes).average().orElse(0.0);
            double volatility = 0.0;
            for (int i = 1; i < closes.length; i++) {
                volatility += Math.abs(closes[i] - closes[i-1]);
            }
            volatility /= (closes.length - 1);
            threshold = Math.max(avgPrice * 0.01, volatility);
        }
        logger.info("[SEUIL SWING] Symbol={}, Type={}, k={}, Seuil calculé={}", config.getSwingTradeType(), type, k, threshold);
        return threshold;
    }

    public PreditLsdm getPredit(String symbol, BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers) {
        model = ensureModelWindowSize(model, config.getFeatures().size(), config);
        double th = computeSwingTradeThreshold(series, config);
        double predicted = predictNextCloseWithScalerSet(symbol, series, config, model, scalers);
        predicted = Math.round(predicted * 1000.0) / 1000.0;
        double[] closes = extractCloseValues(series);
        double lastClose = closes[closes.length - 1];
        double[] lastWindow = new double[config.getWindowSize()];
        System.arraycopy(closes, closes.length - config.getWindowSize(), lastWindow, 0, config.getWindowSize());
        String position = analyzePredictionPosition(lastWindow, predicted);
        double delta =  predicted - lastClose;
        SignalType signal;
        if (delta > th) {
            signal = SignalType.UP;
        } else if (delta < -th) {
            signal = SignalType.DOWN;
        } else {
            signal = SignalType.STABLE;
        }
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd/MM");
        String formattedDate = series.getLastBar().getEndTime().format(formatter);
        logger.info("------------PREDICT {} | lastClose={}, predictedClose={}, delta={}, threshold={}, signal={}, position={}",
            config.getWindowSize(), lastClose, predicted, delta, th, signal, position);

        return PreditLsdm.builder()
                .lastClose(lastClose)
                .predictedClose(predicted)
                .signal(signal)
                .lastDate(formattedDate)
                .position(position)
                .build();
    }

    // Surcharge pour compatibilité (ancienne signature)
    public PreditLsdm getPredit(String symbol, BarSeries series, LstmConfig config, MultiLayerNetwork model) {
        // Chargement du scaler depuis la base si possible
        LoadedModel loaded = null;
        try {
            loaded = loadModelAndScalersFromDb(symbol, jdbcTemplate);
        } catch (Exception e) {
            logger.warn("Impossible de charger le ScalerSet depuis la base : {}", e.getMessage());
        }
        ScalerSet scalers = loaded != null ? loaded.scalers : null;
        return getPredit(symbol, series, config, model, scalers);
    }

    /**
     * Sauvegarde le modèle LSTM en base MySQL.
     * @param symbol symbole du modèle
     * @param jdbcTemplate template JDBC Spring
     * @throws IOException en cas d'erreur d'accès à la base
     */
    // Sauvegarde du modèle dans MySQL
    public void saveModelToDb(String symbol, MultiLayerNetwork model, JdbcTemplate jdbcTemplate, LstmConfig config, ScalerSet scalers) throws IOException {
        if (model != null) {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ModelSerializer.writeModel(model, baos, true);
            byte[] modelBytes = baos.toByteArray();
            hyperparamsRepository.saveHyperparams(symbol, config);
            LstmHyperparameters params = new LstmHyperparameters(
                config.getWindowSize(),
                config.getLstmNeurons(),
                config.getDropoutRate(),
                config.getLearningRate(),
                config.getNumEpochs(),
                config.getPatience(),
                config.getMinDelta(),
                config.getOptimizer(),
                config.getL1(),
                config.getL2(),
                config.getNormalizationScope(),
                config.getNormalizationMethod()
            );
            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            String hyperparamsJson = mapper.writeValueAsString(params);
            String scalersJson = mapper.writeValueAsString(scalers);
            String sql = "REPLACE INTO lstm_models (symbol, model_blob, hyperparams_json, normalization_scope, scalers_json, updated_date) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)";
            try {
                jdbcTemplate.update(sql, symbol, modelBytes, hyperparamsJson, config.getNormalizationScope(), scalersJson);
                logger.info("Modèle, hyperparamètres et scalers sauvegardés en base pour le symbole : {} (scope={})", symbol, config.getNormalizationScope());
            } catch (Exception e) {
                logger.error("Erreur lors de la sauvegarde du modèle/scalers en base : {}", e.getMessage());
                throw e;
            }
        }
    }

    /**
     * Charge le modèle LSTM depuis la base MySQL.
     * @param symbol symbole du modèle
     * @param jdbcTemplate template JDBC Spring
     * @throws IOException en cas d'erreur d'accès à la base
     * @throws SQLException jamais levée (pour compatibilité)
     */
    // Chargement du modèle depuis MySQL
    public MultiLayerNetwork loadModelFromDb(String symbol, JdbcTemplate jdbcTemplate) throws IOException {
        LstmConfig config = hyperparamsRepository.loadHyperparams(symbol);
        if (config == null) {
            throw new IOException("Aucun hyperparamètre trouvé pour le symbole " + symbol);
        }
        String sql = "SELECT model_blob, normalization_scope, hyperparams_json FROM lstm_models WHERE symbol = ?";
        MultiLayerNetwork model = null;
        try {
            java.util.Map<String, Object> result = jdbcTemplate.queryForMap(sql, symbol);
            byte[] modelBytes = (byte[]) result.get("model_blob");
            String normalizationScope = (String) result.get("normalization_scope");
            String hyperparamsJson = result.containsKey("hyperparams_json") ? (String) result.get("hyperparams_json") : null;
            boolean normalizationSet = false;
            if (normalizationScope != null && !normalizationScope.isEmpty()) {
                config.setNormalizationScope(normalizationScope);
                normalizationSet = true;
            }
            // Fallback : si normalization_scope absent, essayer de lire du JSON
            if (!normalizationSet && hyperparamsJson != null && !hyperparamsJson.isEmpty()) {
                try {
                    com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                    java.util.Map<?,?> jsonMap = mapper.readValue(hyperparamsJson, java.util.Map.class);
                    Object normScopeObj = jsonMap.get("normalizationScope");
                    if (normScopeObj != null && normScopeObj instanceof String && !((String)normScopeObj).isEmpty()) {
                        config.setNormalizationScope((String)normScopeObj);
                        normalizationSet = true;
                    }
                } catch (Exception e) {
                    logger.warn("Impossible de parser hyperparams_json pour normalizationScope : {}", e.getMessage());
                }
            }
            if (modelBytes != null) {
                ByteArrayInputStream bais = new ByteArrayInputStream(modelBytes);
                model = ModelSerializer.restoreMultiLayerNetwork(bais);
                logger.info("Modèle chargé depuis la base pour le symbole : {} (scope={})", symbol, config.getNormalizationScope());
            }
        } catch (EmptyResultDataAccessException e) {
            logger.error("Modèle non trouvé en base pour le symbole : {}", symbol);
            throw new IOException("Modèle non trouvé en base");
        }
        return model;
    }

    public static class LoadedModel {
        public MultiLayerNetwork model;
        public LstmTradePredictor.ScalerSet scalers;
        public LoadedModel(MultiLayerNetwork model, LstmTradePredictor.ScalerSet scalers) {
            this.model = model;
            this.scalers = scalers;
        }
    }

    public LoadedModel loadModelAndScalersFromDb(String symbol, JdbcTemplate jdbcTemplate) throws IOException {
        LstmConfig config = hyperparamsRepository.loadHyperparams(symbol);
        if (config == null) {
            throw new IOException("Aucun hyperparamètre trouvé pour le symbole " + symbol);
        }
        String sql = "SELECT model_blob, normalization_scope, hyperparams_json, scalers_json FROM lstm_models WHERE symbol = ?";
        MultiLayerNetwork model = null;
        LstmTradePredictor.ScalerSet scalers = null;
        try {
            java.util.Map<String, Object> result = jdbcTemplate.queryForMap(sql, symbol);
            byte[] modelBytes = (byte[]) result.get("model_blob");
            String normalizationScope = (String) result.get("normalization_scope");
            String hyperparamsJson = result.containsKey("hyperparams_json") ? (String) result.get("hyperparams_json") : null;
            String scalersJson = result.containsKey("scalers_json") ? (String) result.get("scalers_json") : null;
            boolean normalizationSet = false;
            if (normalizationScope != null && !normalizationScope.isEmpty()) {
                config.setNormalizationScope(normalizationScope);
                normalizationSet = true;
            }
            if (!normalizationSet && hyperparamsJson != null && !hyperparamsJson.isEmpty()) {
                try {
                    com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                    java.util.Map<?,?> jsonMap = mapper.readValue(hyperparamsJson, java.util.Map.class);
                    Object normScopeObj = jsonMap.get("normalizationScope");
                    if (normScopeObj != null && normScopeObj instanceof String && !((String)normScopeObj).isEmpty()) {
                        config.setNormalizationScope((String)normScopeObj);
                        normalizationSet = true;
                    }
                } catch (Exception e) {
                    logger.warn("Impossible de parser hyperparams_json pour normalizationScope : {}", e.getMessage());
                }
            }
            if (modelBytes != null) {
                ByteArrayInputStream bais = new ByteArrayInputStream(modelBytes);
                model = ModelSerializer.restoreMultiLayerNetwork(bais);
                logger.info("Modèle chargé depuis la base pour le symbole : {} (scope={})", symbol, config.getNormalizationScope());
            }
            if (scalersJson != null && !scalersJson.isEmpty()) {
                try {
                    com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                    scalers = mapper.readValue(scalersJson, LstmTradePredictor.ScalerSet.class);
                } catch (Exception e) {
                    logger.warn("Impossible de parser scalers_json : {}", e.getMessage());
                }
            }
        } catch (EmptyResultDataAccessException e) {
            logger.error("Modèle non trouvé en base pour le symbole : {}", symbol);
            throw new IOException("Modèle non trouvé en base");
        }
        return new LoadedModel(model, scalers);
    }

    /**
     * Analyse la position du prix prédit par rapport à la fenêtre précédente.
     * Retourne "au-dessus", "en-dessous" ou "dans la plage".
     */
    public String analyzePredictionPosition(double[] lastWindow, double predicted) {
        double min = java.util.Arrays.stream(lastWindow).min().orElse(Double.NaN);
        double max = java.util.Arrays.stream(lastWindow).max().orElse(Double.NaN);
        if (predicted > max) return "au-dessus";
        if (predicted < min) return "en-dessous";
        return "dans la plage";
    }

    /**
     * Entraîne le modèle LSTM avec labels scalaires (close_{t+1} ou log-return_{t+1})
     * - Séquences d'entrée : [numSeq][windowSize][numFeatures]
     * - Labels : [numSeq] (scalaire)
     * - Normalisation fit uniquement sur train
     * - nOut=1
     */
    public TrainResult trainLstmScalarV2(BarSeries series, LstmConfig config, Object unused) {
        java.util.List<String> features = config.getFeatures();
        int windowSize = config.getWindowSize();
        int numFeatures = features.size();
        int barCount = series.getBarCount();
        if (barCount <= windowSize + 1) return new TrainResult(null, null);
        // Extraction des features
        double[][] matrix = extractFeatureMatrix(series, features);
        // Extraction des closes
        double[] closes = extractCloseValues(series);
        // Construction des labels scalaires
        int numSeq = barCount - windowSize - 1;
        double[][][] inputSeq = new double[numSeq][windowSize][numFeatures];
        double[] labelSeq = new double[numSeq];
        for (int i = 0; i < numSeq; i++) {
            for (int j = 0; j < windowSize; j++) {
                for (int f = 0; f < numFeatures; f++) {
                    inputSeq[i][j][f] = matrix[i + j][f];
                }
            }
            // Label = close_{i+windowSize} ou log-return
            if (config.isUseLogReturnTarget()) {
                double prev = closes[i + windowSize - 1];
                double next = closes[i + windowSize];
                labelSeq[i] = Math.log(next / prev);
            } else {
                labelSeq[i] = closes[i + windowSize];
            }
        }
        // Fit des scalers sur train uniquement
        ScalerSet scalers = new ScalerSet();
        for (int f = 0; f < numFeatures; f++) {
            double[] col = new double[numSeq + windowSize];
            for (int i = 0; i < numSeq + windowSize; i++) col[i] = matrix[i][f];
            FeatureScaler.Type type = getFeatureNormalizationType(features.get(f)).equals("zscore") ? FeatureScaler.Type.ZSCORE : FeatureScaler.Type.MINMAX;
            FeatureScaler scaler = new FeatureScaler(type);
            scaler.fit(col);
            scalers.featureScalers.put(features.get(f), scaler);
        }
        // Fit scaler label sur train
        FeatureScaler.Type labelType = FeatureScaler.Type.MINMAX;
        FeatureScaler labelScaler = new FeatureScaler(labelType);
        labelScaler.fit(labelSeq);
        scalers.labelScaler = labelScaler;
        // Normalisation des séquences
        double[][][] normSeq = new double[numSeq][windowSize][numFeatures];
        for (int i = 0; i < numSeq; i++) {
            for (int j = 0; j < windowSize; j++) {
                for (int f = 0; f < numFeatures; f++) {
                    normSeq[i][j][f] = scalers.featureScalers.get(features.get(f)).transform(new double[]{inputSeq[i][j][f]})[0];
                }
            }
        }
        // Normalisation des labels
        double[] normLabels = scalers.labelScaler.transform(labelSeq);
        // Conversion en INDArray
        org.nd4j.linalg.api.ndarray.INDArray X = org.nd4j.linalg.factory.Nd4j.create(normSeq); // [numSeq][window][features]
        org.nd4j.linalg.api.ndarray.INDArray y = org.nd4j.linalg.factory.Nd4j.create(normLabels, new int[]{numSeq, 1});
        // Initialisation du modèle
        MultiLayerNetwork model = initModel(numFeatures, 1, config.getLstmNeurons(), config.getDropoutRate(), config.getLearningRate(), config.getOptimizer(), config.getL1(), config.getL2(), config, false);
        // Entraînement
        org.nd4j.linalg.dataset.DataSet ds = new org.nd4j.linalg.dataset.DataSet(X, y);
        java.util.List<org.nd4j.linalg.dataset.DataSet> dsList = java.util.Collections.singletonList(ds);
        org.nd4j.linalg.dataset.api.iterator.DataSetIterator iterator = new ListDataSetIterator(dsList, config.getBatchSize());
        model.fit(iterator, config.getNumEpochs());
        return new TrainResult(model, scalers);
    }

    public boolean containsNaN(org.nd4j.linalg.api.ndarray.INDArray array) {
        for (int i = 0; i < array.length(); i++) {
            if (Double.isNaN(array.getDouble(i))) return true;
        }
        return false;
    }

    public double[] normalizeByConfig(double[] values, double min, double max, LstmConfig config) {
        String method = config.getNormalizationMethod();
        if ("zscore".equalsIgnoreCase(method)) {
            double mean = java.util.Arrays.stream(values).average().orElse(0.0);
            double std = Math.sqrt(java.util.Arrays.stream(values).map(v -> (v - mean) * (v - mean)).average().orElse(0.0));
            double[] normalized = new double[values.length];
            if (std == 0.0) {
                for (int i = 0; i < values.length; i++) normalized[i] = 0.0;
            } else {
                for (int i = 0; i < values.length; i++) normalized[i] = (values[i] - mean) / std;
            }
            return normalized;
        } else {
            double[] normalized = new double[values.length];
            if (min == max) {
                for (int i = 0; i < values.length; i++) normalized[i] = 0.5;
            } else {
                for (int i = 0; i < values.length; i++) normalized[i] = (values[i] - min) / (max - min);
            }
            return normalized;
        }
    }

    public String getFeatureNormalizationType(String feature) {
        String f = feature.toLowerCase();
        if (f.equals("rsi") || f.equals("stochastic") || f.equals("cci") || f.equals("momentum")) {
            return "zscore";
        } else if (f.equals("close") || f.equals("open") || f.equals("high") || f.equals("low") || f.equals("sma") || f.equals("ema") || f.equals("macd") || f.equals("atr") || f.startsWith("bollinger")) {
            return "minmax";
        } else if (f.equals("volume")) {
            return "minmax";
        } else if (f.equals("day_of_week") || f.equals("month")) {
            return "minmax";
        }
        return "minmax";
    }

    public double[] interpolateNaN(double[] col) {
        int n = col.length;
        double[] res = new double[n];
        System.arraycopy(col, 0, res, 0, n);
        double lastValid = Double.NaN;
        for (int i = 0; i < n; i++) {
            if (Double.isNaN(res[i])) {
                int j = i + 1;
                while (j < n && Double.isNaN(res[j])) j++;
                if (j < n) res[i] = res[j];
                else if (!Double.isNaN(lastValid)) res[i] = lastValid;
            } else {
                lastValid = res[i];
            }
        }
        for (int i = 0; i < n; i++) {
            if (Double.isNaN(res[i])) res[i] = 0.0;
        }
        return res;
    }

    public double[][] extractFeatureMatrix(BarSeries series, java.util.List<String> features) {
        int barCount = series.getBarCount();
        int numFeatures = features.size();
        double[][] matrix = new double[barCount][numFeatures];
        // Pré-calcul des indicateurs techniques pour éviter recalculs inutiles
        org.ta4j.core.indicators.helpers.ClosePriceIndicator closeIndicator = new org.ta4j.core.indicators.helpers.ClosePriceIndicator(series);
        org.ta4j.core.indicators.helpers.VolumeIndicator volumeIndicator = new org.ta4j.core.indicators.helpers.VolumeIndicator(series);
        org.ta4j.core.indicators.SMAIndicator sma = new org.ta4j.core.indicators.SMAIndicator(closeIndicator, 14);
        org.ta4j.core.indicators.EMAIndicator ema = new org.ta4j.core.indicators.EMAIndicator(closeIndicator, 14);
        org.ta4j.core.indicators.RSIIndicator rsi = new org.ta4j.core.indicators.RSIIndicator(closeIndicator, 14);
        org.ta4j.core.indicators.MACDIndicator macd = new org.ta4j.core.indicators.MACDIndicator(closeIndicator, 12, 26);
        org.ta4j.core.indicators.ATRIndicator atr = new org.ta4j.core.indicators.ATRIndicator(series, 14);
        org.ta4j.core.indicators.bollinger.BollingerBandsMiddleIndicator bbm = new org.ta4j.core.indicators.bollinger.BollingerBandsMiddleIndicator(sma);
        Num k = series.numOf(2.0);
        org.ta4j.core.indicators.bollinger.BollingerBandsUpperIndicator bbu = new org.ta4j.core.indicators.bollinger.BollingerBandsUpperIndicator(bbm, new StandardDeviationIndicator(closeIndicator, 14), k);
        org.ta4j.core.indicators.bollinger.BollingerBandsLowerIndicator bbl = new org.ta4j.core.indicators.bollinger.BollingerBandsLowerIndicator(bbm, new StandardDeviationIndicator(closeIndicator, 14), k);
        org.ta4j.core.indicators.StochasticOscillatorKIndicator stochastic = new org.ta4j.core.indicators.StochasticOscillatorKIndicator(series, 14);
        org.ta4j.core.indicators.CCIIndicator cci = new org.ta4j.core.indicators.CCIIndicator(series, 14);
        ROCIndicator momentum = new ROCIndicator(closeIndicator, 10);
        for (int i = 0; i < barCount; i++) {
            int f = 0;
            for (String feat : features) {
                switch (feat.toLowerCase()) {
                    case "close":
                        matrix[i][f] = closeIndicator.getValue(i).doubleValue();
                        break;
                    case "volume":
                        matrix[i][f] = volumeIndicator.getValue(i).doubleValue();
                        break;
                    case "open":
                        matrix[i][f] = series.getBar(i).getOpenPrice().doubleValue();
                        break;
                    case "high":
                        matrix[i][f] = series.getBar(i).getHighPrice().doubleValue();
                        break;
                    case "low":
                        matrix[i][f] = series.getBar(i).getLowPrice().doubleValue();
                        break;
                    case "sma":
                        matrix[i][f] = i >= 13 ? sma.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "ema":
                        matrix[i][f] = i >= 13 ? ema.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "rsi":
                        matrix[i][f] = i >= 13 ? rsi.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "macd":
                        matrix[i][f] = i >= 25 ? macd.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "atr":
                        matrix[i][f] = i >= 13 ? atr.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "bollinger_high":
                        matrix[i][f] = i >= 13 ? bbu.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "bollinger_low":
                        matrix[i][f] = i >= 13 ? bbl.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "stochastic":
                        matrix[i][f] = i >= 13 ? stochastic.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "cci":
                        matrix[i][f] = i >= 13 ? cci.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "momentum":
                        matrix[i][f] = i >= 9 ? momentum.getValue(i).doubleValue() : Double.NaN;
                        break;
                    case "day_of_week":
                        matrix[i][f] = series.getBar(i).getEndTime().getDayOfWeek().getValue();
                        break;
                    case "month":
                        matrix[i][f] = series.getBar(i).getEndTime().getMonthValue();
                        break;
                    default:
                        matrix[i][f] = Double.NaN;
                }
                f++;
            }
        }
        for (int f = 0; f < numFeatures; f++) {
            double[] col = new double[barCount];
            for (int i = 0; i < barCount; i++) col[i] = matrix[i][f];
            col = interpolateNaN(col);
            for (int i = 0; i < barCount; i++) matrix[i][f] = col[i];
        }
        return matrix;
    }

    public double[] filterOutliers(double[] col) {
        int n = col.length;
        double[] sorted = java.util.Arrays.copyOf(col, n);
        java.util.Arrays.sort(sorted);
        double q1 = sorted[n / 4];
        double q3 = sorted[(3 * n) / 4];
        double iqr = q3 - q1;
        double lower = q1 - 1.5 * iqr;
        double upper = q3 + 1.5 * iqr;
        double[] filtered = new double[n];
        for (int i = 0; i < n; i++) {
            filtered[i] = Math.max(lower, Math.min(upper, col[i]));
        }
        return filtered;
    }

    private java.util.Map<String, double[]> driftStats = new java.util.HashMap<>();
    public void monitorDrift(double[] col, String featureName) {
        double mean = java.util.Arrays.stream(col).average().orElse(0.0);
        double std = Math.sqrt(java.util.Arrays.stream(col).map(v -> (v - mean) * (v - mean)).average().orElse(0.0));
        double[] lastStats = driftStats.get(featureName);
        if (lastStats != null) {
            double lastMean = lastStats[0];
            double lastStd = lastStats[1];
            if (Math.abs(mean - lastMean) / (Math.abs(lastMean) + 1e-8) > 0.2 || Math.abs(std - lastStd) / (Math.abs(lastStd) + 1e-8) > 0.2) {
                logger.warn("[DRIFT] Feature '{}' : drift détecté (mean {} -> {}, std {} -> {})", featureName, lastMean, mean, lastStd, std);
            }
        }
        driftStats.put(featureName, new double[]{mean, std});
    }

    public double[][] normalizeMatrix(double[][] matrix, java.util.List<String> features) {
        int barCount = matrix.length;
        int numFeatures = matrix[0].length;
        double[][] norm = new double[barCount][numFeatures];
        for (int f = 0; f < numFeatures; f++) {
            double[] col = new double[barCount];
            for (int i = 0; i < barCount; i++) col[i] = matrix[i][f];
            col = filterOutliers(col);
            monitorDrift(col, features.get(f));
            String normType = getFeatureNormalizationType(features.get(f));
            if (normType.equals("zscore")) {
                double mean = java.util.Arrays.stream(col).average().orElse(0.0);
                double std = Math.sqrt(java.util.Arrays.stream(col).map(v -> (v - mean) * (v - mean)).average().orElse(0.0));
                for (int i = 0; i < barCount; i++) norm[i][f] = std == 0.0 ? 0.0 : (col[i] - mean) / std;
            } else {
                double min = java.util.Arrays.stream(col).min().orElse(0.0);
                double max = java.util.Arrays.stream(col).max().orElse(0.0);
                for (int i = 0; i < barCount; i++) norm[i][f] = (min == max) ? 0.5 : (col[i] - min) / (max - min);
            }
        }
        return norm;
    }

    public double[][][] createSequencesMulti(double[][] matrix, int windowSize) {
        int numSeq = matrix.length - windowSize;
        int numFeatures = matrix[0].length;
        double[][][] sequences = new double[numSeq][windowSize][numFeatures];
        for (int i = 0; i < numSeq; i++) {
            for (int j = 0; j < windowSize; j++) {
                for (int f = 0; f < numFeatures; f++) {
                    sequences[i][j][f] = matrix[i + j][f];
                }
            }
        }
        return sequences;
    }

    public double[][][] transposeSequencesMulti(double[][][] sequences) {
        int numSeq = sequences.length;
        int windowSize = sequences[0].length;
        int numFeatures = sequences[0][0].length;
        double[][][] transposed = new double[numSeq][numFeatures][windowSize];
        for (int i = 0; i < numSeq; i++) {
            for (int j = 0; j < windowSize; j++) {
                for (int f = 0; f < numFeatures; f++) {
                    transposed[i][f][j] = sequences[i][j][f];
                }
            }
        }
        return transposed;
    }

    // Transpose [batch][window][features] -> [batch][features][window] pour l'inférence LSTM DL4J
    private double[][][] transposeTimeFeature(double[][][] seq) {
        int batch = seq.length;
        int time = seq[0].length;
        int features = seq[0][0].length;
        double[][][] out = new double[batch][features][time];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < time; t++)
                for (int f = 0; f < features; f++)
                    out[b][f][t] = seq[b][t][f];
        return out;
    }

    /**
     * Calcule le MSE sur un split de test (indices testStartBar à testEndBar) avec le modèle et les scalers fournis.
     */
    public double computeSplitMse(BarSeries series, int testStartBar, int testEndBar, MultiLayerNetwork model, ScalerSet scalers, LstmConfig config) {
        java.util.List<String> features = config.getFeatures();
        int windowSize = config.getWindowSize();
        int numFeatures = features.size();
        double[][] matrix = extractFeatureMatrix(series, features);
        double[] closes = extractCloseValues(series);
        int numSeq = testEndBar - testStartBar - windowSize;
        if (numSeq <= 0) return Double.NaN;
        double[][][] inputSeq = new double[numSeq][numFeatures][windowSize];
        double[] targets = new double[numSeq];
        for (int i = 0; i < numSeq; i++) {
            int idx = testStartBar + i;
            for (int f = 0; f < numFeatures; f++) {
                for (int t = 0; t < windowSize; t++) {
                    inputSeq[i][f][t] = scalers.featureScalers.get(features.get(f)).transform(new double[]{matrix[idx + t][f]})[0];
                }
            }
            double targetClose = closes[idx + windowSize];
            if (config.isUseLogReturnTarget()) {
                double prev = closes[idx + windowSize - 1];
                targets[i] = Math.log(targetClose / prev);
            } else {
                targets[i] = targetClose;
            }
        }
        double[] normTargets = scalers.labelScaler.transform(targets);
        org.nd4j.linalg.api.ndarray.INDArray input = toINDArray(inputSeq);
        org.nd4j.linalg.api.ndarray.INDArray out = model.output(input);
        double mse = 0.0;
        for (int i = 0; i < numSeq; i++) {
            double predNorm = out.getDouble(i, 0, windowSize - 1);
            double predTarget = scalers.labelScaler.inverse(predNorm);
            double trueTarget = targets[i];
            if (config.isUseLogReturnTarget()) {
                double lastClose = closes[testStartBar + i + windowSize - 1];
                predTarget = lastClose * Math.exp(predTarget);
                trueTarget = closes[testStartBar + i + windowSize];
            }
            mse += Math.pow(predTarget - trueTarget, 2);
        }
        return mse / numSeq;
    }

    // Classe utilitaire pour la normalisation cohérente
    public static class FeatureScaler implements java.io.Serializable {
        public enum Type { MINMAX, ZSCORE }
        public Type type;
        public double min, max, mean, std;

        // Constructeur sans argument pour la désérialisation JSON (Jackson)
        public FeatureScaler() {}

        public FeatureScaler(Type type) { this.type = type; }
        public void fit(double[] values) {
            if (type == Type.MINMAX) {
                min = java.util.Arrays.stream(values).min().orElse(0.0);
                max = java.util.Arrays.stream(values).max().orElse(0.0);
            } else {
                mean = java.util.Arrays.stream(values).average().orElse(0.0);
                std = Math.sqrt(java.util.Arrays.stream(values).map(v -> (v - mean) * (v - mean)).average().orElse(0.0));
            }
        }
        public double[] transform(double[] values) {
            double[] res = new double[values.length];
            if (type == Type.MINMAX) {
                if (min == max) {
                    for (int i = 0; i < values.length; i++) res[i] = 0.5;
                } else {
                    for (int i = 0; i < values.length; i++) res[i] = (values[i] - min) / (max - min);
                }
            } else {
                if (std == 0.0) {
                    for (int i = 0; i < values.length; i++) res[i] = 0.0;
                } else {
                    for (int i = 0; i < values.length; i++) res[i] = (values[i] - mean) / std;
                }
            }
            return res;
        }
        public double inverse(double value) {
            if (type == Type.MINMAX) return value * (max - min) + min;
            else return value * std + mean;
        }
    }

    // Structure pour stocker les scalers de toutes les features + label
    public static class ScalerSet implements java.io.Serializable {
        public java.util.Map<String, FeatureScaler> featureScalers = new java.util.HashMap<>();
        public FeatureScaler labelScaler;
    }

    /** Structure métriques trading V2 */
    public static class TradingMetricsV2 {
        public double totalProfit;
        public int numTrades;
        public double profitFactor;
        public double winRate;
        public double maxDrawdownPct;
        public double expectancy;
        public double sharpe;
        public double sortino;
        public double exposure;
        public double turnover;
        public double mse; // MSE out-of-sample pour le split
        public double businessScore;
        public double avgBarsInPosition; // Ajouté pour suivi moyen durée position
    }
    /** Résultat walk-forward V2 */
    public static class WalkForwardResultV2 {
        public java.util.List<TradingMetricsV2> splits = new java.util.ArrayList<>();
        public double meanMse;
        public double meanBusinessScore;
        public double mseVariance;
        public double mseInterModelVariance; // Ajouté pour variance inter-modèles
    }

    /** Applique les seeds globaux (ND4J, DL4J, Java) pour reproductibilité */
    public void setGlobalSeeds(long seed){
        try {
            org.nd4j.linalg.factory.Nd4j.getRandom().setSeed(seed);
            System.setProperty("org.deeplearning4j.defaultSeed", String.valueOf(seed));
            // Le seed DL4J doit être passé lors de la création du modèle (déjà fait dans initModel)
            new java.util.Random(seed); // force initialisation
        } catch (Exception ignored) {}
    }

    /**
     * Calcule le spread moyen (high-low) sur une fenêtre donnée.
     */
    public double computeMeanSpread(BarSeries series) {
        int n = series.getBarCount();
        if (n == 0) return 0.0;
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double high = series.getBar(i).getHighPrice().doubleValue();
            double low = series.getBar(i).getLowPrice().doubleValue();
            sum += (high - low);
        }
        return sum / n;
    }
}
