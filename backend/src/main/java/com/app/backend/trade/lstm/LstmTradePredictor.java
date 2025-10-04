/**
 * Service central pour:
 *  - Construire / entraîner un modèle LSTM (DeepLearning4J) adapté à des séries financières.
 *  - Extraire et normaliser des features techniques.
 *  - Prédire la prochaine valeur de clôture (soit directement le prix, soit un log-return).
 *  - Évaluer la robustesse via walk-forward testing.
 *  - Simuler une stratégie dérivée des prédictions (pseudo swing trading).
 *  - Détecter un "drift" statistique et relancer un entraînement si nécessaire.
 *
 * IMPORTANT: Cette classe est volontairement "riche" (monolithique) :
 *  Pour un futur refactoring avancé, on pourrait isoler:
 *    - Extraction des features
 *    - Normalisation & scalers
 *    - Construction du modèle
 *    - Simulation trading
 *    - Drift detection
 *  MAIS: Toute modification interne doit être faite avec prudence pour éviter les régressions.
 *
 * Concepts clés:
 *  - Fenêtre (windowSize): nombre de pas temporels utilisés comme entrée de la LSTM.
 *  - Features: vecteurs dérivés (close, rsi, sma, etc.). Chaque feature est normalisée séparément.
 *  - Label: soit prix futur (close t+1), soit log-return (log(close_t / close_{t-1})).
 *  - Normalisation: mélange MinMax + ZScore selon la nature de la feature. (Voir getFeatureNormalizationType)
 *  - Forme tenseur (DL4J):
 *      Séquences initiales: [batch][time][features]
 *      Après permutation (pour ce code): [batch][features][time]
 *      => cohérent avec l'usage explicitement contrôlé + LastTimeStep.
 *
 * ATTENTION:
 *  - NE PAS changer les permutations sans revoir l'init du modèle + InputType.
 *  - NE PAS renommer les méthodes publiques car potentiellement utilisées ailleurs (REST/service).
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
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
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
import java.time.ZonedDateTime;
import java.util.*;
import java.util.Arrays;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.GradientNormalization;

@Service
public class LstmTradePredictor {

    static {
        try {
            // Étape 1: forcer float32 global
            Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
        } catch (Throwable t) {
            System.err.println("[LSTM][INIT][WARN] Échec setDefaultDataTypes FLOAT: " + t.getMessage());
        }
    }

    /* ---------------------------------------------------------
     * LOG & DEPENDANCES
     * --------------------------------------------------------- */
    private static final Logger logger = LoggerFactory.getLogger(LstmTradePredictor.class);
    private final LstmHyperparamsRepository hyperparamsRepository;
    private final JdbcTemplate jdbcTemplate;

    /**
     * Injection des dépendances (repository hyperparamètres + accès DB).
     */
    public LstmTradePredictor(LstmHyperparamsRepository hyperparamsRepository, JdbcTemplate jdbcTemplate) {
        this.hyperparamsRepository = hyperparamsRepository;
        this.jdbcTemplate = jdbcTemplate;
    }

    /* =========================================================
     *               CONSTRUCTION / INITIALISATION MODELE
     * =========================================================
     * Le modèle est un empilement de couches LSTM (potentiellement bidirectionnelles),
     * suivi d'un "LastTimeStep" (prend le dernier vecteur caché), éventuellement
     * une pseudo couche d'attention (dense softmax interne), puis une couche Dense,
     * enfin une OutputLayer (MSE ou CrossEntropy selon régression / classification).
     *
     * NE PAS MODIFIER l'ordre des permutations ou InputType sinon
     * risque d'erreurs shape silencieuses.
     */

    /**
     * Initialise un réseau LSTM selon les hyperparamètres fournis.
     *
     * @param inputSize       Nombre de features en entrée
     * @param outputSize      Taille de la sortie (1 pour régression)
     * @param lstmNeurons     Nombre de neurones par couche LSTM
     * @param dropoutRate     Taux de dropout (entre couches récurrentes si >0)
     * @param learningRate    Taux d'apprentissage
     * @param optimizer       Nom de l'optimiseur ("adam", "rmsprop", sinon SGD)
     * @param l1              Régularisation L1
     * @param l2              Régularisation L2
     * @param config          Configuration globale (contient nombre de couches, bidirectionnel, etc.)
     * @param classification  Si true => softmax + cross entropy
     * @return Réseau initialisé
     */
    public MultiLayerNetwork initModel(
        int inputSize,
        int outputSize,
        int lstmNeurons,
        double dropoutRate,
        double learningRate,
        String optimizer,
        double l1,
        double l2,
        LstmConfig config,
        boolean classification
    ) {
        // Builder de base
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.dataType(DataType.FLOAT); // Étape 1: dtype FLOAT

        // Sélection dynamique de l'updater (optimiseur) selon chaîne.
        builder.updater(
            "adam".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.Adam(learningRate)
                : "rmsprop".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.RmsProp(learningRate)
                : new org.nd4j.linalg.learning.config.Sgd(learningRate)
        );

        // Régularisations - réduites pour permettre plus de variabilité
        builder.l1(l1 * 0.5).l2(l2 * 0.5); // Réduction de 50% pour moins de contraintes

        // Activation des workspaces mémoire (optimisation Dl4J)
        builder.trainingWorkspaceMode(WorkspaceMode.ENABLED)
               .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
               // Étape 8: Gradient clipping pour prévenir dérives / NaN
               .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
               .gradientNormalizationThreshold(1.0);

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();

        // Paramètres dynamiques
        int nLayers = config != null ? config.getNumLstmLayers() : 1;
        boolean bidir = config != null && config.isBidirectional();
        boolean attention = config != null && config.isAttention();

        // Empilement des couches LSTM
        for (int i = 0; i < nLayers; i++) {
            int inSize = (i == 0) ? inputSize : (bidir ? lstmNeurons * 2 : lstmNeurons);

            // Construction d'une couche LSTM avec activation TANH (stabilité des gradients)
            LSTM.Builder lstmBuilder = new LSTM.Builder()
                .nIn(inSize)
                .nOut(lstmNeurons)
                // Étape 7: retour à TANH pour limiter explosions de gradients sur séquences financières
                .activation(Activation.TANH);

            // Si bidirectionnel, on encapsule
            org.deeplearning4j.nn.conf.layers.Layer recurrent =
                bidir
                    ? new org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional(lstmBuilder.build())
                    : lstmBuilder.build();

            if (i == nLayers - 1) {
                // LastTimeStep = extrait uniquement le dernier pas temporel de la séquence
                // => simplifie la suite (plutôt qu'un pooling global)
                listBuilder.layer(new LastTimeStep(recurrent));
            } else {
                listBuilder.layer(recurrent);
                // Dropout plus agressif pour éviter le sur-apprentissage conservateur
                if (dropoutRate > 0.0) {
                    // Étape 9: Dropout récurrent plafonné à 0.25 (suppression *1.5 pour éviter sous-apprentissage)
                    listBuilder.layer(new DropoutLayer.Builder()
                        .dropOut(Math.min(Math.max(dropoutRate, 0.0), 0.25))
                        .build());
                }
            }
        }

        // "Pseudo-attention" simplifiée : dense softmax sur la représentation finale
        if (attention) {
            listBuilder.layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                .nIn(lstmNeurons * (bidir ? 2 : 1))
                .nOut(lstmNeurons * (bidir ? 2 : 1))
                .activation(Activation.SOFTMAX)
                .build());
        }

        int finalRecurrentSize = lstmNeurons * (bidir ? 2 : 1);
        int denseOut = Math.max(32, lstmNeurons / 2); // Augmentation de la taille de la couche dense

        // Couche Dense de projection avec plus de non-linéarité
        listBuilder.layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
            .nIn(finalRecurrentSize)
            .nOut(denseOut)
            .activation(Activation.LEAKYRELU) // CHANGEMENT: LeakyReLU pour éviter les neurones morts
            .build());

        // Couche dense supplémentaire pour plus de capacité d'expression
        listBuilder.layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
            .nIn(denseOut)
            .nOut(Math.max(16, denseOut / 2))
            .activation(Activation.SWISH) // CHANGEMENT: SWISH pour des gradients plus fluides
            .build());

        // Dropout avant la couche finale pour éviter le sur-apprentissage
        if (dropoutRate > 0.0) {
            // Étape 9: Dropout dense final = min(0.2, dropoutRate)
            listBuilder.layer(new DropoutLayer.Builder()
                .dropOut(Math.min(0.2, Math.max(dropoutRate, 0.0)))
                .build());
        }

        // Sélection dynamique de la fonction de perte / activation finale
        Activation outAct;
        ILossFunction lossFn;
        if (outputSize == 1 && !classification) {
            outAct = Activation.IDENTITY;
            // Étape 10: Huber via implémentation custom (delta=1.0)
            lossFn = new LossHuberCustom(1.0);
        } else if (classification) {
            outAct = Activation.SOFTMAX;
            lossFn = new LossMCXENT();
        } else {
            outAct = Activation.IDENTITY;
            lossFn = new LossHuberCustom(1.0);
        }

        listBuilder.layer(new OutputLayer.Builder(lossFn)
            .nIn(Math.max(16, denseOut / 2))
            .nOut(outputSize)
            .activation(outAct)
            .build());

        // IMPORTANT: On force explicitement le type d'entrée
        // Ici on travaille conceptuellement avec du recurrent(inputSize)
        listBuilder.setInputType(InputType.recurrent(inputSize));

        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        // Étape 9: log des valeurs de dropout réellement appliquées (récurrent plafonné 0.25, dense final = min(0.2, dropoutRate))
        double appliedRecurrentDropout = (dropoutRate > 0.0) ? Math.min(Math.max(dropoutRate, 0.0), 0.25) : 0.0;
        double appliedFinalDenseDropout = (dropoutRate > 0.0) ? Math.min(0.2, Math.max(dropoutRate, 0.0)) : 0.0;
        logger.info("[LSTM][Etape9] Dropout recurrent applique={} | Dropout dense final={}", appliedRecurrentDropout, appliedFinalDenseDropout);
        return model;
    }

    /**
     * Vérifie que le modèle existe et correspond (approximativement) au nombre de features.
     * Cette méthode n'altère pas la logique; elle crée un nouveau modèle si null.
     *
     * @param model       Modèle éventuellement existant (peut être null)
     * @param numFeatures Nombre de features actuelles
     * @param config      Configuration LSTM
     * @return Modèle prêt à l'usage
     */
    private MultiLayerNetwork ensureModelWindowSize(MultiLayerNetwork model, int numFeatures, LstmConfig config) {
        if (model == null) {
            logger.info("Initialisation modèle LSTM nIn={}", numFeatures);
            return initModel(
                numFeatures,
                1,
                config.getLstmNeurons(),
                config.getDropoutRate(),
                config.getLearningRate(),
                config.getOptimizer(),
                config.getL1(),
                config.getL2(),
                config,
                false
            );
        }
        return model;
    }

    /* =========================================================
     *                  EXTRACTION DES FEATURES
     * =========================================================
     * On construit une matrice [nbBars][nbFeatures].
     * Chaque feature est calculée dynamiquement seulement si demandée
     * pour limiter le coût.
     */

    /**
     * Extrait une matrice de features depuis une série TA4J optimisée pour le swing trade professionnel.
     * Chaque feature est indexée dans l'ordre fourni par la liste features.
     *
     * @param series   Série temporelle de barres (TA4J)
     * @param features Liste ordonnée des features à extraire
     * @return Matrice double[barCount][featuresCount]
     */
    public double[][] extractFeatureMatrix(BarSeries series, List<String> features) {
        int n = series.getBarCount();
        int fCount = features.size();
        double[][] M = new double[n][fCount];
        if (n == 0) return M;

        // Pré-calcul spécifique realized_vol si demandé
        boolean needRealizedVol = features.contains("realized_vol");
        double[] realizedVol = null;
        if (needRealizedVol) {
            final int WIN = 14; // fenêtre log-returns
            double[] closesRaw = new double[n];
            for (int i = 0; i < n; i++) closesRaw[i] = series.getBar(i).getClosePrice().doubleValue();
            double[] logRet = new double[n];
            logRet[0] = 0.0;
            for (int i = 1; i < n; i++) {
                double prev = closesRaw[i - 1];
                double cur = closesRaw[i];
                if (prev > 0 && cur > 0) {
                    double lr = Math.log(cur / prev);
                    if (Double.isFinite(lr)) logRet[i] = lr; else logRet[i] = 0.0;
                } else logRet[i] = 0.0;
            }
            realizedVol = new double[n];
            double sum = 0.0, sum2 = 0.0;
            // Sliding window
            for (int i = 0; i < n; i++) {
                double r = logRet[i];
                sum += r; sum2 += r * r;
                if (i >= WIN) { // retirer élément sorti de fenêtre
                    double old = logRet[i - WIN];
                    sum -= old;
                    sum2 -= old * old;
                }
                if (i >= WIN) { // fenêtre pleine WIN éléments (indices i-WIN+1 .. i)
                    double mean = sum / WIN;
                    double var = (sum2 / WIN) - mean * mean;
                    if (var < 0) var = 0;
                    realizedVol[i] = Math.sqrt(var) * Math.sqrt(WIN);
                } else {
                    realizedVol[i] = 0.0; // insuffisant historique
                }
                if (!Double.isFinite(realizedVol[i])) realizedVol[i] = 0.0;
            }
        }

        // Indicateurs de base
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        HighPriceIndicator high = new HighPriceIndicator(series);
        LowPriceIndicator low = new LowPriceIndicator(series);
        VolumeIndicator vol = new VolumeIndicator(series);
        org.ta4j.core.indicators.helpers.OpenPriceIndicator open = new org.ta4j.core.indicators.helpers.OpenPriceIndicator(series);

        // RSI variants pour swing trade
        RSIIndicator rsi = features.contains("rsi") ? new RSIIndicator(close, 14) : null;
        RSIIndicator rsi14 = features.contains("rsi_14") ? new RSIIndicator(close, 14) : null;
        RSIIndicator rsi21 = features.contains("rsi_21") ? new RSIIndicator(close, 21) : null;

        // SMA variants pour swing trade
        SMAIndicator sma = features.contains("sma") ? new SMAIndicator(close, 14) : null;
        SMAIndicator sma20 = (features.contains("sma_20") || features.contains("bollinger_high") || features.contains("bollinger_low") || features.contains("bollinger_width"))
            ? new SMAIndicator(close, 20) : null;
        SMAIndicator sma50 = features.contains("sma_50") ? new SMAIndicator(close, 50) : null;

        // EMA variants pour swing trade
        EMAIndicator ema = features.contains("ema") ? new EMAIndicator(close, 14) : null;
        EMAIndicator ema12 = features.contains("ema_12") ? new EMAIndicator(close, 12) : null;
        EMAIndicator ema26 = features.contains("ema_26") ? new EMAIndicator(close, 26) : null;
        EMAIndicator ema50 = features.contains("ema_50") ? new EMAIndicator(close, 50) : null;

        // MACD complet pour swing trade
        MACDIndicator macd = features.contains("macd") ? new MACDIndicator(close, 12, 26) : null;
        org.ta4j.core.indicators.EMAIndicator macdSignal = features.contains("macd_signal") && macd != null
            ? new org.ta4j.core.indicators.EMAIndicator(macd, 9) : null;

        // ATR variants pour swing trade
        ATRIndicator atr = features.contains("atr") ? new ATRIndicator(series, 14) : null;
        ATRIndicator atr14 = features.contains("atr_14") ? new ATRIndicator(series, 14) : null;
        ATRIndicator atr21 = features.contains("atr_21") ? new ATRIndicator(series, 21) : null;

        // Bollinger Bands complets
        StandardDeviationIndicator sd20 = (features.contains("bollinger_high") || features.contains("bollinger_low") || features.contains("bollinger_width"))
            ? new StandardDeviationIndicator(close, 20) : null;

        // Stochastic complet
        StochasticOscillatorKIndicator stoch = features.contains("stochastic") ? new StochasticOscillatorKIndicator(series, 14) : null;
        org.ta4j.core.indicators.StochasticOscillatorDIndicator stochD = features.contains("stochastic_d")
            ? new org.ta4j.core.indicators.StochasticOscillatorDIndicator(stoch != null ? stoch : new StochasticOscillatorKIndicator(series, 14)) : null;

        // Williams %R pour swing trade
        org.ta4j.core.indicators.WilliamsRIndicator williamsR = features.contains("williams_r")
            ? new org.ta4j.core.indicators.WilliamsRIndicator(series, 14) : null;

        // CCI pour swing trade
        CCIIndicator cci = features.contains("cci") ? new CCIIndicator(series, 20) : null;

        // ROC (Rate of Change) pour swing trade
        org.ta4j.core.indicators.ROCIndicator roc = features.contains("roc")
            ? new org.ta4j.core.indicators.ROCIndicator(close, 12) : null;

        // ADX et DI pour trend strength
        org.ta4j.core.indicators.adx.ADXIndicator adx = features.contains("adx")
            ? new org.ta4j.core.indicators.adx.ADXIndicator(series, 14) : null;
        org.ta4j.core.indicators.adx.PlusDIIndicator diPlus = features.contains("di_plus")
            ? new org.ta4j.core.indicators.adx.PlusDIIndicator(series, 14) : null;
        org.ta4j.core.indicators.adx.MinusDIIndicator diMinus = features.contains("di_minus")
            ? new org.ta4j.core.indicators.adx.MinusDIIndicator(series, 14) : null;

        // OBV pour volume analysis
        org.ta4j.core.indicators.volume.OnBalanceVolumeIndicator obv = features.contains("obv")
            ? new org.ta4j.core.indicators.volume.OnBalanceVolumeIndicator(series) : null;

        boolean needMomentum = features.contains("momentum");
        boolean needVolumeRatio = features.contains("volume_ratio");
        boolean needPricePosition = features.contains("price_position");
        boolean needVolatilityRegime = features.contains("volatility_regime");
        // Étape 5: préparation suivi skew volume
        boolean needVolume = features.contains("volume");
        double[] rawVolumeForSkew = needVolume ? new double[n] : null;

        // Parcours de chaque barre
        for (int i = 0; i < n; i++) {
            double closeVal = close.getValue(i).doubleValue();
            double highVal = high.getValue(i).doubleValue();
            double lowVal = low.getValue(i).doubleValue();
            double openVal = open.getValue(i).doubleValue();
            double volVal = vol.getValue(i).doubleValue();
            if (needVolume) rawVolumeForSkew[i] = volVal; // stock brut pour skew
            ZonedDateTime t = series.getBar(i).getEndTime();

            for (int f = 0; f < fCount; f++) {
                String feat = features.get(f);
                double val = 0.0;

                // Switch exhaustif pour toutes les features de swing trade
                switch (feat) {
                    case "close" -> val = closeVal;
                    case "high" -> val = highVal;
                    case "low" -> val = lowVal;
                    case "open" -> val = openVal;
                    case "volume" -> {
                        // Étape 5: appliquer log1p(volume) avant normalisation
                        double safeVol = volVal;
                        if (safeVol < 0) safeVol = 0; // sécurité
                        double logV = Math.log1p(safeVol);
                        if (!Double.isFinite(logV)) logV = 0.0;
                        val = logV;
                    }

                    // RSI variants
                    case "rsi" -> val = rsi != null ? rsi.getValue(i).doubleValue() : 0.0;
                    case "rsi_14" -> val = rsi14 != null ? rsi14.getValue(i).doubleValue() : 0.0;
                    case "rsi_21" -> val = rsi21 != null ? rsi21.getValue(i).doubleValue() : 0.0;

                    // SMA variants
                    case "sma" -> val = sma != null ? sma.getValue(i).doubleValue() : 0.0;
                    case "sma_20" -> val = sma20 != null ? sma20.getValue(i).doubleValue() : 0.0;
                    case "sma_50" -> val = sma50 != null ? sma50.getValue(i).doubleValue() : 0.0;

                    // EMA variants
                    case "ema" -> val = ema != null ? ema.getValue(i).doubleValue() : 0.0;
                    case "ema_12" -> val = ema12 != null ? ema12.getValue(i).doubleValue() : 0.0;
                    case "ema_26" -> val = ema26 != null ? ema26.getValue(i).doubleValue() : 0.0;
                    case "ema_50" -> val = ema50 != null ? ema50.getValue(i).doubleValue() : 0.0;

                    // MACD complet
                    case "macd" -> val = macd != null ? macd.getValue(i).doubleValue() : 0.0;
                    case "macd_signal" -> val = macdSignal != null ? macdSignal.getValue(i).doubleValue() : 0.0;
                    case "macd_histogram" -> {
                        if (macd != null && macdSignal != null) {
                            val = macd.getValue(i).doubleValue() - macdSignal.getValue(i).doubleValue();
                        } else val = 0.0;
                    }

                    // ATR variants
                    case "atr" -> val = atr != null ? atr.getValue(i).doubleValue() : 0.0;
                    case "atr_14" -> val = atr14 != null ? atr14.getValue(i).doubleValue() : 0.0;
                    case "atr_21" -> val = atr21 != null ? atr21.getValue(i).doubleValue() : 0.0;

                    // Bollinger Bands
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
                    case "bollinger_width" -> {
                        if (sma20 != null && sd20 != null) {
                            double sdv = sd20.getValue(i).doubleValue();
                            val = 4 * sdv; // Width = Upper - Lower = 4 * stddev
                        } else val = 0.0;
                    }

                    // Stochastic
                    case "stochastic" -> val = stoch != null ? stoch.getValue(i).doubleValue() : 0.0;
                    case "stochastic_d" -> val = stochD != null ? stochD.getValue(i).doubleValue() : 0.0;
                    case "williams_r" -> val = williamsR != null ? williamsR.getValue(i).doubleValue() : 0.0;

                    // Autres oscillateurs
                    case "cci" -> val = cci != null ? cci.getValue(i).doubleValue() : 0.0;
                    case "roc" -> val = roc != null ? roc.getValue(i).doubleValue() : 0.0;

                    // Momentum
                    case "momentum" -> {
                        if (needMomentum && i >= 10) {
                            // Momentum multi-période pour capturer différentes vitesses de mouvement
                            double mom_3 = (close.getValue(i).doubleValue() - close.getValue(i - 3).doubleValue()) / close.getValue(i - 3).doubleValue();
                            double mom_10 = (close.getValue(i).doubleValue() - close.getValue(i - 10).doubleValue()) / close.getValue(i - 10).doubleValue();
                            val = (mom_3 + mom_10) / 2; // Moyenne pondérée des momentums
                        } else {
                            val = 0.0;
                        }
                    }

                    // Nouvelle feature : Force de la tendance (Rate of Change normalisé)
                    case "trend_strength" -> {
                        if (i >= 20) {
                            double roc5 = (closeVal - close.getValue(i - 5).doubleValue()) / close.getValue(i - 5).doubleValue();
                            double roc10 = (closeVal - close.getValue(i - 10).doubleValue()) / close.getValue(i - 10).doubleValue();
                            double roc20 = (closeVal - close.getValue(i - 20).doubleValue()) / close.getValue(i - 20).doubleValue();
                            val = (roc5 * 0.5 + roc10 * 0.3 + roc20 * 0.2); // Pondération décroissante
                        } else val = 0.0;
                    }

                    // NOUVELLE FEATURE : Gap de prix (différence overnight/intraday)
                    case "price_gap" -> {
                        if (i > 0) {
                            double prevClose = close.getValue(i - 1).doubleValue();
                            double gap = (openVal - prevClose) / prevClose;
                            val = gap; // Capture les gaps de marché
                        } else val = 0.0;
                    }

                    // NOUVELLE FEATURE : Range expansion (expansion/contraction de volatilité)
                    case "range_expansion" -> {
                        if (i >= 10) {
                            double currentRange = (highVal - lowVal) / closeVal;
                            double avgRange = 0;
                            for (int j = 1; j <= 10; j++) {
                                if (i - j >= 0) {
                                    double pastHigh = high.getValue(i - j).doubleValue();
                                    double pastLow = low.getValue(i - j).doubleValue();
                                    double pastClose = close.getValue(i - j).doubleValue();
                                    avgRange += (pastHigh - pastLow) / pastClose;
                                }
                            }
                            avgRange /= 10;
                            val = avgRange > 0 ? (currentRange / avgRange - 1) : 0; // Expansion relative
                        } else val = 0.0;
                    }

                    // NOUVELLE FEATURE : Momentum des hauts/bas (breakout detection)
                    case "breakout_momentum" -> {
                        if (i >= 20) {
                            // Calcul des niveaux de résistance/support récents
                            double highestHigh = closeVal;
                            double lowestLow = closeVal;
                            for (int j = 1; j <= 20; j++) {
                                if (i - j >= 0) {
                                    highestHigh = Math.max(highestHigh, high.getValue(i - j).doubleValue());
                                    lowestLow = Math.min(lowestLow, low.getValue(i - j).doubleValue());
                                }
                            }
                            // Force du breakout
                            double upBreakout = closeVal > highestHigh ? (closeVal - highestHigh) / highestHigh : 0;
                            double downBreakout = closeVal < lowestLow ? (lowestLow - closeVal) / lowestLow : 0;
                            val = upBreakout - downBreakout; // Positif = breakout haussier
                        } else val = 0.0;
                    }

                    // NOUVELLE FEATURE : Accélération du momentum
                    case "momentum_acceleration" -> {
                        if (i >= 15) {
                            // Calcul de l'accélération du momentum sur 3 périodes
                            double mom1 = i >= 5 ? (closeVal - close.getValue(i - 5).doubleValue()) / close.getValue(i - 5).doubleValue() : 0;
                            double mom2 = i >= 10 ? (close.getValue(i - 5).doubleValue() - close.getValue(i - 10).doubleValue()) / close.getValue(i - 10).doubleValue() : 0;
                            double mom3 = i >= 15 ? (close.getValue(i - 10).doubleValue() - close.getValue(i - 15).doubleValue()) / close.getValue(i - 15).doubleValue() : 0;
                            val = (mom1 - mom2) + (mom2 - mom3); // Accélération du momentum
                        } else val = 0.0;
                    }

                    // Nouvelle feature : Volatilité de momentum (capture l'accélération/décélération)
                    case "momentum_volatility" -> {
                        if (i >= 15) {
                            double[] returns = new double[10];
                            for (int j = 0; j < 10; j++) {
                                if (i - j - 1 >= 0) {
                                    returns[j] = (close.getValue(i - j).doubleValue() - close.getValue(i - j - 1).doubleValue())
                                                / close.getValue(i - j - 1).doubleValue();
                                }
                            }
                            double meanRet = Arrays.stream(returns).average().orElse(0.0);
                            double variance = Arrays.stream(returns).map(r -> Math.pow(r - meanRet, 2)).average().orElse(0.0);
                            val = Math.sqrt(variance) * 100; // Amplification pour plus de sensibilité
                        } else val = 0.0;
                    }

                    // Nouvelle feature : Position relative dans la bande de Bollinger (plus discriminante)
                    case "bollinger_position" -> {
                        if (sma20 != null && sd20 != null) {
                            double mid = sma20.getValue(i).doubleValue();
                            double sdv = sd20.getValue(i).doubleValue();
                            double upper = mid + 2 * sdv;
                            double lower = mid - 2 * sdv;
                            val = (upper > lower) ? (closeVal - lower) / (upper - lower) : 0.5;
                            // Amplification des signaux extrêmes
                            if (val > 0.8) val = 0.8 + (val - 0.8) * 2; // Amplifier les signaux de surachat
                            if (val < 0.2) val = 0.2 - (0.2 - val) * 2; // Amplifier les signaux de survente
                        } else val = 0.5;
                    }

                    // NOUVELLE FEATURE : Momentum croisé (momentum vs moyenne mobile)
                    case "cross_momentum" -> {
                        if (sma20 != null && i >= 5) {
                            double currentMom = (closeVal - close.getValue(i - 5).doubleValue()) / close.getValue(i - 5).doubleValue();
                            double smaLevel = sma20.getValue(i).doubleValue();
                            double priceVsSma = (closeVal - smaLevel) / smaLevel;
                            val = currentMom * Math.signum(priceVsSma) * 2; // Amplifier quand momentum et position s'alignent
                        } else val = 0.0;
                    }

                    // Nouvelle feature : Divergence de momentum (RSI vs Prix) - version améliorée
                    case "momentum_divergence" -> {
                        if (rsi14 != null && i >= 20) {
                            double[] prices = new double[10];
                            double[] rsiVals = new double[10];
                            for (int j = 0; j < 10; j++) {
                                if (i - j >= 0) {
                                    prices[j] = close.getValue(i - j).doubleValue();
                                    rsiVals[j] = rsi14.getValue(i - j).doubleValue();
                                }
                            }
                            double meanPrice = Arrays.stream(prices).average().orElse(0.0);
                            double meanRsi = Arrays.stream(rsiVals).average().orElse(0.0);
                            double numerator = 0, denomPrice = 0, denomRsi = 0;
                            for (int j = 0; j < 10; j++) {
                                double priceDiff = prices[j] - meanPrice;
                                double rsiDiff = rsiVals[j] - meanRsi;
                                numerator += priceDiff * rsiDiff;
                                denomPrice += priceDiff * priceDiff;
                                denomRsi += rsiDiff * rsiDiff;
                            }
                            double correlation = (denomPrice * denomRsi > 0) ? numerator / Math.sqrt(denomPrice * denomRsi) : 0.0;
                            val = (1.0 - Math.abs(correlation)) * 2;
                        } else val = 0.0;
                    }
                    case "realized_vol" -> {
                        // Valeur déjà pré-calculée
                        val = needRealizedVol ? realizedVol[i] : 0.0;
                    }
                }

                // Nettoyage valeurs invalides
                if (Double.isNaN(val) || Double.isInfinite(val)) val = 0.0;
                M[i][f] = val;
            }
        }

        // Vérification diversité realized_vol (si calculée)
        if (needRealizedVol) {
            try {
                int col = features.indexOf("realized_vol");
                java.util.Set<Double> distinct = new java.util.HashSet<>();
                for (int i = 0; i < n; i++) distinct.add(M[i][col]);
                int distinctCount = distinct.size();
                if (distinctCount <= 20) {
                    logger.warn("[FEATURE][REALIZED_VOL] Faible diversité ({} valeurs distinctes <=20) - vérifier dataset", distinctCount);
                } else {
                    logger.debug("[FEATURE][REALIZED_VOL] Diversité OK ({} valeurs distinctes)", distinctCount);
                }
            } catch (Exception e) {
                logger.warn("[FEATURE][REALIZED_VOL] Échec comptage diversité: {}", e.getMessage());
            }
        }
        // Étape 5: calcul skew avant/après transformation volume
        if (needVolume) {
            try {
                int vCol = features.indexOf("volume");
                double[] volLog = new double[n];
                for (int i = 0; i < n; i++) volLog[i] = M[i][vCol];
                for (int i = 0; i < n; i++) if (rawVolumeForSkew[i] < 0 || !Double.isFinite(rawVolumeForSkew[i])) rawVolumeForSkew[i] = 0.0;
                double skewRaw = computeSkewness(rawVolumeForSkew);
                double skewLog = computeSkewness(volLog);
                double absRaw = Math.abs(skewRaw);
                double absLog = Math.abs(skewLog);
                double reduction = absRaw > 1e-9 ? (absRaw - absLog) / absRaw : 0.0;
                String status = reduction >= 0.30 ? "ACCEPTÉ" : "ATTENTION";
                logger.info("[FEATURE][VOLUME_LOG1P] skew_raw={} skew_log={} reduction={} ({})", String.format(Locale.US, "%.4f", skewRaw), String.format(Locale.US, "%.4f", skewLog), String.format(Locale.US, "%.1f%%", reduction * 100.0), status);
                if (reduction < 0.30) {
                    logger.warn("[FEATURE][VOLUME_LOG1P] Réduction skew < 30% ({}).", String.format(Locale.US, "%.1f%%", reduction * 100.0));
                }
            } catch (Exception e) {
                logger.warn("[FEATURE][VOLUME_LOG1P] Échec calcul skew: {}", e.getMessage());
            }
        }
        return M;
    }

    // Méthode utilitaire: calcul skewness (3ème moment centré normalisé)
    private double computeSkewness(double[] data) {
        if (data == null || data.length == 0) return 0.0;
        int n = data.length;
        double mean = 0.0;
        for (double v : data) mean += v;
        mean /= n;
        double m2 = 0.0, m3 = 0.0;
        for (double v : data) {
            double d = v - mean;
            double d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
        }
        if (m2 == 0) return 0.0;
        double var = m2 / n;
        double std = Math.sqrt(var);
        if (std < 1e-12) return 0.0;
        double skew = (m3 / n) / (std * std * std);
        if (!Double.isFinite(skew)) return 0.0;
        return skew;
    }

    /* =========================================================
     *               OUTILS MANIPULATION SHAPES
     * ========================================================= */

    /**
     * Transpose un tenseur [batch][time][features] -> [batch][features][time].
     * NOTE: Méthode utilitaire isolée; dans le code principal on utilise souvent
     * une permutation Nd4j directe (permute).
     */
    public double[][][] transposeTimeFeature(double[][][] seq) {
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

    /**
     * Conversion utilitaire vers INDArray (Nd4j).
     */
    public org.nd4j.linalg.api.ndarray.INDArray toINDArray(double[][][] sequences) {
        return Nd4j.create(sequences);
    }

    /* =========================================================
     *              NORMALISATION & SCALERS
     * =========================================================
     * Chaque feature a son scaler dédié (MinMax ou ZScore).
     * Le label a un scaler (MinMax) pour faciliter l’inversion post-prédiction.
     * ATTENTION: Reconstruction a posteriori (rebuildScalers) peut ne PAS
     * reproduire exactement l'état d'entraînement initial.
     */

    /**
     * Représente un scaler de normalisation simple (MinMax ou ZScore).
     * Stocké pour permettre transformation + inversion (inverse).
     */
    public static class FeatureScaler implements Serializable {
        public enum Type { MINMAX, ZSCORE }

        // Paramètres communs (selon le type)
        public double min = Double.POSITIVE_INFINITY;
        public double max = Double.NEGATIVE_INFINITY;
        public double mean = 0.0;
        public double std = 0.0;
        public Type type;

        public FeatureScaler(Type type){ this.type = type; }

        /**
         * Apprend les paramètres de normalisation sur un tableau brut.
         * Pour MINMAX: calcule min / max
         * Pour ZSCORE: calcule mean / std
         */
        public void fit(double[] data){
            if (type == Type.MINMAX) {
                for (double v : data) {
                    if (v < min) min = v;
                    if (v > max) max = v;
                }
                if (min == Double.POSITIVE_INFINITY) { // Sécurité si data vide
                    min = 0; max = 1;
                }
            } else {
                double s = 0;
                for (double v : data) s += v;
                mean = data.length > 0 ? s / data.length : 0;

                double var = 0;
                for (double v : data) var += (v - mean) * (v - mean);
                std = data.length > 0 ? Math.sqrt(var / data.length) : 1.0;
                if (std == 0) std = 1.0;
            }
        }

        /**
         * Transforme un tableau selon le scaler appris.
         * @return Nouveau tableau normalisé.
         */
        public double[] transform(double[] data){
            double[] out = new double[data.length];
            if (type == Type.MINMAX) {
                double range = (max - min) == 0 ? 1e-9 : (max - min);
                for (int i = 0; i < data.length; i++)
                    out[i] = (data[i] - min) / range;
            } else {
                for (int i = 0; i < data.length; i++)
                    out[i] = (data[i] - mean) / (std == 0 ? 1e-9 : std);
            }
            return out;
        }

        /**
         * Inversion d'une valeur (utile pour repasser du domaine normalisé au prix réel ou log-return).
         */
        public double inverse(double v){
            return type == Type.MINMAX ? min + v * (max - min) : mean + v * std;
        }
    }

    /**
     * Conteneur groupant tous les scalers de features + scaler de label.
     */
    public static class ScalerSet implements Serializable {
        public Map<String, FeatureScaler> featureScalers = new HashMap<>();
        public FeatureScaler labelScaler;
        // Étape 22: distribution des labels (log-return) au moment de l'entraînement
        // Nullable pour compat rétro (anciens JSON sans ces champs)
        public Double labelDistMean;   // mean(labelSeq)
        public Double labelDistStd;    // std(labelSeq)
    }

    /* =========================================================
     *                     ENTRAINEMENT
     * ========================================================= */

    /**
     * Petit wrapper pour retourner modèle + scalers.
     */
    public static class TrainResult {
        public MultiLayerNetwork model;
        public ScalerSet scalers;
        // Step13: métadonnées validation
        public Double bestValLoss;
        public Double finalValLoss;
        public boolean bestBetterThanLast;
        // Step21: métrique variance résiduelle prix train & alerte platitude
        public Double residualVariance; // var(predictedCloseTrain - closeTrain)
        public boolean flatModelAlert;
        public TrainResult(MultiLayerNetwork m, ScalerSet s){this.model=m;this.scalers=s;}
        public TrainResult(MultiLayerNetwork m, ScalerSet s, Double bestVal, Double finalVal, boolean improved){
            this.model=m; this.scalers=s; this.bestValLoss=bestVal; this.finalValLoss=finalVal; this.bestBetterThanLast=improved;
        }
    }

    // Step13: compteurs globaux (ratio améliorations bestValLoss vs dernier valLoss)
    private static class Step13Stats {
        private static final java.util.concurrent.atomic.AtomicInteger totalRuns = new java.util.concurrent.atomic.AtomicInteger();
        private static final java.util.concurrent.atomic.AtomicInteger improvedRuns = new java.util.concurrent.atomic.AtomicInteger();
    }

    /**
     * Extraction rapide des valeurs de clôture d'une série.
     */
    public double[] extractCloseValues(BarSeries series) {
        double[] closes = new double[series.getBarCount()];
        for (int i = 0; i < series.getBarCount(); i++)
            closes[i] = series.getBar(i).getClosePrice().doubleValue();
        return closes;
    }

    /**
     * Sélectionne le type de normalisation pour une feature donnée.
     * Rationnel: certaines variables oscillent autour d'une moyenne (RSI, MACD...) => ZSCORE;
     * d'autres grandissent avec le prix => MINMAX.
     */
    public String getFeatureNormalizationType(String feature) {
        return switch (feature) {
            // Features oscillantes autour d'une moyenne -> Z-Score pour capturer les écarts
            case "rsi", "momentum", "stochastic", "cci", "macd", "macd_histogram",
                 "roc", "williams_r", "momentum_volatility", "momentum_divergence",
                 "trend_strength", "bollinger_position", "price_gap", "range_expansion",
                 "breakout_momentum", "momentum_acceleration", "cross_momentum",
                 "realized_vol" -> "zscore"; // <-- ajout realized_vol
            // Prix, volumes, ATR -> MinMax
            default -> "minmax";
        };
    }

    /**
     * Entraîne un modèle LSTM pour prédire la prochaine clôture (ou log-return).
     *
     * Cette méthode implémente le pipeline complet d'entraînement d'un réseau LSTM :
     * - Validation et préparation des données d'entrée
     * - Construction de séquences temporelles glissantes (sliding windows)
     * - Création des labels de prédiction (prix futurs ou log-returns)
     * - Normalisation feature par feature avec scalers dédiés
     * - Transformation en tenseurs ND4J avec permutation des dimensions
     * - Initialisation et configuration du modèle neuronal
     * - Boucle d'entraînement avec monitoring des performances
     *
     * Architecture des données :
     * - Input : séquences de longueur windowSize avec numFeatures par pas de temps
     * - Output : prédiction scalaire (prix futur ou log-return)
     * - Normalisation : MinMax pour prix/volumes, ZScore pour oscillateurs
     * - Dimensions finales : [batch, features, time] (après permutation)
     *
     * Points critiques :
     * - L'ordre des permutations doit être cohérent avec initModel & LastTimeStep
     * - Chaque feature a son scaler dédié pour une normalisation optimale
     * - La reproductibilité dépend des seeds globales fixées en amont
     *
     * Pipeline:
     *  1. Validation features (fallback 'close' si vide)
     *  2. Construction de séquences glissantes (fenêtres)
     *  3. Création labels (prix futur ou log-return)
     *  4. Normalisation feature par feature + label
     *  5. Transformation en NDArray & permutation => [batch, features, time]
     *  6. Initialisation du modèle
     *  7. Fit sur n epochs (score loggé)
     *
     * ATTENTION: Toute modification de l'ordre des dimensions doit être
     * alignée avec initModel & LastTimeStep.
     *
     * @param series Série d'entraînement (historique OHLCV complet)
     * @param config Configuration LSTM (hyperparamètres, features, normalisation)
     * @param unused Paramètre non utilisé (maintenu pour compatibilité legacy)
     * @return Résultat contenant modèle entraîné + scalers pour inversion/prédiction
     */
    public TrainResult trainLstmScalarV2(BarSeries series, LstmConfig config, Object unused) {
        logger.info("[TRAIN][ENV] Backend={} dtype={}", Nd4j.getExecutioner().getClass().getName(), Nd4j.defaultFloatingPointType()); // Étape 1 log
        // ===== PHASE 1: VALIDATION ET PRÉPARATION DES FEATURES =====

        // Récupération de la liste des features à utiliser pour l'entraînement
        // Features = indicateurs techniques (close, rsi, sma, macd, etc.)
        List<String> features = config.getFeatures();

        // Sécurité : si aucune feature spécifiée, utilise 'close' par défaut
        // Évite les erreurs fatales et garantit au minimum le prix de clôture
        if (features == null || features.isEmpty()) {
            logger.error("[TRAIN] Liste de features vide/null -> fallback ['close']");
            features = java.util.List.of("close"); // Fallback sécurisé sur le prix de clôture
            config.setFeatures(new java.util.ArrayList<>(features)); // Mise à jour de la config
        }

        // Extraction des paramètres de base depuis la configuration
        int windowSize = config.getWindowSize();    // Taille de la fenêtre temporelle (ex: 30 bars)
        int numFeatures = features.size();          // Nombre d'indicateurs à utiliser

        // Validation critique : vérifier qu'on a au moins une feature
        if (numFeatures <= 0) {
            throw new IllegalStateException("numFeatures=0 après fallback, config=" + config.getWindowSize());
        }

        // Nombre total de barres (chandelles) dans la série historique
        int barCount = series.getBarCount();

        // Vérification des données suffisantes pour construire au moins une séquence complète
        // Il faut windowSize barres pour l'input + au moins 1 barre pour le label
        if (barCount <= windowSize + 1) {
            logger.warn("[TRAIN] Données insuffisantes: {} barres pour windowSize={}", barCount, windowSize);
            return new TrainResult(null, null); // Échec : pas assez de données
        }

        // ===== PHASE 2: EXTRACTION DES DONNÉES BRUTES =====

        // Construction de la matrice complète des features [barCount][numFeatures]
        // Chaque ligne = une barre temporelle, chaque colonne = une feature (close, rsi, etc.)
        double[][] matrix = extractFeatureMatrix(series, features);

        // Extraction séparée des prix de clôture pour construction des labels
        double[] closes = extractCloseValues(series);

        // ===== PHASE 3: CONSTRUCTION DES SÉQUENCES D'ENTRAÎNEMENT =====

        // Calcul du nombre de séquences d'entraînement possibles
        // On a besoin de windowSize barres pour l'input + 1 barre pour le label
        int numSeq = barCount - windowSize - 1;

        // Création des tenseurs d'entrée : [numSeq][windowSize][numFeatures]
        // Chaque séquence = windowSize pas de temps avec numFeatures par pas
        double[][][] inputSeq = new double[numSeq][windowSize][numFeatures];

        // Création du vecteur des labels : [numSeq]
        // Chaque label = valeur à prédire pour la séquence correspondante
        double[] labelSeq = new double[numSeq];

        // Construction des séquences par fenêtre glissante (sliding window)
        for (int i = 0; i < numSeq; i++) {
            // Pour chaque séquence i, copier windowSize barres consécutives
            for (int j = 0; j < windowSize; j++) {
                System.arraycopy(matrix[i + j], 0, inputSeq[i][j], 0, numFeatures);
            }

            // ===== CONSTRUCTION DU LABEL (CIBLE DE PRÉDICTION) =====
            if (config.isUseLogReturnTarget()) {
                if (config.isUseMultiHorizonAvg()) { // Correction ici
                    // Mode multi-horizon : moyenne des log-returns t+1..t+H
                    int H = config.getHorizonBars();
                    double prev = closes[i + windowSize - 1];
                    double sumLogRet = 0.0;
                    int count = 0;
                    for (int h = 1; h <= H; h++) {
                        int idx = i + windowSize - 1 + h;
                        if (idx < closes.length) {
                            double next = closes[idx];
                            double logRet = Math.log(next / prev);
                            sumLogRet += logRet;
                            prev = next;
                            count++;
                        }
                    }
                    labelSeq[i] = (count > 0) ? (sumLogRet / count) : 0.0;
                } else {
                    // Mode log-return simple : t+1
                    double prev = closes[i + windowSize - 1];
                    double next = closes[i + windowSize];
                    labelSeq[i] = Math.log(next / prev);
                }
            } else {
                // Mode prix direct : on prédit directement le prix futur
                labelSeq[i] = closes[i + windowSize];
            }
        }

        // ===== PHASE 4: APPRENTISSAGE DES SCALERS (NORMALISATION) =====

        // Création du conteneur pour tous les scalers de normalisation
        ScalerSet scalers = new ScalerSet();

        // Construction d'un scaler dédié pour chaque feature
        for (int f = 0; f < numFeatures; f++) {
            // Extraction de toutes les valeurs historiques pour cette feature
            // Utilise toutes les barres disponibles (pas seulement les séquences)
            double[] col = new double[numSeq + windowSize];
            for (int i = 0; i < numSeq + windowSize; i++) {
                col[i] = matrix[i][f]; // Toutes les valeurs de la feature f
            }

            // Sélection du type de normalisation selon la nature de la feature
            // RSI, MACD, momentum -> Z-Score (centré-réduit)
            // Prix, volumes, ATR -> Min-Max (0-1)
            FeatureScaler.Type type =
                getFeatureNormalizationType(features.get(f)).equals("zscore")
                    ? FeatureScaler.Type.ZSCORE    // Normalisation (x - mean) / std
                    : FeatureScaler.Type.MINMAX;   // Normalisation (x - min) / (max - min)

            // Création et apprentissage du scaler sur les données historiques
            FeatureScaler scaler = new FeatureScaler(type);
            scaler.fit(col); // Calcule min/max ou mean/std selon le type

            // Stockage du scaler avec le nom de la feature comme clé
            scalers.featureScalers.put(features.get(f), scaler);
        }

        // Création du scaler pour les labels:
        //  - Si prédiction de log-return => ZSCORE (évite compression extrême autour de 0)
        //  - Sinon (prix direct) => MINMAX
        FeatureScaler.Type labelType = config.isUseLogReturnTarget()
            ? FeatureScaler.Type.ZSCORE
            : FeatureScaler.Type.MINMAX;
        FeatureScaler labelScaler = new FeatureScaler(labelType);
        labelScaler.fit(labelSeq);
        scalers.labelScaler = labelScaler; // Stockage pour utilisation ultérieure
        // Étape 22: stocker distribution label brute (log-return ou moyenne multi-horizon) pour dérive future
        if (config.isUseLogReturnTarget()) {
            double m = 0; for (double v : labelSeq) m += v; m = labelSeq.length>0? m/labelSeq.length:0;
            double var=0; for (double v: labelSeq){ double d=v-m; var+=d*d; } var = labelSeq.length>0? var/labelSeq.length:0; double std = Math.sqrt(var);
            scalers.labelDistMean = m;
            scalers.labelDistStd = std>1e-12? std : 1e-12; // éviter division par 0
            logger.info("[STEP22][LABEL_DIST][TRAIN] mean={} std={}", String.format(java.util.Locale.US, "%.6f", scalers.labelDistMean), String.format(java.util.Locale.US, "%.6f", scalers.labelDistStd));
        }

        // Vérification qualité normalisation label (std ≈ 1 si ZSCORE)
        double[] normLabels = scalers.labelScaler.transform(labelSeq);
        if (labelType == FeatureScaler.Type.ZSCORE) {
            double m=0, v=0; int n=normLabels.length;
            for(double d: normLabels) m += d; m = n>0? m/n:0;
            for(double d: normLabels) v += (d-m)*(d-m); v = n>0? v/n:0; double std = Math.sqrt(v);
            if (std < 1e-3) {
                logger.warn("[TRAIN][LABEL][WARN] Std normalisée très faible (<1e-3) => plateau potentiel. std={}", std);
            } else {
                logger.debug("[TRAIN][LABEL] Normalisation label ZSCORE ok. mean={} std={}", String.format("%.4f", m), String.format("%.4f", std));
            }
        }

        // ===== PHASE 5: NORMALISATION DES SÉQUENCES D'ENTRAÎNEMENT =====

        // Création du tenseur normalisé avec mêmes dimensions que l'original
        double[][][] normSeq = new double[numSeq][windowSize][numFeatures];

        // Application de la normalisation séquence par séquence
        for (int i = 0; i < numSeq; i++) {          // Pour chaque séquence
            for (int j = 0; j < windowSize; j++) {   // Pour chaque pas de temps
                for (int f = 0; f < numFeatures; f++) { // Pour chaque feature
                    // Application du scaler spécifique à cette feature
                    // transform() retourne un tableau, on prend le premier élément [0]
                    normSeq[i][j][f] =
                        scalers.featureScalers
                            .get(features.get(f))                                    // Récupère le scaler
                            .transform(new double[]{inputSeq[i][j][f]})[0];         // Normalise la valeur
                }
            }
        }

        // ===== PHASE 6: CONVERSION EN TENSEURS ND4J =====

        // Conversion du tableau Java en tenseur ND4J (format DeepLearning4J)
        // Dimensions initiales : [batch, time, features] (format standard séquentiel)
        org.nd4j.linalg.api.ndarray.INDArray X = Nd4j.create(normSeq);

        // PERMUTATION CRITIQUE : [batch, time, features] -> [batch, features, time]
        // Cette permutation est nécessaire pour la compatibilité avec l'architecture LSTM
        // et la couche LastTimeStep utilisée dans initModel()
        X = X.permute(0, 2, 1).dup('c'); // dup('c') = copie contiguë en mémoire

        // Vérification de cohérence des dimensions après permutation
        // Détection précoce d'erreurs de shape qui causeraient des échecs silencieux
        if (X.size(1) != numFeatures || X.size(2) != windowSize) {
            logger.warn("[SHAPE][TRAIN] Incohérence shape après permute: expected features={} time={} got features={} time={}",
                numFeatures, windowSize, X.size(1), X.size(2));
        }

        // Normalisation des labels avec le scaler dédié
        // (normLabels déjà calculé ci-dessus si besoin mais recalcul léger ok)
        normLabels = scalers.labelScaler.transform(labelSeq);

        // Conversion des labels en tenseur ND4J : [numSeq, 1] (régression scalaire)
        org.nd4j.linalg.api.ndarray.INDArray y = Nd4j.create(normLabels, new long[]{numSeq, 1});

        // ===== PHASE 7: INITIALISATION DU MODÈLE NEURONAL =====

        // Récupération du nombre effectif de features après permutation
        // (sécurité au cas où la permutation aurait altéré les dimensions)
        int effectiveFeatures = (int) X.size(1);

        // Détection et log des incohérences de dimensions
        if (effectiveFeatures != numFeatures) {
            logger.warn("[INIT][ADAPT] numFeatures déclaré={} mais tensor features={} => reconstruction modèle",
                numFeatures, effectiveFeatures);
        }

        // Construction du modèle LSTM avec l'architecture spécifiée dans la config
        MultiLayerNetwork model = initModel(
            effectiveFeatures,              // Nombre de features d'entrée (après vérification)
            1,                             // Taille de sortie (1 pour régression scalaire)
            config.getLstmNeurons(),       // Nombre de neurones par couche LSTM
            config.getDropoutRate(),       // Taux de dropout pour régularisation
            config.getLearningRate(),      // Taux d'apprentissage de l'optimiseur
            config.getOptimizer(),         // Type d'optimiseur (Adam, RMSprop, SGD)
            config.getL1(),                // Régularisation L1 (sparsité)
            config.getL2(),                // Régularisation L2 (poids petits)
            config,                        // Configuration complète (couches, attention, etc.)
            false                          // Mode régression (pas classification)
        );

        // Log détaillé des dimensions pour debug et monitoring
        logger.debug("[TRAIN] X shape={} (batch={} features={} time={}) y shape={} expectedFeatures={} lstmNeurons={}",
            Arrays.toString(X.shape()), X.size(0), X.size(1), X.size(2),
            Arrays.toString(y.shape()), numFeatures, config.getLstmNeurons());

        // ===== PHASE 8: PRÉPARATION DES DONNÉES POUR L'ENTRAÎNEMENT =====

        // Création du DataSet DeepLearning4J (associe inputs X et labels y)
        org.nd4j.linalg.dataset.DataSet ds = new org.nd4j.linalg.dataset.DataSet(X, y);

        // ===== Étape 11: Split interne validation (derniers 15% des séquences) =====
        int numSeqTotal = (int) X.size(0);
        int valCount = (int) Math.round(numSeqTotal * 0.15);
        if (valCount < 1 && numSeqTotal >= 20) valCount = 1; // sécurité minimale
        int trainCount = numSeqTotal - valCount;
        boolean useInternalVal = trainCount > 1 && valCount >= 1;
        org.nd4j.linalg.api.ndarray.INDArray XTrain, yTrain, XVal = null, yVal = null;
        if (useInternalVal) {
            XTrain = X.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(0, trainCount),
                org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup('c');
            yTrain = y.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(0, trainCount), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup('c');
            XVal = X.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(trainCount, numSeqTotal),
                org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup('c');
            yVal = y.get(org.nd4j.linalg.indexing.NDArrayIndex.interval(trainCount, numSeqTotal), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup('c');
        } else {
            // Fallback: tout en train (ancien comportement)
            XTrain = X;
            yTrain = y;
            logger.warn("[TRAIN][VAL][FALLBACK] Validation interne désactivée (trainCount={} valCount={} total={})", trainCount, valCount, numSeqTotal);
        }
        org.nd4j.linalg.dataset.DataSet trainDs = new org.nd4j.linalg.dataset.DataSet(XTrain, yTrain);
        org.nd4j.linalg.dataset.api.iterator.DataSetIterator iterator =
            new ListDataSetIterator<>(trainDs.asList(), config.getBatchSize());
        org.nd4j.linalg.dataset.DataSet valDs = null;
        if (useInternalVal) {
            valDs = new org.nd4j.linalg.dataset.DataSet(XVal, yVal);
            logger.info("[TRAIN][VAL] Activation split validation interne 85%/15% (train={} val={})", trainCount, valCount);
        }

        // ===== PHASE 9: BOUCLE D'ENTRAÎNEMENT PRINCIPALE =====

        // Récupération du nombre d'époques depuis la configuration
        int epochs = config.getNumEpochs();

        // Timestamp de début pour mesure de performance globale
        long t0 = System.currentTimeMillis();

        // Variables pour Early Stopping et sauvegarde du meilleur modèle
        double bestScore = Double.POSITIVE_INFINITY; // score train (fallback)
        double bestValLoss = Double.POSITIVE_INFINITY; // suivi validation
        MultiLayerNetwork bestModel = null;              // Sauvegarde du meilleur modèle
        int patience = config.getPatience();             // Patience train (fallback)
        double minDelta = config.getMinDelta();          // Amélioration minimale
        int epochsWithoutImprovement = 0;               // Compteur train
        int epochsWithoutValImprovement = 0;            // Compteur validation
        int patienceVal = config.getPatienceVal() > 0 ? config.getPatienceVal() : 5;
        Double lastValLoss = null; // Step13: stocke le dernier valLoss

        // Étape 10: holder baseline variance résiduelle pour suivi amélioration HUBER
        Double[] baselineResidualVarHolder = new Double[]{null};

        // Étape 12: Variables scheduler LR (réduction multiplicative simple toutes les 25 epochs)
        double currentLearningRate = config.getLearningRate();
        boolean lrReducedOnce = false;                        // Indique si une première réduction a eu lieu
        double bestValLossAtFirstLrReduction = Double.NaN;    // Snapshot du bestValLoss au moment de la 1ère réduction
        boolean lrFirstReductionImproved = false;             // Flag acceptation: amélioration après 1ère réduction

        // Boucle d'entraînement epoch par epoch
        for (int epoch = 1; epoch <= epochs; epoch++) {
            iterator.reset();
            model.fit(iterator);
            // Calcul des losses train & validation après fit (cohérentes avec état courant)
            double trainLoss = model.score(trainDs);
            Double valLoss = null;
            boolean valImproved = false;
            boolean acceptanceValBetterThanTrain = false;
            if (useInternalVal) {
                valLoss = model.score(valDs);
                lastValLoss = valLoss; // Step13: mise à jour dernier valLoss
                acceptanceValBetterThanTrain = valLoss < (trainLoss + minDelta);
                valImproved = (bestValLoss - valLoss) > minDelta;
                if (valImproved) {
                    bestValLoss = valLoss;
                    epochsWithoutValImprovement = 0;
                    if (bestModel == null) {
                        bestModel = model.clone();
                    } else {
                        bestModel.setParams(model.params().dup());
                    }
                } else {
                    epochsWithoutValImprovement++;
                }
                // Étape 12: vérification acceptation amélioration après première réduction
                if (lrReducedOnce && !lrFirstReductionImproved && valLoss != null && Double.isFinite(bestValLossAtFirstLrReduction)) {
                    if ((bestValLossAtFirstLrReduction - valLoss) > minDelta) {
                        lrFirstReductionImproved = true;
                        logger.info("[TRAIN][LR-SCHED][ACCEPT] Amélioration val_loss après 1ère réduction LR: avant={} après={}",
                            String.format(Locale.US, "%.6f", bestValLossAtFirstLrReduction),
                            String.format(Locale.US, "%.6f", valLoss));
                    }
                }
            } else {
                // Fallback ancien critère train
                double score = trainLoss;
                boolean hasImproved = (bestScore - score) > minDelta;
                if (hasImproved) {
                    bestScore = score;
                    epochsWithoutImprovement = 0;
                    if (bestModel == null) bestModel = model.clone(); else bestModel.setParams(model.params().dup());
                } else {
                    epochsWithoutImprovement++;
                }
            }

            // Logging variance résiduelle (conserve bloc existant simplifié)
            org.nd4j.linalg.api.ndarray.INDArray predsTrainDebug = null;
            try { predsTrainDebug = model.output(XTrain, false); } catch (Exception ignored) {}
            if (predsTrainDebug != null) {
                org.nd4j.linalg.api.ndarray.INDArray residualsDbg = yTrain.sub(predsTrainDebug);
                double residualVar = residualsDbg.varNumber().doubleValue();
                if (epoch == 1) {
                    logger.info("[TRAIN][HUBER] Var résiduelle train epoch1={} ", String.format(Locale.US, "%.6f", residualVar));
                }
            }

            // Étape 12: Scheduler LR simple (appliqué APRÈS calcul des losses de l'époque -> agit pour l'époque suivante)
            if (epoch % 25 == 0) {
                double newLR = currentLearningRate * 0.9; // réduction multiplicative
                // Floor de sécurité pour éviter LR trop bas (optionnel)
                if (newLR < 1e-6) newLR = 1e-6;
                if (!lrReducedOnce) {
                    lrReducedOnce = true;
                    bestValLossAtFirstLrReduction = bestValLoss; // snapshot pour critère Acceptation
                }
                if (Math.abs(newLR - currentLearningRate) > 1e-12) {
                    applyLearningRate(model, newLR);
                    logger.info("[TRAIN][LR-SCHED] Réduction LR epoch={} ancienLR={} nouveauLR={}", epoch,
                        String.format(Locale.US, "%.8f", currentLearningRate),
                        String.format(Locale.US, "%.8f", newLR));
                    currentLearningRate = newLR;
                }
            }

            // Early stopping conditions
            boolean stop = false;
            if (useInternalVal) {
                if (patienceVal > 0 && epochsWithoutValImprovement >= patienceVal) {
                    logger.info("[TRAIN][EARLY-STOP][VAL] Arrêt anticipé epoch={} (patienceVal={} sans amélioration) bestValLoss={}", epoch, patienceVal, String.format(Locale.US, "%.6f", bestValLoss));
                    stop = true;
                }
            } else {
                if (patience > 0 && epochsWithoutImprovement >= patience) {
                    logger.info("[TRAIN][EARLY-STOP][TRAIN] Arrêt anticipé epoch={} (patience={} sans amélioration) bestScore={}", epoch, patience, String.format(Locale.US, "%.6f", bestScore));
                    stop = true;
                }
            }

            // Logging périodique enrichi
            if (epoch == 1 || epoch == epochs || epoch % Math.max(1, epochs / 10) == 0 || stop) {
                Runtime rt = Runtime.getRuntime();
                long used = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024);
                long max = rt.maxMemory() / (1024 * 1024);
                long elapsed = System.currentTimeMillis() - t0;
                if (useInternalVal) {
                    logger.info("[TRAIN][EPOCH][VAL] {}/{} trainLoss={} valLoss={} bestVal={} noValImprove={}/{} acceptValBetterTrain={} improved={} lr={} reducedOnce={} lrAccept={} elapsedMs={} memMB={}/{}", epoch, epochs,
                        String.format(Locale.US, "%.6f", trainLoss),
                        String.format(Locale.US, "%.6f", valLoss),
                        String.format(Locale.US, "%.6f", bestValLoss),
                        epochsWithoutValImprovement, patienceVal,
                        acceptanceValBetterThanTrain,
                        valImproved,
                        String.format(Locale.US, "%.8f", currentLearningRate),
                        lrReducedOnce,
                        lrFirstReductionImproved,
                        elapsed, used, max);
                } else {
                    logger.info("[TRAIN][EPOCH] {}/{} trainLoss={} bestTrain={} noImprove={}/{} lr={} elapsedMs={} memMB={}/{}", epoch, epochs,
                        String.format("%.6f", trainLoss),
                        String.format("%.6f", bestScore),
                        epochsWithoutImprovement, patience,
                        String.format(Locale.US, "%.8f", currentLearningRate),
                        elapsed, used, max);
                }
            }
            if (stop) break;
        }

        // ===== PHASE 10: SÉLECTION DU MODÈLE FINAL =====
        MultiLayerNetwork finalModel = bestModel != null ? bestModel : model;
        if (bestModel != null) {
            if (useInternalVal) {
                logger.info("[TRAIN][FINAL][VAL] Retour meilleur modèle validation (bestValLoss={})", String.format(Locale.US, "%.6f", bestValLoss));
            } else {
                logger.info("[TRAIN][FINAL] Retour meilleur modèle train (bestScore={})", String.format(Locale.US, "%.6f", bestScore));
            }
        } else {
            logger.warn("[TRAIN][FINAL] Aucun modèle amélioré sauvegardé (fallback dernier état)");
        }

        // ===== STEP 13: ACCEPTATION (best model vs dernier) =====
        boolean bestBetterThanLast = false;
        if (useInternalVal && Double.isFinite(bestValLoss) && lastValLoss != null && Double.isFinite(lastValLoss)) {
            // Critère strict: meilleure epoch différente (strictement plus basse)
            bestBetterThanLast = (bestValLoss + minDelta) < lastValLoss;
            logger.info("[TRAIN][ACCEPT][STEP13] bestValLoss={} lastValLoss={} improvedVsLast={}",
                String.format(Locale.US, "%.6f", bestValLoss),
                String.format(Locale.US, "%.6f", lastValLoss),
                bestBetterThanLast);
            // Compteurs globaux (static) pour ratio >=40%
            try {
                Step13Stats.totalRuns.incrementAndGet();
                if (bestBetterThanLast) Step13Stats.improvedRuns.incrementAndGet();
                int tot = Step13Stats.totalRuns.get();
                int imp = Step13Stats.improvedRuns.get();
                double ratio = tot > 0 ? (100.0 * imp / tot) : 0.0;
                logger.info("[TRAIN][ACCEPT][STEP13][RATIO] improvedRuns={}/{} ({}%)", imp, tot, String.format(Locale.US, "%.2f", ratio));
            } catch (Exception ignore) {}
        } else if (useInternalVal) {
            logger.info("[TRAIN][ACCEPT][STEP13] Données insuffisantes pour calcul acceptation (bestValLoss={} lastValLoss={})", bestValLoss, lastValLoss);
        }

        // ===== STEP 21: VARIANCE DES PRÉDICTIONS (détection modèle plat) =====
        Double residualVarStep21 = null; boolean flatAlert=false;
        try {
            // Recalcule prédictions sur l'ensemble train (XTrain ou X si fallback) avec modèle final retenu
            org.nd4j.linalg.api.ndarray.INDArray XForVar = (useInternalVal ? XTrain : X); // XTrain défini plus haut
            if (XForVar != null && XForVar.size(0) > 5) {
                org.nd4j.linalg.api.ndarray.INDArray predNormAll = finalModel.output(XForVar, false);
                int nPred = (int) predNormAll.size(0);
                double[] residuals = new double[nPred];
                for (int i = 0; i < nPred; i++) {
                    double predNorm = predNormAll.getDouble(i);
                    double predTarget = scalers.labelScaler.inverse(predNorm);
                    // Reconstruction prix futur préd it
                    // Index i correspond à séquence i: fenêtre couvrant closes[i .. i+windowSize-1], target = closes[i+windowSize]
                    int baseIdx = i + windowSize - 1;
                    int futureIdx = i + windowSize;
                    if (futureIdx >= closes.length) break; // sécurité
                    double baseClose = closes[baseIdx];
                    double trueClose = closes[futureIdx];
                    double predictedClose;
                    if (config.isUseLogReturnTarget()) {
                        // Si multi-horizon moyenne, on traite la moyenne comme log-return agrégé (approximation acceptable pour alerte platitude)
                        predictedClose = baseClose * Math.exp(predTarget);
                    } else {
                        predictedClose = predTarget; // cible prix direct
                    }
                    double resid = predictedClose - trueClose;
                    if (!Double.isFinite(resid)) resid = 0.0;
                    residuals[i] = resid;
                }
                // Calcul variance
                int m = 0; double meanR = 0.0; for (double r : residuals){ if (Double.isFinite(r)){ meanR += r; m++; } }
                if (m > 0) meanR /= m; double var=0; for (double r: residuals){ if (Double.isFinite(r)){ double d=r-meanR; var += d*d; }}
                if (m > 0) var /= m; residualVarStep21 = var;
                double seuil = config.getPredictionResidualVarianceMin();
                if (var < seuil) {
                    flatAlert = true;
                    logger.warn("[TRAIN][STEP21][ALERT] Variance résiduelle trop faible var={} < seuil={} => modèle potentiellement trop plat", String.format(Locale.US, "%.8g", var), String.format(Locale.US, "%.8g", seuil));
                } else {
                    logger.info("[TRAIN][STEP21] Variance résiduelle train var={} (seuil={})", String.format(Locale.US, "%.8g", var), String.format(Locale.US, "%.8g", seuil));
                }
            } else {
                logger.debug("[TRAIN][STEP21] Skip calcul variance (jeu train insuffisant)");
            }
        } catch (Exception e) {
            logger.error("[TRAIN][STEP21][FAIL] {}", e.getMessage());
        }

        // ===== PHASE 11: RETOUR DU RÉSULTAT FINAL =====
        TrainResult tr = new TrainResult(finalModel, scalers, useInternalVal? bestValLoss: null, useInternalVal? lastValLoss: null, bestBetterThanLast);
        tr.residualVariance = residualVarStep21; tr.flatModelAlert = flatAlert;
        return tr;
    }

    /* =========================================================
     *                    PREDICTION (INFERENCE)
     * =========================================================
     * Processus:
     *  - Recalcule toutes les features sur la série complète (peut être optimisable).
     *  - Applique les scalers (reconstruction si manquants).
     *  - Construit la dernière fenêtre (taille windowSize).
     *  - Permute pour [1, features, time].
     *  - Fait un forward pass et inverse la normalisation.
     *  - Si label = log-return => reconstruit prix prédictif.
     */

    /**
     * Prédit la prochaine clôture (ou log-return reconverti en prix) en utilisant un modèle LSTM entraîné.
     *
     * Cette méthode implémente le pipeline complet d'inférence LSTM pour prédire le prix futur :
     * - Validation des données d'entrée et des paramètres
     * - Reconstruction/validation des scalers de normalisation
     * - Extraction et normalisation des features sur toute la série
     * - Construction de la dernière séquence temporelle (fenêtre de prédiction)
     * - Transformation en tenseur ND4J avec permutation des dimensions
     * - Exécution du forward pass (inférence neuronale)
     * - Dénormalisation et reconversion vers le domaine des prix
     * - Application de limitations pour éviter les prédictions aberrantes
     *
     * Architecture du processus :
     * 1. Validation des prérequis (données, scalers, taille)
     * 2. Extraction complète des features sur la série
     * 3. Normalisation feature par feature avec les scalers d'entraînement
     * 4. Construction de la séquence d'inférence (derniers windowSize points)
     * 5. Permutation des dimensions : [batch, time, features] -> [batch, features, time]
     * 6. Forward pass neuronal pour obtenir la prédiction normalisée
     * 7. Dénormalisation avec le label scaler
     * 8. Reconversion log-return -> prix si nécessaire
     * 9. Application des limites de sécurité
     *
     * Points critiques :
     * - Les scalers doivent être cohérents avec ceux utilisés à l'entraînement
     * - La permutation des dimensions doit correspondre à l'architecture du modèle
     * - La reconversion log-return/prix doit être cohérente avec l'entraînement
     * - Les limitations évitent les prédictions économiquement impossibles
     *
     * Gestion d'erreurs :
     * - Reconstruction automatique des scalers si incohérents
     * - Validation des dimensions après permutation
     * - Limitation des prédictions aberrantes
     *
     * @param series  Série temporelle complète (la dernière barre = point de prédiction actuel)
     * @param config  Configuration LSTM (fenêtre, features, normalisation, limitations)
     * @param model   Modèle LSTM déjà entraîné et prêt pour l'inférence
     * @param scalers Scalers de normalisation utilisés lors de l'entraînement
     * @return Prix prédit ajusté et limité pour éviter les aberrations
     */
    public double predictNextCloseScalarV2(BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers) {
        // ===== PHASE 1: EXTRACTION DES PARAMÈTRES DE CONFIGURATION =====

        // Récupération de la liste des features utilisées par le modèle
        // Doit être identique à celle utilisée lors de l'entraînement
        List<String> features = config.getFeatures();

        // Taille de la fenêtre temporelle (nombre de pas de temps en entrée)
        int windowSize = config.getWindowSize();

        // ===== VALIDATION DES DONNÉES SUFFISANTES =====
        // Vérification qu'on a assez de barres pour construire une séquence complète
        // Il faut au minimum windowSize + 1 barres (windowSize pour l'input + contexte)
        if (series.getBarCount() <= windowSize) {
            throw new IllegalArgumentException("Pas assez de barres: " + series.getBarCount() + " <= " + windowSize);
        }

        // ===== PHASE 2: VALIDATION ET RECONSTRUCTION DES SCALERS =====

        // Vérification de la cohérence des scalers avec la configuration actuelle
        // Les scalers sont critiques car ils doivent être identiques à ceux de l'entraînement
        if (scalers == null                                              // Scalers complètement absents
            || scalers.featureScalers == null                          // Map des scalers de features nulle
            || scalers.featureScalers.size() != features.size()        // Nombre de scalers != nombre de features
            || scalers.labelScaler == null) {                          // Scaler de label absent

            logger.warn("[PREDICT] Scalers null/incomplets -> rebuild automatique");

            // Reconstruction d'urgence des scalers basée sur la série actuelle
            // ATTENTION: peut différer des scalers d'entraînement original
            scalers = rebuildScalers(series, config);
        }

        // Migration rétro-compatible: anciens modèles log-return avec labelScaler MINMAX -> reconstruire ZSCORE
        if (config.isUseLogReturnTarget() && scalers != null && scalers.labelScaler != null &&
            scalers.labelScaler.type == FeatureScaler.Type.MINMAX) {
            try {
                logger.warn("[MIGRATION][LABEL_SCALER] Ancien labelScaler MINMAX détecté pour log-return -> reconstruction ZSCORE");
                scalers.labelScaler = rebuildLabelScalerForLogReturn(series, config);
            } catch (Exception e) {
                logger.error("[MIGRATION][LABEL_SCALER][FAIL] {}", e.getMessage());
            }
        }

        // ===== PHASE 3: EXTRACTION DE LA MATRICE DE FEATURES COMPLÈTE =====

        // Calcul de toutes les features sur l'intégralité de la série temporelle
        // Nécessaire car la normalisation s'applique sur toute la série
        double[][] matrix = extractFeatureMatrix(series, features);

        // Nombre de features à traiter (doit correspondre aux scalers)
        int numFeatures = features.size();

        // ===== PHASE 4: NORMALISATION FEATURE PAR FEATURE =====

        // Création de la matrice normalisée avec mêmes dimensions que l'originale
        double[][] normMatrix = new double[matrix.length][numFeatures];

        // Normalisation colonne par colonne (une feature = une colonne)
        for (int f = 0; f < numFeatures; f++) {
            // Extraction de toutes les valeurs historiques pour cette feature
            double[] col = new double[matrix.length];
            for (int i = 0; i < matrix.length; i++) {
                col[i] = matrix[i][f]; // Copie de la colonne f dans un tableau temporaire
            }

            // Application du scaler spécifique à cette feature
            // Le scaler a été appris lors de l'entraînement avec fit()
            double[] normCol = scalers.featureScalers.get(features.get(f)).transform(col);

            // Recopie de la colonne normalisée dans la matrice finale
            for (int i = 0; i < matrix.length; i++) {
                normMatrix[i][f] = normCol[i];
            }
        }

        // ===== PHASE 5: CONSTRUCTION DE LA SÉQUENCE D'INFÉRENCE =====

        // Création du tenseur d'entrée : [1 batch][windowSize time][numFeatures]
        // Batch size = 1 car on prédit un seul point à la fois
        double[][][] seq = new double[1][windowSize][numFeatures];

        // Extraction des derniers windowSize points normalisés pour la prédiction
        for (int j = 0; j < windowSize; j++) {
            // Copie des features de la barre (length - windowSize + j)
            System.arraycopy(
                normMatrix[normMatrix.length - windowSize + j],
                0,
                seq[0][j],
                0,
                numFeatures
            );
        }

        // ===== PHASE 6: TRANSFORMATION EN TENSEUR ND4J AVEC PERMUTATION =====

        // Conversion en tenseur ND4J : [batch, time, features] (format initial)
        // Puis permutation vers [batch, features, time] pour compatibilité modèle
        org.nd4j.linalg.api.ndarray.INDArray input =
            Nd4j.create(seq)                    // Création tenseur initial [1, windowSize, numFeatures]
                .permute(0, 2, 1)              // Permutation: [batch, time, features] -> [batch, features, time]
                .dup('c');                     // Copie contiguë en mémoire pour performance

        // ===== VALIDATION DES DIMENSIONS APRÈS PERMUTATION =====
        // Vérification critique pour détecter les erreurs de shape
        if (input.size(1) != numFeatures || input.size(2) != windowSize) {
            logger.warn("[SHAPE][PRED] Incohérence shape input: expected features={} time={} got features={} time={}",
                numFeatures, windowSize, input.size(1), input.size(2));
        }

        // ===== PHASE 7: EXÉCUTION DU FORWARD PASS (INFÉRENCE) =====

        // Passage des données dans le réseau neuronal pour obtenir la prédiction
        // model.output() effectue la forward pass complète : LSTM + Dense + Output
        double predNorm = model.output(input).getDouble(0); // Index 0 car sortie scalaire

        // ===== PHASE 8: DÉNORMALISATION DE LA PRÉDICTION =====

        // Inversion de la normalisation pour repasser dans le domaine original
        // Utilise le label scaler qui a été appris sur les targets d'entraînement
        double predTarget = scalers.labelScaler.inverse(predNorm);

        // ===== PHASE 9: RÉCUPÉRATION DU PRIX DE RÉFÉRENCE COHÉRENT =====

        // CORRECTION: Pour cohérence avec l'entraînement, utilise le dernier prix de la séquence d'entrée
        // Pendant l'entraînement: prev = closes[i + windowSize - 1] (dernier prix de la fenêtre)
        // Pendant la prédiction: doit utiliser le même principe
        double[] closes = extractCloseValues(series);
        double referencePrice = closes[closes.length - 1]; // Dernier prix de la série (équivalent au "prev" de l'entraînement)

        // ===== PHASE 10: RECONVERSION VERS LE DOMAINE DES PRIX =====

        // Gestion des deux modes de prédiction selon le type d'entraînement
        double predicted;
        if (config.isUseLogReturnTarget()) {
            // Mode log-return: reconversion exponentielle vers prix absolu
            // predTarget = log(prix_futur / prix_référence) donc prix_futur = prix_référence * exp(predTarget)
            // CORRECTION: Utilise referencePrice au lieu de lastClose pour cohérence temporelle
            predicted = referencePrice * Math.exp(predTarget);

            logger.debug("[PREDICT][LOG-RETURN] referencePrice={}, predTarget={}, exp(predTarget)={}, predicted={}",
                        String.format("%.3f", referencePrice),
                        String.format("%.6f", predTarget),
                        String.format("%.6f", Math.exp(predTarget)),
                        String.format("%.3f", predicted));
        } else {
            // Mode prix direct : la prédiction est déjà dans le domaine des prix
            predicted = predTarget;
        }

        // ===== PHASE 11: APPLICATION DES LIMITATIONS DE SÉCURITÉ =====

        // Récupération du pourcentage de limitation depuis la configuration
        double limitPct = config.getLimitPredictionPct();

        // Application des bornes si limitation activée (limitPct > 0)
        // Note: Les limitations utilisent referencePrice comme base pour cohérence
        if (limitPct > 0) {
            // Calcul des bornes relative au prix de référence
            double min = referencePrice * (1 - limitPct);    // Borne inférieure (ex: -20%)
            double max = referencePrice * (1 + limitPct);    // Borne supérieure (ex: +20%)

            // Écrêtage de la prédiction dans les bornes calculées
            if (predicted < min) {
                logger.debug("[PREDICT] Limitation basse appliquée: {} -> {}", predicted, min);
                predicted = min;
            } else if (predicted > max) {
                logger.debug("[PREDICT] Limitation haute appliquée: {} -> {}", predicted, max);
                predicted = max;
            }
        }

        // ===== PHASE 12: LOG DE DEBUG ET RETOUR =====

        // Log détaillé pour debug et monitoring des prédictions
        logger.debug("[PREDICT] Prédiction: referencePrice={}, predicted={}, predNorm={}, predTarget={}, limitPct={}",
                    String.format("%.3f", referencePrice),
                    String.format("%.3f", predicted),
                    String.format("%.6f", predNorm),
                    String.format("%.6f", predTarget),
                    limitPct);

        // Retour de la prédiction finale ajustée et limitée
        return predicted;
    }

    /**
     * Alias conservant signature existante.
     */
    public double predictNextCloseWithScalerSet(String symbol, BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers){
        return predictNextCloseScalarV2(series, config, model, scalers);
    }

    /**
     * (Étape 6 - Cache) Variante ultra légère de prédiction réutilisant:
     *  - Matrice normalisée complète (normMatrix)
     *  - Tableau des closes (closes)
     * Évite: extractFeatureMatrix + normalisation à chaque appel.
     * Hypothèses: normMatrix et closes couvrent au moins endBarInclusive.
     */
    public double predictNextCloseScalarCached(int endBarInclusive,
                                               double[][] normMatrix,
                                               double[] closes,
                                               LstmConfig config,
                                               MultiLayerNetwork model,
                                               ScalerSet scalers) {
        int windowSize = config.getWindowSize();
        int numFeatures = normMatrix[0].length;
        if (endBarInclusive < windowSize - 1) {
            throw new IllegalArgumentException("endBarInclusive=" + endBarInclusive + " < windowSize-1");
        }
        // Construction séquence [1][window][features]
        double[][][] seq = new double[1][windowSize][numFeatures];
        int start = endBarInclusive - windowSize + 1;
        for (int t = 0; t < windowSize; t++) {
            System.arraycopy(normMatrix[start + t], 0, seq[0][t], 0, numFeatures);
        }
        org.nd4j.linalg.api.ndarray.INDArray input = Nd4j.create(seq).permute(0, 2, 1).dup('c');
        double predNorm = model.output(input).getDouble(0);
        double predTarget = scalers.labelScaler.inverse(predNorm);
        double referencePrice = closes[endBarInclusive];
        double predicted;
        if (config.isUseLogReturnTarget()) {
            predicted = referencePrice * Math.exp(predTarget);
        } else {
            predicted = predTarget;
        }
        double limitPct = config.getLimitPredictionPct();
        if (limitPct > 0) {
            double min = referencePrice * (1 - limitPct);
            double max = referencePrice * (1 + limitPct);
            if (predicted < min) predicted = min; else if (predicted > max) predicted = max;
        }
        return predicted;
    }

    // === Step15: métriques trading & décision contrarian (réintégration) ===
    public static class TradingMetricsV2 implements Serializable {
        public double totalProfit, profitFactor, winRate, maxDrawdownPct, expectancy, sharpe, sortino, exposure, turnover, avgBarsInPosition, mse, businessScore, calmar;
        public int numTrades;
        public int contrarianTrades;
        public int normalTrades;
        public double contrarianRatio;
        // Étape 18: stats position sizing
        public double positionValueMean;
        public double positionValueStd;
    }
    private static class ContrarianDecision { boolean active; double adjustedSignalStrength; String reason; }
    private ContrarianDecision evaluateContrarian(double rawSignalStrength, double lastClose, double[] rsiValues, int bar, BarSeries sub) {
        ContrarianDecision d = new ContrarianDecision(); d.active=false; d.adjustedSignalStrength=rawSignalStrength; if (rawSignalStrength < -0.02) { double currentRsi = rsiValues[bar]; int recentWindow = Math.min(10, sub.getBarCount()); if (recentWindow > 0) { double supportLevel = Double.POSITIVE_INFINITY; for (int i=0;i<recentWindow;i++){ double low = sub.getBar(sub.getBarCount()-1-i).getLowPrice().doubleValue(); if (low < supportLevel) supportLevel = low; } double distanceToSupport = supportLevel>0? (lastClose - supportLevel)/supportLevel : 1.0; if (currentRsi < 35 && distanceToSupport < 0.05 && rawSignalStrength < -0.025) { d.active = true; d.adjustedSignalStrength = Math.abs(rawSignalStrength) * 0.6; d.reason = String.format("RSI=%.1f support=%.3f dist=%.2f%% raw=%.4f", currentRsi, supportLevel, distanceToSupport*100, rawSignalStrength); logger.info("[DEBUG][CONTRARIAN] bar={} rsi={} support={} dist={} signalAdj={}", bar, String.format("%.1f", currentRsi), String.format("%.3f", supportLevel), String.format("%.2f", distanceToSupport*100), String.format("%.4f", d.adjustedSignalStrength)); } } } return d; }

    /* =========================================================
     *                WALK-FORWARD EVALUATION
     * =========================================================
     * Découpage de la série en segments successifs:
     *  - Entraînement sur la partie passée
     *  - Test sur un split futur
     *  - Mesure métriques + MSE
     *  - Calcul d'un businessScore pondéré
     */

    /**
     * Résultat global d'un walk-forward avec statistiques agrégées.
     */
    public static class WalkForwardResultV2 implements Serializable {
        public List<TradingMetricsV2> splits = new ArrayList<>();
        public double meanMse, meanBusinessScore, mseVariance, mseInterModelVariance;
        public int totalTestedBars;
    }


    /**
     * Version de walkForwardEvaluate qui teste UNIQUEMENT sur les données out-of-sample
     * à partir d'un point de départ spécifié (pour éviter le data leakage).
     *
     * @param series série temporelle complète (nécessaire pour le contexte historique)
     * @param config configuration LSTM
     * @param preTrainedModel modèle entraîné uniquement sur la partie train
     * @param preTrainedScalers scalers calculés uniquement sur la partie train
     * @param testStartFromBar index à partir duquel commencer les tests (données non vues)
     * @return résultats d'évaluation walk-forward sur données out-of-sample uniquement
     */
    public WalkForwardResultV2 walkForwardEvaluateOutOfSample(BarSeries series, LstmConfig config,
                                                              MultiLayerNetwork preTrainedModel,
                                                              ScalerSet preTrainedScalers,
                                                              int testStartFromBar) {
        // ===== PHASE 1: INITIALISATION ET VALIDATION =====

        WalkForwardResultV2 result = new WalkForwardResultV2();

        if (preTrainedModel == null || preTrainedScalers == null) {
            logger.error("[WALK-FORWARD-OOS] Modèle ou scalers pré-entraînés manquants");
            return result;
        }

        int windowSize = config.getWindowSize();
        int totalBars = series.getBarCount();

        // Vérifier que nous avons assez de données pour tester
        if (testStartFromBar + windowSize + 10 >= totalBars) {
            logger.warn("[WALK-FORWARD-OOS] Données insuffisantes pour test out-of-sample: testStart={}, totalBars={}",
                       testStartFromBar, totalBars);
            return result;
        }

        // ===== PHASE 2: CALCUL DE LA SEGMENTATION TEMPORELLE OUT-OF-SAMPLE =====

        // Nombre de barres disponibles pour les tests (partie non vue par le modèle)
        int testBarsAvailable = totalBars - testStartFromBar;
        int splits = Math.max(1, Math.min(config.getWalkForwardSplits(), testBarsAvailable / 50)); // Au moins 50 barres par split

        if (splits == 0) {
            logger.warn("[WALK-FORWARD-OOS] Pas assez de données pour créer des splits de test");
            return result;
        }

        int splitSize = testBarsAvailable / splits;
        double sumMse = 0, sumBusiness = 0;
        int mseCount = 0, businessCount = 0;
        List<Double> mseList = new ArrayList<>();

        logger.info("[WALK-FORWARD-OOS] Début évaluation out-of-sample: {} splits sur données [{}, {}]",
                   splits, testStartFromBar, totalBars);

        // ===== PHASE 3: BOUCLE PRINCIPALE SUR CHAQUE SPLIT OUT-OF-SAMPLE =====
        int totalTestedBars = 0;
        for (int s = 1; s <= splits; s++) {
            // Calcul des bornes du split dans la zone out-of-sample uniquement
            int testStartBar = testStartFromBar + (s - 1) * splitSize + config.getEmbargoBars();
            int testEndBar = (s == splits) ? totalBars : testStartFromBar + s * splitSize;
            totalTestedBars = totalTestedBars + testEndBar - testStartBar;
            // Vérification que le split est suffisamment grand
            if (testStartBar + windowSize + 5 >= testEndBar) {
                logger.debug("[WALK-FORWARD-OOS] Split {} trop petit, ignoré", s);
                continue;
            }

            logger.debug("[WALK-FORWARD-OOS] Split {}/{}: test=[{},{}] (out-of-sample uniquement)",
                        s, splits, testStartBar, testEndBar);

            // ===== SIMULATION TRADING SUR DONNÉES NON VUES =====
            // IMPORTANT: testStartBar et testEndBar sont dans la zone [testStartFromBar, totalBars]
            // Ces données n'ont JAMAIS été vues par le modèle pendant l'entraînement

            TradingMetricsV2 metrics = simulateTradingWalkForward(
                series,                    // Série complète (pour contexte historique)
                testStartBar,              // Début test (dans zone out-of-sample)
                testEndBar,                // Fin test (dans zone out-of-sample)
                preTrainedModel,           // Modèle entraîné sur [0, testStartFromBar] uniquement
                preTrainedScalers,         // Scalers calculés sur [0, testStartFromBar] uniquement
                config                     // Configuration
            );

            // Calcul MSE sur ce split out-of-sample
            metrics.mse = computeSplitMse(series, testStartBar, testEndBar,
                                        preTrainedModel, preTrainedScalers, config);

            // Calcul du business score
            double businessScore = computeBusinessScore(
                metrics.profitFactor, metrics.winRate,
                metrics.maxDrawdownPct, metrics.expectancy, config
            );
            metrics.businessScore = businessScore;

            // ===== AGRÉGATION DES RÉSULTATS =====

            result.splits.add(metrics);

            if (Double.isFinite(metrics.mse)) {
                sumMse += metrics.mse;
                mseList.add(metrics.mse);
                mseCount++;
            }

            if (Double.isFinite(businessScore)) {
                sumBusiness += businessScore;
                businessCount++;
            }

            logger.debug("[WALK-FORWARD-OOS] Split {}: MSE={}, PF={}, WinRate={}, BusinessScore={}",
                        s, metrics.mse, metrics.profitFactor, metrics.winRate, businessScore);
        }

        // ===== PHASE 4: CALCUL DES MOYENNES =====

        result.meanMse = mseCount > 0 ? sumMse / mseCount : Double.NaN;
        result.meanBusinessScore = businessCount > 0 ? sumBusiness / businessCount : Double.NaN;
        result.totalTestedBars = totalTestedBars;

        // Calcul de la variance MSE
        if (mseList.size() > 1) {
            double variance = mseList.stream()
                .mapToDouble(mse -> Math.pow(mse - result.meanMse, 2))
                .average().orElse(0.0);
            result.mseVariance = variance;
            result.mseInterModelVariance = variance;
        }

        logger.info("[WALK-FORWARD-OOS] Fin évaluation out-of-sample: meanMSE={}, meanBusinessScore={}, {} splits valides sur données non vues",
                   result.meanMse, result.meanBusinessScore, result.splits.size());

        return result;
    }

    /**
     * Simule une stratégie swing trade optimisée (LONG ONLY - pas de short selling)
     * sur un intervalle de test en utilisant les prédictions successives du modèle.
     * Adaptée pour Alpaca avec frais réalistes et logique swing trade professionnelle.
     *
     * @return TradingMetricsV2 métriques remplies.
     */
    public TradingMetricsV2 simulateTradingWalkForward(
        BarSeries fullSeries,
        int testStartBar,
        int testEndBar,
        MultiLayerNetwork model,
        ScalerSet scalers,
        LstmConfig config
    ) {
        TradingMetricsV2 tm = new TradingMetricsV2();
        // ===== Étape 6: Pré-calcul & cache features / indicateurs =====
        long cacheStartNs = System.nanoTime();
        List<String> features = config.getFeatures();
        double[][] rawMatrix = extractFeatureMatrix(fullSeries, features); // [bars][features]
        int numFeatures = features.size();
        double[][] normMatrix = new double[rawMatrix.length][numFeatures];
        if (scalers == null || scalers.featureScalers == null || scalers.featureScalers.size() != numFeatures || scalers.labelScaler == null) {
            logger.warn("[SIM][CACHE] Scalers incomplets -> rebuild (coût ponctuel)");
            scalers = rebuildScalers(fullSeries, config);
        }
        if (config.isUseLogReturnTarget() && scalers.labelScaler.type == FeatureScaler.Type.MINMAX) {
            try {
                scalers.labelScaler = rebuildLabelScalerForLogReturn(fullSeries, config);
            } catch (Exception e) { logger.error("[SIM][CACHE][LABEL_MIGRATION] {}", e.getMessage()); }
        }
        // Normalisation colonne par colonne (évite recalculs multiples)
        for (int f = 0; f < numFeatures; f++) {
            double[] col = new double[rawMatrix.length];
            for (int i = 0; i < rawMatrix.length; i++) col[i] = rawMatrix[i][f];
            double[] normCol = scalers.featureScalers.get(features.get(f)).transform(col);
            for (int i = 0; i < rawMatrix.length; i++) normMatrix[i][f] = normCol[i];
        }
        double[] closes = extractCloseValues(fullSeries);
        // Cache sous forme Map (acceptation plan) barIndex -> features normalisées
        Map<Integer, double[]> featureCache = new HashMap<>(normMatrix.length);
        for (int i = 0; i < normMatrix.length; i++) featureCache.put(i, normMatrix[i]);
        // Indicateurs lourds pré-calculés (ATR, RSI)
        ATRIndicator atrFull = new ATRIndicator(fullSeries, 14);
        RSIIndicator rsiFull = new RSIIndicator(new ClosePriceIndicator(fullSeries), 14);
        double[] atrValues = new double[fullSeries.getBarCount()];
        double[] rsiValues = new double[fullSeries.getBarCount()];
        for (int i = 0; i < fullSeries.getBarCount(); i++) {
            atrValues[i] = atrFull.getValue(i).doubleValue();
            rsiValues[i] = rsiFull.getValue(i).doubleValue();
        }
        long cacheReadyNs = System.nanoTime();
        logger.debug("[SIM][CACHE] Pré-calcul features+indicateurs en {} ms (bars={})", (cacheReadyNs - cacheStartNs)/1_000_000, fullSeries.getBarCount());
        // ===============================================================

        // Étape 18: calcul medianAtrPct (sans fuite: uniquement historique avant testStartBar)
        double medianAtrPct;
        {
            List<Double> atrPctHist = new ArrayList<>();
            for (int i = 0; i < Math.min(testStartBar, fullSeries.getBarCount()); i++) {
                double c = closes[i];
                if (c > 0) {
                    double pct = atrValues[i] / c;
                    if (Double.isFinite(pct) && pct > 0) atrPctHist.add(pct);
                }
            }
            if (atrPctHist.isEmpty()) {
                medianAtrPct = 0.01; // fallback 1%
            } else {
                Collections.sort(atrPctHist);
                int mIdx = atrPctHist.size() / 2;
                if (atrPctHist.size() % 2 == 1) medianAtrPct = atrPctHist.get(mIdx); else medianAtrPct = (atrPctHist.get(mIdx - 1) + atrPctHist.get(mIdx)) / 2.0;
                if (!Double.isFinite(medianAtrPct) || medianAtrPct <= 0) medianAtrPct = 0.01;
            }
        }
        logger.info("[POS][ADAPT] medianAtrPct (historique avant test) = {}%", String.format(Locale.US, "%.4f", medianAtrPct * 100));

        // ===== Étape 17: Cache prédictions intra-split =====
        Map<Integer, Double> predictionCache = new HashMap<>();
        int predictionComputeCount = 0;
        int predictionCacheHit = 0;
        // ===============================================================

        double equity = 0, peak = 0, trough = 0;
        boolean inPos = false; // Pas de longPos/shortPos - LONG ONLY
        double entry = 0;
        int barsInPos = 0;
        int horizon = Math.max(5, config.getHorizonBars());

        List<Double> tradePnL = new ArrayList<>();
        List<Double> tradeReturns = new ArrayList<>();
        List<Integer> barsInPosList = new ArrayList<>();
        // Étape 18: collecte des valeurs de position (valeur notionnelle) pour stats dispersion
        List<Double> positionValues = new ArrayList<>();

        double capital = config.getCapital();
        double riskPct = config.getRiskPct();
        double sizingK = config.getSizingK();
        double feePct = 0.0; // commission-free
        double slippagePct = config.getSlippagePct();
        double positionSize = 0;
        double consecutiveLosses = 0;
        double maxConsecutiveLosses = 3;

        // Buffer circulaire pour distribution des |predictedDelta| (Step 14)
        final int deltaBufCapacity = 200;
        double[] deltaBuf = new double[deltaBufCapacity];
        int deltaBufCount = 0;
        int deltaBufIndex = 0;
        final double deltaFloor = 0.0007; // floor minimal
        final int minSamplesForPercentile = 30; // warmup minimal

        // Boucle principale
        for (int bar = testStartBar; bar < testEndBar - 1; bar++) {
            if (bar < config.getWindowSize()) continue; // pas assez d'historique
            BarSeries sub = fullSeries.getSubSeries(0, bar + 1); // conservé pour fonctions existantes (threshold, volume)
            double threshold = computeSwingTradeThreshold(sub, config); // Toujours utilisé pour target sizing
            double atr = atrValues[bar];
            if (!inPos) {
                if (consecutiveLosses >= maxConsecutiveLosses) {
                    if (bar % 5 == 0) consecutiveLosses = Math.max(0, consecutiveLosses - 1);
                    continue;
                }
                double predicted;
                Double cachedPred = predictionCache.get(bar);
                if (cachedPred != null) {
                    predicted = cachedPred;
                    predictionCacheHit++;
                } else {
                    try {
                        predicted = predictNextCloseScalarCached(bar, normMatrix, closes, config, model, scalers);
                    } catch (Exception e) {
                        predicted = predictNextCloseScalarV2(sub, config, model, scalers); // fallback
                    }
                    predictionCache.put(bar, predicted);
                    predictionComputeCount++;
                }
                double lastClose = closes[bar];
                double rawDelta = (predicted - lastClose) / lastClose; // predictedDelta relatif
                double signalStrength = rawDelta; // alias logique existante

                // Calcul du threshold adaptatif via percentile 65 des |predictedDelta| (sans inclure le courant)
                double entryThreshold;
                if (deltaBufCount >= minSamplesForPercentile) {
                    // Copier tampon courant dans tableau triable
                    int n = deltaBufCount;
                    double[] tmp = new double[n];
                    for (int i = 0; i < n; i++) tmp[i] = deltaBuf[i];
                    Arrays.sort(tmp);
                    int idx = (int)Math.floor(0.65 * (n - 1));
                    double perc65 = tmp[idx];
                    entryThreshold = Math.max(perc65, deltaFloor);
                } else {
                    entryThreshold = deltaFloor; // warmup
                }

                ContrarianDecision contrarianDecision = evaluateContrarian(signalStrength, lastClose, rsiValues, bar, sub);
                if (contrarianDecision.active) {
                    signalStrength = contrarianDecision.adjustedSignalStrength;
                    logger.info("[DEBUG][CONTRARIAN] bar={} rsi={} signalAdj={}", bar,
                            String.format("%.1f", rsiValues[bar]),
                            String.format("%.4f", signalStrength));
                }

                // Logs debug (pour comparaison historique) : on conserve threshold (baseline) + entryThreshold distributionnel
                double signalStrengthPct = signalStrength * 100;
                double entryThresholdPct = entryThreshold * 100;
                double thresholdPct = threshold * 100;
                logger.info("[DEBUG][RAW][CACHE][DIST] bar={}, predicted={}, lastClose={}, signalStrength={}% ({}), entryThreshold(P65)={}% ({}), baselineThreshold={}% ({}), bufSize={}",
                        bar,
                        String.format("%.3f", predicted),
                        String.format("%.3f", lastClose),
                        String.format("%.4f", signalStrengthPct), String.format("%.6f", signalStrength),
                        String.format("%.4f", entryThresholdPct), String.format("%.6f", entryThreshold),
                        String.format("%.4f", thresholdPct), String.format("%.6f", threshold),
                        deltaBufCount);

                if (entryThreshold > 0.02) // garde-fou extrême improbable
                    logger.warn("[DEBUG][EMERGENCY][CACHE] entryThreshold distrib trop élevé ({}%), clamp 0.3%", String.format("%.4f", entryThreshold * 100));
                // Plus de réduction temporelle: adaptiveThreshold == entryThreshold
                double adaptiveThreshold = entryThreshold;
                boolean wouldEnter = signalStrength > adaptiveThreshold;
                logger.info("[DEBUG][ENTRY][CACHE][DIST] bar={} wouldEnter={} signal={} thresh={} adaptive={} bufSize={}", bar, wouldEnter,
                        String.format("%.6f", signalStrength), String.format("%.6f", entryThreshold), String.format("%.6f", adaptiveThreshold), deltaBufCount);
                if (signalStrength > adaptiveThreshold) {
                    double currentRsi = rsiValues[bar];
                    logger.info("[DEBUG][ENTRY][CACHE] bar={}, currentRsi={} rawDeltaAbs={} entryThreshold={}", bar, currentRsi,
                            String.format("%.6f", Math.abs(rawDelta)), String.format("%.6f", entryThreshold));
                    if (currentRsi > 75) {
                        // Update buffer avant continue pour ne pas perdre ce delta historique
                        double absDelta = Math.abs(rawDelta);
                        if (deltaBufCount < deltaBufCapacity) {
                            deltaBuf[deltaBufCount++] = absDelta;
                        } else {
                            deltaBuf[deltaBufIndex] = absDelta;
                            deltaBufIndex = (deltaBufIndex + 1) % deltaBufCapacity;
                        }
                        continue;
                    }
                    // Volume check (non pré-calculé – coût acceptable)
                    VolumeIndicator vol = new VolumeIndicator(sub);
                    double currentVol = vol.getValue(sub.getEndIndex()).doubleValue();
                    double avgVol = 0;
                    int volPeriod = Math.min(20, sub.getBarCount() - 1);
                    for (int i = sub.getEndIndex() - volPeriod + 1; i <= sub.getEndIndex(); i++) {
                        avgVol += vol.getValue(i).doubleValue();
                    }
                    avgVol /= volPeriod;
                    if (currentVol < avgVol * 0.8) {
                        double absDelta = Math.abs(rawDelta);
                        if (deltaBufCount < deltaBufCapacity) {
                            deltaBuf[deltaBufCount++] = absDelta;
                        } else {
                            deltaBuf[deltaBufIndex] = absDelta;
                            deltaBufIndex = (deltaBufIndex + 1) % deltaBufCapacity;
                        }
                        continue;
                    }
                    inPos = true;
                    entry = lastClose;
                    barsInPos = 0;
                    double atrPct = atr / lastClose;
                    double riskAmount = capital * riskPct;
                    double stopDistance = atrPct * sizingK;
                    positionSize = stopDistance > 0 ? riskAmount / (lastClose * stopDistance) : 0;
                    // Étape 18: plafonnement adaptatif
                    double hardCap = capital * 0.10; // cap fixe historique 10% du capital
                    double dynFactor = (medianAtrPct > 0) ? (0.5 * atrPct / medianAtrPct) : 0.5; // 0.5 * ratio ATR relatifs
                    // Limites de robustesse sur dynFactor
                    if (!Double.isFinite(dynFactor) || dynFactor <= 0) dynFactor = 0.5;
                    dynFactor = Math.max(0.05, Math.min(dynFactor, 1.5)); // clamp pour éviter extrêmes
                    double dynamicCap = capital * dynFactor;
                    double positionValueMax = Math.min(hardCap, dynamicCap);
                    if (positionValueMax <= 0) positionValueMax = hardCap; // fallback
                    double rawPositionValue = positionSize * lastClose;
                    if (rawPositionValue > positionValueMax && lastClose > 0) {
                        positionSize = positionValueMax / lastClose;
                    }
                    double finalPositionValue = positionSize * lastClose;
                    positionValues.add(finalPositionValue);
                    logger.debug("[POS][ADAPT] bar={} atrPct={} medianAtrPct={} dynFactor={} dynCap={} hardCap={} finalCap={} rawPosValue={} finalPosValue={}",
                            bar,
                            String.format(Locale.US, "%.5f", atrPct),
                            String.format(Locale.US, "%.5f", medianAtrPct),
                            String.format(Locale.US, "%.3f", dynFactor),
                            String.format(Locale.US, "%.2f", dynamicCap),
                            String.format(Locale.US, "%.2f", hardCap),
                            String.format(Locale.US, "%.2f", positionValueMax),
                            String.format(Locale.US, "%.2f", rawPositionValue),
                            String.format(Locale.US, "%.2f", finalPositionValue));
                }
                // Mise à jour buffer après décision (n'inclut pas le delta dans son propre calcul de percentile)
                double absDelta = Math.abs(rawDelta);
                if (deltaBufCount < deltaBufCapacity) {
                    deltaBuf[deltaBufCount++] = absDelta;
                } else {
                    deltaBuf[deltaBufIndex] = absDelta;
                    deltaBufIndex = (deltaBufIndex + 1) % deltaBufCapacity;
                }
            } else {
                barsInPos++;
                double current = closes[bar];
                boolean exit = false;
                double pnl = 0;
                double atrStop = entry - (atr * sizingK * 1.5);
                double basicTarget = entry * (1 + threshold * 3);
                double atrTarget = entry + (atr * sizingK * 2.5);
                double target = Math.max(basicTarget, atrTarget);
                double trailingStop = Math.max(atrStop, current * 0.95);
                if (current <= trailingStop || current >= target || barsInPos >= horizon) {
                    pnl = (current - entry) * positionSize;
                    exit = true;
                } else if (barsInPos >= 3) {
                    Double cachedPred2 = predictionCache.get(bar);
                    double newPredicted;
                    if (cachedPred2 != null) {
                        newPredicted = cachedPred2;
                        predictionCacheHit++;
                    } else {
                        try {
                            newPredicted = predictNextCloseScalarCached(bar, normMatrix, closes, config, model, scalers);
                        } catch (Exception ignored) {
                            newPredicted = current; // fallback neutre
                        }
                        predictionCache.put(bar, newPredicted);
                        predictionComputeCount++;
                    }
                    if (newPredicted < current * 0.98) {
                        pnl = (current - entry) * positionSize;
                        exit = true;
                    }
                }
                if (exit) {
                    double entrySlippage = slippagePct * entry * positionSize;
                    double exitSlippage = slippagePct * current * positionSize;
                    double totalCosts = entrySlippage + exitSlippage;
                    pnl -= totalCosts;
                    tradePnL.add(pnl);
                    tradeReturns.add(pnl / (entry * positionSize));
                    barsInPosList.add(barsInPos);
                    if (pnl < 0) consecutiveLosses++; else consecutiveLosses = 0;
                    equity += pnl;
                    if (equity > peak) { peak = equity; trough = equity; }
                    else if (equity < trough) { trough = equity; }
                    inPos = false; entry = 0; barsInPos = 0; positionSize = 0;
                }
            }
        }

        logger.debug("[SIM][CACHE][PRED] computes={} hits={} hitRatio={} totalBarsSimul={}",
                predictionComputeCount,
                predictionCacheHit,
                (predictionComputeCount + predictionCacheHit) > 0 ? String.format("%.2f", (predictionCacheHit * 100.0)/(predictionComputeCount + predictionCacheHit)) : "0.00",
                (testEndBar - testStartBar));

        // Agrégation métriques
        double gains = 0, losses = 0;
        int win = 0, loss = 0;
        for (double p : tradePnL) {
            if (p > 0) { gains += p; win++; }
            else if (p < 0) { losses += p; loss++; }
        }
        logger.info("[--------][DEBUG][TRADES] Nb trades={}, wins={}, losses={}, gains={}, lossesSum={}", tradePnL.size(), win, loss, gains, losses);

        tm.totalProfit = gains + losses;
        tm.numTrades = tradePnL.size();
        tm.profitFactor = losses != 0 ? gains / Math.abs(losses) : (gains > 0 ? Double.POSITIVE_INFINITY : 0);
        tm.winRate = tm.numTrades > 0 ? (double) win / tm.numTrades : 0.0;

        double ddAbs = peak - trough;
        tm.maxDrawdownPct = peak != 0 ? ddAbs / Math.abs(peak) : 0.0;

        double avgGain = win > 0 ? gains / win : 0;
        double avgLoss = loss > 0 ? Math.abs(losses) / loss : 0;
        tm.expectancy = (win + loss) > 0 ? (tm.winRate * avgGain - (1 - tm.winRate) * avgLoss) : 0;

        // Calcul Sharpe et Sortino optimisés
        double meanRet = tradeReturns.stream().mapToDouble(d -> d).average().orElse(0);
        double stdRet = Math.sqrt(tradeReturns.stream().mapToDouble(d -> {
            double m = d - meanRet;
            return m * m;
        }).average().orElse(0));

        tm.sharpe = stdRet > 0 ? meanRet / stdRet * Math.sqrt(Math.max(1, tradeReturns.size())) : 0;

        // Sortino : uniquement downside deviation
        double downsideStd = Math.sqrt(tradeReturns.stream()
            .filter(r -> r < meanRet)
            .mapToDouble(r -> {
                double dr = r - meanRet;
                return dr * dr;
            }).average().orElse(0));
        tm.sortino = downsideStd > 0 ? meanRet / downsideStd : 0;

        // Exposure et turnover pour swing trade
        tm.exposure = barsInPosList.isEmpty() ? 0.0 :
            barsInPosList.stream().mapToInt(i -> i).sum() / (double)(testEndBar - testStartBar);
        tm.turnover = tm.numTrades > 0 ? (double)tm.numTrades / ((testEndBar - testStartBar) / 252.0) : 0; // Annualisé
        tm.avgBarsInPosition = barsInPosList.stream().mapToInt(i -> i).average().orElse(0);

        double capitalBase = config.getCapital() > 0 ? config.getCapital() : 1.0;
        double annualizedReturn = tm.totalProfit / capitalBase;
        tm.calmar = tm.maxDrawdownPct > 0 ? (annualizedReturn / tm.maxDrawdownPct) : 0.0;

        // Step15: calcul ratio contrarian
        int counted = tm.contrarianTrades + tm.normalTrades;
        if (tm.numTrades < counted) {
            int diff = counted - tm.numTrades;
            if (tm.normalTrades >= diff) tm.normalTrades -= diff; else if (tm.contrarianTrades >= diff) tm.contrarianTrades -= diff;
        } else if (tm.numTrades > counted) {
            tm.normalTrades += (tm.numTrades - counted);
        }
        tm.contrarianRatio = tm.numTrades > 0 ? (double) tm.contrarianTrades / tm.numTrades : 0.0;
        if (tm.contrarianRatio > 0.35) {
            logger.warn("[STEP15][CHECK] ratio contrarian élevé {}% (>35%)", String.format("%.2f", tm.contrarianRatio * 100));
        } else {
            logger.info("[STEP15][CHECK] ratio contrarian {}% (<=35%)", String.format("%.2f", tm.contrarianRatio * 100));
        }
        // Étape 18: calcul stats dispersion position sizing
        if (!positionValues.isEmpty()) {
            double sumPV = 0; for (double v : positionValues) sumPV += v; double meanPV = sumPV / positionValues.size();
            double varPV = 0; for (double v : positionValues) { double d = v - meanPV; varPV += d * d; } varPV = varPV / positionValues.size();
            double stdPV = Math.sqrt(varPV);
            tm.positionValueMean = meanPV;
            tm.positionValueStd = stdPV;
            logger.info("[POS][ADAPT][STATS] positions={} meanValue={} stdValue={} stdPctCapital={}",
                    positionValues.size(),
                    String.format(Locale.US, "%.2f", meanPV),
                    String.format(Locale.US, "%.2f", stdPV),
                    String.format(Locale.US, "%.2f", (stdPV / capital) * 100));
        }

        return tm;
    }

    /**
     * Calcule le MSE sur un intervalle (test) en réutilisant la logique de prédiction.
     *
     * @return MSE ou NaN si pas de points.
     */
    public double computeSplitMse(BarSeries series, int testStartBar, int testEndBar, MultiLayerNetwork model, ScalerSet scalers, LstmConfig config) {
        int window = config.getWindowSize();
        double se = 0;
        int count = 0;
        double[] closes = extractCloseValues(series);

        for (int t = testStartBar; t < testEndBar; t++) {
            if (t - window < 1) continue; // si log-return => besoin de t-1
            BarSeries sub = series.getSubSeries(0, t); // exclut t (label à t)
            double pred = predictNextCloseScalarV2(sub, config, model, scalers);
            if (config.isUseLogReturnTarget()) {
                // Comparer dans l'espace log-return
                double predLogReturn = Math.log(pred / closes[t - 1]);
                double actual = Math.log(closes[t] / closes[t - 1]);
                double err = predLogReturn - actual;
                se += err * err;
            } else {
                // Comparer dans l'espace prix
                double actual = closes[t];
                double err = pred - actual;
                se += err * err;
            }
            count++;
        }
        return count > 0 ? se / count : Double.NaN;
    }

    // Étape 22: extraction des derniers labels réalisés (log-return simple ou moyenne multi-horizon)
    // Retourne un tableau de taille <= lookback (peut être plus petit si historique insuffisant)
    private double[] computeRecentLabelValues(BarSeries series, LstmConfig config, int lookback) {
        if (!config.isUseLogReturnTarget()) {
            return new double[0];
        }
        int n = series.getBarCount();
        if (n < 3) return new double[0];
        double[] closes = extractCloseValues(series);
        int windowSize = config.getWindowSize();
        int horizon = Math.max(1, config.getHorizonBars());

        // Dernier index de départ i pour lequel la cible (fenêtre + horizon) existe
        // En training labelSeq[i] utilise closes[i + windowSize -1] comme prev puis jusqu'à horizon après
        int lastStart;
        if (config.isUseMultiHorizonAvg()) {
            lastStart = closes.length - windowSize - horizon; // besoin fenêtre + H pas futurs
        } else {
            lastStart = closes.length - windowSize - 1; // besoin fenêtre + 1 pas futur
        }
        if (lastStart < 0) return new double[0];

        int firstStart = Math.max(0, lastStart - lookback + 1);
        java.util.List<Double> labels = new java.util.ArrayList<>();

        for (int i = firstStart; i <= lastStart; i++) {
            if (config.isUseMultiHorizonAvg()) {
                double prev = closes[i + windowSize - 1];
                double sum = 0.0; int c = 0; double curPrev = prev;
                for (int h = 1; h <= horizon; h++) {
                    int idx = i + windowSize - 1 + h;
                    if (idx >= closes.length) break;
                    double next = closes[idx];
                    if (curPrev > 0 && next > 0) {
                        double lr = Math.log(next / curPrev);
                        if (Double.isFinite(lr)) { sum += lr; c++; }
                    }
                    curPrev = next;
                }
                labels.add(c > 0 ? (sum / c) : 0.0);
            } else {
                int prevIdx = i + windowSize - 1;
                int nextIdx = i + windowSize;
                if (nextIdx < closes.length) {
                    double a = closes[prevIdx];
                    double b = closes[nextIdx];
                    double lr = (a > 0 && b > 0) ? Math.log(b / a) : 0.0;
                    labels.add(Double.isFinite(lr) ? lr : 0.0);
                }
            }
        }
        double[] out = new double[labels.size()];
        for (int k = 0; k < labels.size(); k++) out[k] = labels.get(k);
        return out;
    }

    /* =========================================================
     *                DRIFT DETECTION (STATISTIQUE)
     * =========================================================
     * Logique simplifiée: on découpe l'historique d'une feature en 2 segments
     * (past / recent) et on compare:
     *  - Shift de moyenne (exprimé en écart-types passés)
     *  - KL divergence symétrique approximée (histogrammes)
     */

    public static class DriftDetectionResult {
        public boolean drift;
        public String driftType;
        public double kl;
        public double meanShift;
    }

    public static class DriftReportEntry {
        public java.time.Instant eventDate;
        public String symbol;
        public String feature;
        public String driftType;
        public double kl;
        public double meanShift;
        public double mseBefore;
        public double mseAfter;
        public boolean retrained;
    }

    /**
     * Shortcut booléen du test de drift (utilise version détaillée).
     */
    public boolean checkDriftForFeature(String feat, double[] values, FeatureScaler scaler, double klThreshold, double meanShiftSigma) {
        return checkDriftForFeatureDetailed(feat, values, scaler, klThreshold, meanShiftSigma).drift;
    }

    /**
     * Test détaillé de drift (moyenne + KL).
     *
     * @param feat            Nom de la feature
     * @param values          Valeurs brutes
     * @param scaler          Scaler associé (non modifié ici)
     * @param klThreshold     Seuil KL
     * @param meanShiftSigma  Seuil de shift de moyenne (en sigma)
     * @return Résultat (drift ou non + métriques)
     */
    public DriftDetectionResult checkDriftForFeatureDetailed(String feat, double[] values, FeatureScaler scaler, double klThreshold, double meanShiftSigma) {
        DriftDetectionResult r = new DriftDetectionResult();
        int n = values.length;
        if (n < 40) return r; // pas assez de données pour un test fiable

        int half = n / 2;
        double[] past = Arrays.copyOfRange(values, 0, half);
        double[] recent = Arrays.copyOfRange(values, half, n);

        double meanPast = Arrays.stream(past).average().orElse(0);
        double meanRecent = Arrays.stream(recent).average().orElse(0);
        double varPast = Arrays.stream(past).map(v -> (v - meanPast) * (v - meanPast)).sum() / past.length;
        double varRecent = Arrays.stream(recent).map(v -> (v - meanRecent) * (v - meanRecent)).sum() / recent.length;
        double stdPast = Math.sqrt(varPast + 1e-9);

        r.meanShift = (meanRecent - meanPast) / stdPast;
        r.kl = approximateSymmetricKl(past, recent, 20);

        if (Math.abs(r.meanShift) > meanShiftSigma) {
            r.drift = true;
            r.driftType = "mean_shift";
        }
        if (r.kl > klThreshold) {
            r.drift = true;
            r.driftType = (r.driftType == null ? "kl" : r.driftType + "+kl");
        }
        return r;
    }

    /**
     * Approximation KL symétrique via histogrammes discrets.
     */
    private double approximateSymmetricKl(double[] a, double[] b, int bins) {
        double min = Math.min(Arrays.stream(a).min().orElse(0), Arrays.stream(b).min().orElse(0));
        double max = Math.max(Arrays.stream(a).max().orElse(1), Arrays.stream(b).max().orElse(1));
        if (max - min == 0) return 0;

        double[] ha = new double[bins];
        double[] hb = new double[bins];
        double w = (max - min) / bins;

        for (double v : a) {
            int idx = (int) Math.floor((v - min) / w);
            if (idx < 0) idx = 0;
            else if (idx >= bins) idx = bins - 1;
            ha[idx]++;
        }
        for (double v : b) {
            int idx = (int) Math.floor((v - min) / w);
            if (idx < 0) idx = 0;
            else if (idx >= bins) idx = bins - 1;
            hb[idx]++;
        }

        double sumA = Arrays.stream(ha).sum();
        double sumB = Arrays.stream(hb).sum();
        double kl1 = 0, kl2 = 0;

        for (int i = 0; i < bins; i++) {
            double pa = (ha[i] + 1e-9) / sumA;
            double pb = (hb[i] + 1e-9) / sumB;
            kl1 += pa * Math.log(pa / pb);
            kl2 += pb * Math.log(pb / pa);
        }
        return 0.5 * (kl1 + kl2);
    }

    /**
     * Vérifie le drift sur chaque feature, et déclenche un réentraînement si drift détecté.
     * Retourne un rapport détaillé.
     */
    public List<DriftReportEntry> checkDriftAndRetrainWithReport(BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers, String symbol) {
        List<DriftReportEntry> reports = new ArrayList<>();
        if (model == null || scalers == null) return reports;

        double mseBefore = Double.NaN;
        double mseAfter = Double.NaN;
        try {
            int total = series.getBarCount();
            int testStart = Math.max(0, total - (config.getWindowSize() * 3));
            mseBefore = computeSplitMse(series, testStart, total, model, scalers, config);
        } catch (Exception ignored) {}

        boolean retrain = false;
        // Étape 22: Drift label (log-return) ciblé avant boucle features pour ne pas dépendre d'elles
        if (config.isUseLogReturnTarget()) {
            try {
                int LOOKBACK = 250;
                double[] recentLabels = computeRecentLabelValues(series, config, LOOKBACK);
                if (recentLabels.length > 30) { // taille minimale
                    double m=0; for(double v: recentLabels) m+=v; m=recentLabels.length>0? m/recentLabels.length:0;
                    double var=0; for(double v: recentLabels){ double d=v-m; var+=d*d; } var = recentLabels.length>0? var/recentLabels.length:0; double std = Math.sqrt(var);
                    scalers.labelDistMean = m;
                    scalers.labelDistStd = std>1e-12? std : 1e-12; // éviter division par 0
                    logger.info("[STEP22][LABEL_DIST][TRAIN] mean={} std={}", String.format(java.util.Locale.US, "%.6f", scalers.labelDistMean), String.format(java.util.Locale.US, "%.6f", scalers.labelDistStd));
                } else {
                    logger.debug("[STEP22][LABEL_DIST] Série insuffisante pour LOOKBACK={} (size={})", 250, recentLabels.length);
                }
            } catch (Exception ex) {
                logger.warn("[STEP22][LABEL_DRIFT] Erreur calcul drift label: {}", ex.getMessage());
            }
        }

        for (String feat : config.getFeatures()) {
            FeatureScaler sc = scalers.featureScalers.get(feat);
            if (sc == null) continue;

            double[] vals = new double[series.getBarCount()];
            // (Extraction isolée pour la feature)
            for (int i = 0; i < series.getBarCount(); i++)
                vals[i] = extractFeatureMatrix(series, Collections.singletonList(feat))[i][0];

            DriftDetectionResult res = checkDriftForFeatureDetailed(
                feat,
                vals,
                sc,
                config.getKlDriftThreshold(),
                config.getMeanShiftSigmaThreshold()
            );
            if (res.drift) retrain = true;

            DriftReportEntry entry = new DriftReportEntry();
            entry.eventDate = java.time.Instant.now();
            entry.symbol = symbol;
            entry.feature = feat;
            entry.driftType = res.driftType;
            entry.kl = res.kl;
            entry.meanShift = res.meanShift;
            entry.mseBefore = mseBefore;
            entry.retrained = false;
            reports.add(entry);
        }

        // Si drift détecté => réentraînement complet
        if (retrain) {
            TrainResult tr = trainLstmScalarV2(series, config, null);
            if (tr.model != null) {
                // Copie des paramètres dans l'ancien modèle
                model.setParams(tr.model.params());
                scalers.featureScalers = tr.scalers.featureScalers;
                scalers.labelScaler = tr.scalers.labelScaler;
                try {
                    int total = series.getBarCount();
                    int testStart = Math.max(0, total - (config.getWindowSize() * 3));
                    mseAfter = computeSplitMse(series, testStart, total, model, scalers, config);
                } catch (Exception ignored) {}
                for (DriftReportEntry r : reports) {
                    r.retrained = true;
                    r.mseAfter = mseAfter;
                }
            }
        }
        return reports;
    }


    /* =========================================================
     *              THRESHOLD & SPREAD UTILITAIRES
     * ========================================================= */

    /**
     * Calcule un seuil de swing trading relatif basé soit sur ATR, soit sur variance des returns.
     * Version optimisée pour des seuils plus réalistes en trading moderne.
     *
     * @return seuil (en pourcentage relatif, ex: 0.01 = 1%)
     */
    public double computeSwingTradeThreshold(BarSeries series, LstmConfig config) {
        double k = config.getThresholdK();
        String type = config.getThresholdType();

        double rawThreshold = 0.0;

        if ("ATR".equalsIgnoreCase(type)) {
            ATRIndicator atr = new ATRIndicator(series, 14);
            double lastATR = atr.getValue(series.getEndIndex()).doubleValue();
            double lastClose = series.getLastBar().getClosePrice().doubleValue();
            rawThreshold = k * lastATR / (lastClose == 0 ? 1 : lastClose);

            // Log pour diagnostic
            logger.debug("[SEUIL SWING][ATR] rawThreshold={}% (ATR={}, close={}, k={})",
                String.format("%.4f", rawThreshold * 100), lastATR, lastClose, k);

        } else if ("returns".equalsIgnoreCase(type)) {
            double[] closes = extractCloseValues(series);
            if (closes.length < 3) {
                rawThreshold = 0.005; // 0.5% par défaut
            } else {
                double[] logRet = new double[closes.length - 1];
                for (int i = 1; i < closes.length; i++)
                    logRet[i - 1] = Math.log(closes[i] / closes[i - 1]);
                double mean = Arrays.stream(logRet).average().orElse(0);
                double std = Math.sqrt(Arrays.stream(logRet).map(r -> (r - mean) * (r - mean)).sum() / logRet.length);
                rawThreshold = k * std;

                // Log pour diagnostic
                logger.debug("[SEUIL SWING][RETURNS] rawThreshold={}% (std={}, k={})",
                    String.format("%.4f", rawThreshold * 100), std, k);
            }
        } else {
            // Fallback intelligent basé sur la volatilité récente
            rawThreshold = 0.01 * k; // 1% * k par défaut
        }

        // CORRECTION CRITIQUE : Limiter les seuils aberrants
        // Un seuil de plus de 1% est généralement trop élevé pour le trading moderne
        double maxAllowedThreshold = 0.01; // 1% maximum
        double minAllowedThreshold = 0.001; // 0.1% minimum

        // Application des bornes
        double adjustedThreshold = Math.max(minAllowedThreshold, Math.min(rawThreshold, maxAllowedThreshold));

        // Log des ajustements si nécessaire
        if (Math.abs(adjustedThreshold - rawThreshold) > 0.0001) {
            logger.info("[SEUIL SWING][ADJUST] Seuil ajusté: {}% -> {}% (bornes: {}%-{}%)",
                String.format("%.4f", rawThreshold * 100),
                String.format("%.4f", adjustedThreshold * 100),
                String.format("%.1f", minAllowedThreshold * 100),
                String.format("%.1f", maxAllowedThreshold * 100));
        } else {
            logger.debug("[SEUIL SWING][OK] Seuil final: {}%", String.format("%.4f", adjustedThreshold * 100));
        }

        return adjustedThreshold;
    }

    /**
     * Moyenne des spreads high-low sur la série (approximation coûts).
     */
    public double computeMeanSpread(BarSeries series) {
        int n = series.getBarCount();
        if (n == 0) return 0;
        double sum = 0;
        for (int i = 0; i < n; i++)
            sum += (series.getBar(i).getHighPrice().doubleValue() - series.getBar(i).getLowPrice().doubleValue());
        return sum / n;
    }

    /* =========================================================
     *                        PREDIT
     * =========================================================
     * Méthode "haut niveau" utilisée pour produire un objet métier PreditLsdm
     * contenant: dernier cours, prédiction, signal (UP/DOWN/STABLE), position relative.
     */

    /**
     * Produit l'objet de prédiction (inclut signal directionnel).
     *
     * @param symbol  Symbole (utilisé pour logs)
     * @param series  Série complète
     * @param config  Config LSTM
     * @param model   Modèle potentiellement déjà en mémoire
     * @param scalers Scalers associés
     * @return Objet PreditLsdm (DTO)
     */
    public PreditLsdm getPredit(String symbol, BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers) {
        int window = config.getWindowSize();
        if (series.getBarCount() <= window + 1) {
            double last = series.getLastBar().getClosePrice().doubleValue();
            return PreditLsdm.builder()
                .lastClose(last)
                .predictedClose(last)
                .signal(SignalType.STABLE)
                .lastDate(series.getLastBar().getEndTime().format(java.time.format.DateTimeFormatter.ofPattern("dd/MM")))
                .position("insuffisant")
                .build();
        }

        model = ensureModelWindowSize(model, config.getFeatures().size(), config);
        if (scalers == null) {
            scalers = rebuildScalers(series, config);
        }

        double th = computeSwingTradeThreshold(series, config);
        double predicted = predictNextCloseWithScalerSet(symbol, series, config, model, scalers);
        predicted = Math.round(predicted * 1000.0) / 1000.0;

        double[] closes = extractCloseValues(series);
        double lastClose = closes[closes.length - 1];
        double delta = predicted - lastClose;

        SignalType signal =
            delta > th ? SignalType.UP :
                (delta < -th ? SignalType.DOWN : SignalType.STABLE);

        String position = analyzePredictionPosition(
            Arrays.copyOfRange(closes, closes.length - config.getWindowSize(), closes.length),
            predicted
        );

        String formattedDate = series.getLastBar()
            .getEndTime()
            .format(java.time.format.DateTimeFormatter.ofPattern("dd/MM"));

        logger.info("PREDICT win={} last={} pred={} delta={} thr={} signal={}",
            config.getWindowSize(), lastClose, predicted, delta, th, signal);

        return PreditLsdm.builder()
            .lastClose(lastClose)
            .predictedClose(predicted)
            .signal(signal)
            .lastDate(formattedDate)
            .position(position)
            .build();
    }

    /**
     * Analyse la position relative de la prédiction par rapport à la fenêtre historique.
     * Renvoie "au-dessus", "en-dessous" ou "dans la plage".
     */
    public String analyzePredictionPosition(double[] lastWindow, double predicted) {
        double min = Arrays.stream(lastWindow).min().orElse(Double.NaN);
        double max = Arrays.stream(lastWindow).max().orElse(Double.NaN);
        if (predicted > max) return "au-dessus";
        if (predicted < min) return "en-dessous";
        return "dans la plage";
    }

    /* =========================================================
     *                      PERSISTENCE
     * =========================================================
     * Sauvegarde sur base: modèle sérialisé + hyperparamètres + scalers.
     */

    /**
     * Sauvegarde le modèle + config + scalers en base (table lstm_models).
     * Utilise REPLACE INTO (écrase l'existant).
     *
     */
    public void saveModelToDb(String symbol, JdbcTemplate jdbcTemplate, MultiLayerNetwork model,  LstmConfig config, ScalerSet scalers,
                              double mse, double profitFactor, double winRate, double maxDrawdown, double rmse, double sumProfit, int totalTrades, double businessScore,
                              int totalSeriesTested) throws IOException {
        if (model == null) return;
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ModelSerializer.writeModel(model, baos, true);
        byte[] modelBytes = baos.toByteArray();

        // Sauvegarde hyperparamètres via repository + JSON
        // hyperparamsRepository.saveHyperparams(symbol, config);

        com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        String hyperparamsJson = mapper.writeValueAsString(config);
        String scalersJson = mapper.writeValueAsString(scalers);

        String sql = "REPLACE INTO lstm_models (symbol, model_blob, hyperparams_json, normalization_scope, scalers_json, mse, profit_factor, win_rate, max_drawdown, rmse, sum_profit, total_trades, business_score, updated_date, total_series_tested) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, CURRENT_TIMESTAMP,?)";
        jdbcTemplate.update(sql, symbol, modelBytes, hyperparamsJson, config.getNormalizationScope(), scalersJson, mse, profitFactor, winRate, maxDrawdown, rmse, sumProfit, totalTrades, businessScore, totalSeriesTested);
    }

    /**
     * Charge uniquement le modèle (ancienne version sans scalers).
     */
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
                logger.info("Modèle chargé {}", symbol);
                return model;
            }
        } catch (EmptyResultDataAccessException e) {
            throw new IOException("Modèle non trouvé");
        }
        return null;
    }

    /**
     * Charge modèle + scalers (JSON) + hyperparams.
     */
    public LoadedModel loadModelAndScalersFromDb(String symbol, JdbcTemplate jdbcTemplate) throws IOException {
        LstmConfig config = hyperparamsRepository.loadHyperparams(symbol);
        if (config == null) throw new IOException("Aucun hyperparamètre pour " + symbol);

        String sql = "SELECT model_blob, normalization_scope, scalers_json FROM lstm_models WHERE symbol = ?";
        MultiLayerNetwork model = null;
        ScalerSet scalers = null;

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
            logger.info("Chargé modèle+scalers pour {} (scalers={})", symbol, scalers != null);
        } catch (EmptyResultDataAccessException e) {
            throw new IOException("Modèle non trouvé");
        }
        return new LoadedModel(model, scalers);
    }

    /**
     * Wrapper pour retour groupé.
     */
    public static class LoadedModel {
        public MultiLayerNetwork model;
        public ScalerSet scalers;
        public LoadedModel(MultiLayerNetwork m, ScalerSet s){this.model=m;this.scalers=s;}
    }

    /* =========================================================
     *                         SEEDS
     * =========================================================
     * Fixer la seed permet (partiellement) de rendre reproductibles les résultats.
     */

    public void setGlobalSeeds(long seed){
        Nd4j.getRandom().setSeed(seed);
        // Force le chargement d'une classe (hack mineur historique, conservé)
        org.deeplearning4j.nn.api.OptimizationAlgorithm.valueOf("STOCHASTIC_GRADIENT_DESCENT");
        logger.debug("Seeds fixés seed={}", seed);
    }

    /**
     * Reconstruit des scalers "post-hoc" à partir d'une série.
     * ATTENTION: Peut diverger de ceux appris lors de l'entraînement initial
     * (car le dataset peut avoir évolué).
     */
    public ScalerSet rebuildScalers(BarSeries series, LstmConfig config){
        List<String> features = config.getFeatures();
        ScalerSet set = new ScalerSet();
        double[][] matrix = extractFeatureMatrix(series, features);
        int n = matrix.length;

        for (int f = 0; f < features.size(); f++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) col[i] = matrix[i][f];
            FeatureScaler.Type type =
                getFeatureNormalizationType(features.get(f)).equals("zscore")
                    ? FeatureScaler.Type.ZSCORE
                    : FeatureScaler.Type.MINMAX;
            FeatureScaler sc = new FeatureScaler(type);
            sc.fit(col);
            set.featureScalers.put(features.get(f), sc);
        }

        // Label scaler: ZSCORE si log-return, sinon MINMAX (prix direct)
        if (config.isUseLogReturnTarget()) {
            set.labelScaler = rebuildLabelScalerForLogReturn(series, config);
        } else {
            double[] closes = extractCloseValues(series);
            FeatureScaler lab = new FeatureScaler(FeatureScaler.Type.MINMAX);
            lab.fit(closes);
            set.labelScaler = lab;
        }

        logger.warn("[SCALERS][REBUILD] Reconstruction ad-hoc des scalers (peut diverger de l'entraînement initial).");
        return set;
    }

    // Reconstruit un label scaler ZSCORE pour log-return (avec gestion multi-horizon) pour migration.
    private FeatureScaler rebuildLabelScalerForLogReturn(BarSeries series, LstmConfig config) {
        double[] closes = extractCloseValues(series);
        if (closes.length < 2) {
            FeatureScaler fallback = new FeatureScaler(FeatureScaler.Type.ZSCORE);
            fallback.fit(new double[]{0,0});
            return fallback;
        }
        int windowSize = config.getWindowSize();
        int barCount = closes.length;
        int numSeq = barCount - windowSize - 1;
        if (numSeq < 1) numSeq = barCount - 1;
        double[] labelSeq = new double[Math.max(0, numSeq)];
        for (int i = 0; i < labelSeq.length; i++) {
            if (config.isUseMultiHorizonAvg()) {
                int H = config.getHorizonBars();
                double prev = closes[i + windowSize - 1];
                double sum=0; int count=0;
                for (int h=1; h<=H; h++) {
                    int idx = i + windowSize - 1 + h;
                    if (idx < closes.length) {
                        double next = closes[idx];
                        double logRet = Math.log(next / prev);
                        sum += logRet;
                        prev = next;
                        count++;
                    }
                }
                labelSeq[i] = count>0? (sum/count) : 0.0;
            } else {
                double prev = closes[i + windowSize - 1];
                double next = closes[i + windowSize];
                labelSeq[i] = Math.log(next / prev);
            }
        }
        FeatureScaler z = new FeatureScaler(FeatureScaler.Type.ZSCORE);
        z.fit(labelSeq);
        double[] norm = z.transform(labelSeq);
        double m=0,v=0; int n=norm.length; for(double d: norm)m+=d; m = n>0? m/n:0; for(double d: norm)v += (d-m)*(d-m); v = n>0? v/n:0; double std=Math.sqrt(v);
        logger.info("[MIGRATION][LABEL_SCALER] Nouveau label scaler ZSCORE construit. meanNorm={} stdNorm={}", String.format("%.4f", m), String.format("%.4f", std));
        return z;
    }

    // Étape 12: Méthode utilitaire pour appliquer un nouveau learning rate à toutes les couches
    private void applyLearningRate(MultiLayerNetwork model, double newLR) {
        if (model == null) return;
        try {
            model.setLearningRate(newLR);
        } catch (Exception e) {
            logger.warn("[TRAIN][LR-SCHED] Impossible d'appliquer nouveau LR={} : {}", newLR, e.getMessage());
        }
    }

    // Score métier combinant facteurs de performance (PF, winRate, drawdown, expectancy) -> [0,1]
    private double computeBusinessScore(double profitFactor, double winRate, double maxDrawdownPct, double expectancy, LstmConfig config) {
        double cap = config.getBusinessProfitFactorCap() > 0 ? config.getBusinessProfitFactorCap() : 3.0;
        double pfComp = Double.isFinite(profitFactor) && profitFactor > 0 ? Math.min(profitFactor, cap) / cap : 0.0;
        double wrComp = (winRate >= 0 && winRate <= 1) ? winRate : 0.0;
        double expComp;
        if (Double.isFinite(expectancy)) {
            // Normalisation douce expectancy ∈ R vers [0,1]
            double norm = expectancy / (1.0 + Math.abs(expectancy)); // (-1,1)
            expComp = (norm + 1.0) / 2.0; // (0,1)
        } else expComp = 0.5;
        double gamma = config.getBusinessDrawdownGamma() > 0 ? config.getBusinessDrawdownGamma() : 1.0;
        double ddPenalty = Math.tanh(gamma * Math.max(0, maxDrawdownPct)); // (0,1)
        double stabilityComp = 1.0 - ddPenalty; // drawdown faible => proche 1
        double score = 0.35 * pfComp + 0.20 * wrComp + 0.20 * expComp + 0.25 * stabilityComp;
        if (!Double.isFinite(score)) score = 0.0;
        return Math.max(0.0, Math.min(1.0, score));
    }
}
