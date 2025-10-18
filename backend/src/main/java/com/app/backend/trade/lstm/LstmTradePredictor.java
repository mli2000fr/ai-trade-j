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
import com.app.backend.trade.model.TradeStylePrediction;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
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
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

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

        // Détection backend GPU pour adaptation dynamique (réduction dropout / +neurones)
        boolean gpuBackend = false;
        try {
            String execCls = Nd4j.getExecutioner().getClass().getName().toLowerCase(Locale.ROOT);
            gpuBackend = execCls.contains("cuda") || execCls.contains("cudnn");
        } catch (Exception ignored) {}
        int effectiveLstmNeurons = lstmNeurons;
        double effectiveDropout = dropoutRate;
        if (gpuBackend) {
            // Réduction encore plus forte du coût régularisation (dropout) et augmentation capacité
            if (effectiveDropout > 0) {
                effectiveDropout = Math.max(0.0, Math.min(0.05, effectiveDropout * 0.25)); // /4, borne haute 0.05
            }
            effectiveLstmNeurons = Math.max(8, (int)Math.round(lstmNeurons * 1.35));
            if (effectiveLstmNeurons != lstmNeurons || effectiveDropout != dropoutRate) {
                logger.info("[LSTM][ADAPT][GPU] backend=GPU lstmNeurons {}->{} dropout {}->{}", lstmNeurons, effectiveLstmNeurons, dropoutRate, effectiveDropout);
            }
        } else {
            if (effectiveDropout > 0) {
                effectiveDropout = Math.max(0.0, Math.min(0.05, effectiveDropout)); // borne haute 0.05
            }
            logger.debug("[LSTM][ADAPT] backend=CPU (dropout plafonné à 0.05)");
        }

        // Remplace références locales par versions effectives
        lstmNeurons = effectiveLstmNeurons;
        dropoutRate = effectiveDropout;

        // Sélection dynamique de l'updater (optimiseur) selon chaîne.
        builder.updater(
            "adam".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.Adam(learningRate)
                : "rmsprop".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.RmsProp(learningRate)
                : new org.nd4j.linalg.learning.config.Sgd(learningRate)
        );

        // Régularisations - réduites pour permettre plus de variabilité
        builder.l1(l1 * 0.1).l2(l2 * 0.1); // Réduction de 90% pour moins de contraintes

        // Activation des workspaces mémoire (optimisation Dl4J)
        builder.trainingWorkspaceMode(WorkspaceMode.ENABLED)
               .inferenceWorkspaceMode(WorkspaceMode.ENABLED);
        if (config != null && config.isEnableGradientClipping()) {
            builder.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                   .gradientNormalizationThreshold(config.getGradientClippingThreshold());
            logger.info("[LSTM][OPT] Gradient clipping activé threshold={}", config.getGradientClippingThreshold());
        } else {
            logger.info("[LSTM][OPT] Gradient clipping désactivé (micro-optim GPU)");
        }

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
                .helperAllowFallback(true) // Étape 6: autoriser fallback CPU si cuDNN indisponible (et kernels rapides sinon)
                // dataType=FLOAT déjà défini globalement via builder.dataType(DataType.FLOAT)
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
                    // Dropout récurrent plafonné à 0.05
                    listBuilder.layer(new DropoutLayer.Builder()
                        .dropOut(Math.min(Math.max(dropoutRate, 0.0), 0.05))
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
            // Dropout dense final = min(0.05, dropoutRate)
            listBuilder.layer(new DropoutLayer.Builder()
                .dropOut(Math.min(0.05, Math.max(dropoutRate, 0.0)))
                .build());
        }

        // Sélection dynamique la fonction de perte / activation finale
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
                .l1(l1 * 0.05)
                .l2(l2 * 0.05)
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
        logger.info("[LSTM][INIT] defaultFPType={} modelParamsType={}", Nd4j.defaultFloatingPointType(), model.params().dataType());
        // Étape 9: log des valeurs de dropout réellement appliquées (récurrent plafonné 0.10, dense final = min(0.2, dropoutRate))
        double appliedRecurrentDropout = (dropoutRate > 0.0) ? Math.min(Math.max(dropoutRate, 0.0), 0.05) : 0.0;
        double appliedFinalDenseDropout = (dropoutRate > 0.0) ? Math.min(0.05, Math.max(dropoutRate, 0.0)) : 0.0;
        logger.info("[LSTM][Etape9] Dropout recurrent applique={} | Dropout dense final={} | neurons={}", appliedRecurrentDropout, appliedFinalDenseDropout, lstmNeurons);
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

        // --- CACHE GPU/CPU ---
        String symbol = series.getName() != null ? series.getName() : "UNKNOWN";
        String interval = "default"; // À adapter si interval stocké ailleurs
        long lastBarEndTime = n > 0 ? series.getBar(n-1).getEndTime().toEpochSecond() : 0L;
        String featureSetVersion = "v1"; // À incrémenter si features changent
        String cacheKey = LstmFeatureMatrixCache.computeKey(symbol, interval, n, lastBarEndTime, featureSetVersion, features);
        double[][] cached = LstmFeatureMatrixCache.load(cacheKey);
        if (cached != null) return cached;
        // --- FIN CACHE ---

        // Pré-calcul séries primitives pour indicateurs composites
        double[] closesRaw = new double[n];
        double[] highsRaw = new double[n];
        double[] lowsRaw = new double[n];
        double[] volumesRaw = new double[n];
        for (int i = 0; i < n; i++) {
            closesRaw[i] = series.getBar(i).getClosePrice().doubleValue();
            highsRaw[i] = series.getBar(i).getHighPrice().doubleValue();
            lowsRaw[i] = series.getBar(i).getLowPrice().doubleValue();
            volumesRaw[i] = series.getBar(i).getVolume().doubleValue();
        }

        // Pré-calcul spécifique realized_vol si demandé
        boolean needRealizedVol = features.contains("realized_vol");
        double[] realizedVol = null;
        if (needRealizedVol) {
            final int WIN = 14; // fenêtre log-returns
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

        // Indicateurs de base (utilisent BarSeries, mais pourraient être réécrits pour utiliser les tableaux si besoin)
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
        // À la toute fin, AVANT le return :
        LstmFeatureMatrixCache.save(cacheKey, M);
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
        @JsonIgnoreProperties(ignoreUnknown = true)
        public enum Type { MINMAX, ZSCORE }

        public double min = Double.POSITIVE_INFINITY;
        public double max = Double.NEGATIVE_INFINITY;
        public double mean = 0.0;
        public double std = 0.0;
        public Type type;

        // Constructeur no-arg requis pour Jackson
        public FeatureScaler(){
            this.type = Type.MINMAX; // défaut sûr
            this.min = 0; this.max = 1; // évite divisions 0
            this.mean = 0; this.std = 1.0;
        }

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
        public Double labelDistMean;
        public Double labelDistStd;
        // Constructeur no-arg pour robustesse
        public ScalerSet() {}
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
     *  3. Création labels (prix futur ou log-returns)
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
     * @return Résultat contenant modèle entraîné + scalers pour inversion/prédiction
     */
    public TrainResult trainLstmScalarV2(BarSeries series, LstmConfig config) {
        // Mode ultra agressif : désactivation complète de la deadzone
        config.setDisableDeadzone(true); // Deadzone désactivée pour agressivité maximale
        // Pour désactiver complètement, décommentez la ligne suivante :
        // config.setDeadzoneFactor(0.0);

        // === PROFILING GPU (Étape 7) : activation temporaire du debug ND4J pour tracer transferts CPU->GPU ===
        // NOTE: désactiver (supprimer ou mettre false) une fois l'analyse terminée pour éviter surcharge de logs.
        try {
            if (!"true".equalsIgnoreCase(System.getProperty("nd4j.exec.debug"))) {
                System.setProperty("nd4j.exec.debug", "true");
                logger.warn("[TRAIN][PROFILING] nd4j.exec.debug activé (temporaire) - analyser les logs pour transferts host/device");
            }
        } catch (Exception e) {
            logger.warn("[TRAIN][PROFILING] Impossible d'activer nd4j.exec.debug: {}", e.toString());
        }
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

        // Enrichissement automatique des features si trop pauvre
        if (features.size() == 1 && features.contains("close")) {
            logger.warn("[TRAIN][FEATURES] Liste trop pauvre, enrichissement automatique avec indicateurs dynamiques");
            List<String> enrichFeatures = new java.util.ArrayList<>(features);
            enrichFeatures.add("rsi");
            enrichFeatures.add("momentum");
            enrichFeatures.add("volatility");
            enrichFeatures.add("macd");
            enrichFeatures.add("sma20");
            enrichFeatures.add("ema20");
            config.setFeatures(enrichFeatures);
            features = enrichFeatures;
        }
        // Forcer l'utilisation du log-return comme label pour plus de dynamique
        if (!config.isUseLogReturnTarget()) {
            logger.warn("[TRAIN][LABEL] Forçage du mode log-return pour plus de dynamique");
            config.setUseLogReturnTarget(true);
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
        // On a besoin de windowSize barres pour l'input + horizonBars barres pour le label
        int horizonBars = config.getHorizonBars();
        int numSeq = barCount - windowSize - horizonBars;

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
                if (config.isUseMultiHorizonAvg()) {
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

        /* Ajout de bruit gaussien faible aux labels, évite trop plat
        java.util.Random rnd = new java.util.Random();
        for (int i = 0; i < labelSeq.length; i++) {
            labelSeq[i] += rnd.nextGaussian() * 0.0005; // bruit faible, ajustable
        }*/


        // Amplification du label si la std est trop faible (mode agressif)
        double mean = 0.0;
        for (double v : labelSeq) mean += v;
        mean /= labelSeq.length;
        double std = 0.0;
        for (double v : labelSeq) std += (v - mean) * (v - mean);
        std = Math.sqrt(std / labelSeq.length);
        double minStd = 0.001; // seuil de volatilité minimale (plus agressif)
        double amplifyFactor = 1000.0; // facteur d'amplification très agressif
        if (std < minStd) {
            logger.warn("[TRAIN][LABEL][AGGRESSIVE] std trop faible ({}) => amplification massive des labels par {}", std, amplifyFactor);
            for (int i = 0; i < labelSeq.length; i++) {
                labelSeq[i] *= amplifyFactor;
            }
        }


        // Normalisation des scalers
        ScalerSet scalers = new ScalerSet();
        for (int f = 0; f < numFeatures; f++) {
            double[] col = new double[numSeq + windowSize];
            for (int i = 0; i < numSeq + windowSize; i++) {
                col[i] = matrix[i][f]; // Toutes les valeurs de la feature f
            }
            FeatureScaler.Type type =
                getFeatureNormalizationType(features.get(f)).equals("zscore")
                    ? FeatureScaler.Type.ZSCORE    // Normalisation (x - mean) / std
                    : FeatureScaler.Type.MINMAX;   // Normalisation (x - min) / (max - min)
            FeatureScaler scaler = new FeatureScaler(type);
            scaler.fit(col); // Calcule min/max ou mean/std selon le type
            scalers.featureScalers.put(features.get(f), scaler);
        }
        FeatureScaler labelScaler = new FeatureScaler(config.isUseLogReturnTarget() ? FeatureScaler.Type.ZSCORE : FeatureScaler.Type.MINMAX);
        labelScaler.fit(labelSeq);
        scalers.labelScaler = labelScaler;
        // Étape 22: stocker distribution label brute (log-return ou moyenne multi-horizon) pour dérive future
        if (config.isUseLogReturnTarget()) {
            double m = 0; for (double v : labelSeq) m += v; m = labelSeq.length>0? m/labelSeq.length:0;
            double var=0; for (double v: labelSeq){ double d=v-m; var+=d*d; } var = labelSeq.length>0? var/labelSeq.length:0; double stdLabelDist = Math.sqrt(var);
            scalers.labelDistMean = m;
            scalers.labelDistStd = stdLabelDist>1e-12? stdLabelDist : 1e-12; // éviter division par 0
            logger.info("[STEP22][LABEL_DIST][TRAIN] mean={} std={}", String.format(java.util.Locale.US, "%.6f", scalers.labelDistMean), String.format(java.util.Locale.US, "%.6f", scalers.labelDistStd));
        }

        // Vérification qualité normalisation label (std ≈ 1 si ZSCORE)
        double[] normLabels = scalers.labelScaler.transform(labelSeq);
        if (labelScaler.type == FeatureScaler.Type.ZSCORE) {
            double m=0, v=0; int n=normLabels.length;
            for(double d: normLabels) m += d; m = n>0? m/n:0;
            for(double d: normLabels) v += (d-m)*(d-m); v = n>0? v/n:0; double normStd = Math.sqrt(v);
            if (normStd < 1e-3) {
                logger.warn("[TRAIN][LABEL][WARN] Std normalisée très faible (<1e-3) => plateau potentiel. std={}", normStd);
            } else {
                logger.debug("[TRAIN][LABEL] Normalisation label ZSCORE ok. mean={} std={}", String.format("%.4f", m), String.format("%.4f", normStd));
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

        // Split interne validation 85/15 (restauré après insertion Step2)
        int numSeqTotal = (int) X.size(0);
        int valCount = (int) Math.round(numSeqTotal * 0.15);
        if (valCount < 1 && numSeqTotal >= 20) valCount = 1; // sécurité
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
            logger.info("[TRAIN][VAL] Activation split validation interne 85%/15% (train={} val={})", trainCount, valCount);
        } else {
            XTrain = X; yTrain = y;
            logger.warn("[TRAIN][VAL][FALLBACK] Validation interne désactivée (trainCount={} valCount={} total={})", trainCount, valCount, numSeqTotal);
        }
        org.nd4j.linalg.dataset.DataSet trainDs = new org.nd4j.linalg.dataset.DataSet(XTrain, yTrain);
        org.nd4j.linalg.dataset.DataSet valDs = null;
        if (useInternalVal) {
            valDs = new org.nd4j.linalg.dataset.DataSet(XVal, yVal);
        }

        // ===== STEP2: Ajustement dynamique batch size (exploration GPU) =====
        int requestedBatch = config.getBatchSize();
        int effectiveBatchSize = requestedBatch;
        int trainSeqCount = (int) XTrain.size(0);
        // Nouvelle logique : maximiser la variabilité des gradients
        if (trainSeqCount >= 128) {
            effectiveBatchSize = Math.min(requestedBatch * 2, trainSeqCount / 2); // batch plus grand si dataset large
        } else if (trainSeqCount >= 32) {
            effectiveBatchSize = Math.max(16, requestedBatch); // batch intermédiaire
        } else {
            effectiveBatchSize = Math.max(8, Math.min(requestedBatch, trainSeqCount)); // batch réduit si dataset petit
        }
        // Sécurité : batch ne doit pas dépasser le nombre de séquences
        if (effectiveBatchSize > trainSeqCount) effectiveBatchSize = trainSeqCount;
        int patienceValLocal = config.getPatienceVal() > 0 ? config.getPatienceVal() : 5;
        // Augmenter la patience pour laisser le modèle explorer plus de solutions
        patienceValLocal = Math.max(patienceValLocal, 20);
        int patience = config.getPatience() > 0 ? config.getPatience() : 5;
        patience = Math.max(patience, 20);
        // Augmenter le nombre d'epochs pour exploration
        int epochs = config.getNumEpochs();
        epochs = Math.max(epochs, 100);
        if (effectiveBatchSize > requestedBatch) {
            int old = patienceValLocal;
            patienceValLocal = Math.max(old + 1, (int) Math.round(old * 1.3));
            logger.info("[TRAIN][BATCH][STEP2] Augmentation patienceVal (+30%) {} -> {}", old, patienceValLocal);
        }
        logger.info("[TRAIN][BATCH][STEP2] BatchSize={} trainSeq={} totalSeq={} features={} window={} patienceVal={} (requested={})", effectiveBatchSize, trainSeqCount, numSeqTotal, numFeatures, windowSize, patienceValLocal, requestedBatch);
        org.nd4j.linalg.dataset.api.iterator.DataSetIterator iterator = new ListDataSetIterator<>(trainDs.asList(), effectiveBatchSize);
        if (config.isUseAsyncIterator()) {
            int qSize = config.getAsyncQueueSize();
            if (qSize < 2) qSize = 2; else if (qSize > 8) qSize = 8; // bornes de sécurité mémoire
            logger.info("[TRAIN][ASYNC][STEP4] Activation AsyncDataSetIterator queueSize={} (batch={} seqTrain={})", qSize, effectiveBatchSize, trainSeqCount);
            iterator = new AsyncDataSetIterator(iterator, qSize);
        } else {
            logger.info("[TRAIN][ASYNC][STEP4] Async iterator désactivé (useAsyncIterator=false)");
        }
        // ===== PHASE 9: BOUCLE D'ENTRAÎNEMENT PRINCIPALE =====


        // Timestamp de début pour mesure de performance globale
        long t0 = System.currentTimeMillis();

        // Variables pour Early Stopping et sauvegarde du meilleur modèle
        double bestScore = Double.POSITIVE_INFINITY; // score train (fallback)
        double bestValLoss = Double.POSITIVE_INFINITY; // suivi validation
        MultiLayerNetwork bestModel = null;              // Sauvegarde du meilleur modèle

        double minDelta = config.getMinDelta();          // Amélioration minimale
        int epochsWithoutImprovement = 0;               // Compteur train
        int epochsWithoutValImprovement = 0;            // Compteur validation
        int patienceVal = patienceValLocal; // remplace valeur initiale (après ajustement Step2)
        Double lastValLoss = null; // Step13: stocke le dernier valLoss

        // Étape 10: holder baseline variance résiduelle pour suivi amélioration HUBER
        Double[] baselineResidualVarHolder = new Double[]{null};

        // Étape 12: Variables scheduler LR (réduction multiplicative simple toutes les 25 epochs)
        double currentLearningRate = config.getLearningRate();
        boolean lrReducedOnce = false;                        // Indique si une première réduction a eu lieu
        double bestValLossAtFirstLrReduction = Double.NaN;    // Snapshot du bestValLoss au moment de la 1ère réduction
        boolean lrFirstReductionImproved = false;             // Flag acceptation: amélioration après 1ère réduction

        // === STEP GA (Gradient Accumulation simulée) ===
        // Objectif: si trainSeqCount est très petit mais batch cible demandé élevé, on simule un batch plus large
        // en effectuant plusieurs passes (accumulation implicite) avec un learning rate divisé pour approximer
        // l'effet d'un grand batch (stabilité). On reste minimaliste pour ne pas complexifier la logique.
        boolean useGradAccum = false;
        int gradAccumSteps = 1;
        int targetBatchRequest = requestedBatch; // batch demandé initialement par config
        // Critères: dataset minuscule (< 64) ET batch demandé au moins 2x plus grand que dataset
        if (trainSeqCount > 0 && trainSeqCount < 64 && targetBatchRequest > trainSeqCount * 2) {
            // Nombre de passes pour approximer un batch effectif proche du batch demandé
            gradAccumSteps = (int) Math.ceil((double) targetBatchRequest / Math.max(1.0, trainSeqCount));
            if (gradAccumSteps > 1) {
                useGradAccum = true;
                double baseLR = currentLearningRate;
                double scaledLR = baseLR / gradAccumSteps; // réduction proportionnelle
                if (scaledLR < 1e-7) scaledLR = 1e-7;      // floor sécurité
                applyLearningRate(model, scaledLR);
                logger.warn("[TRAIN][GA] Activation gradient accumulation simulée: trainSeq={} requestedBatch={} steps={} LR {} -> {}", trainSeqCount, targetBatchRequest, gradAccumSteps,
                        String.format(java.util.Locale.US, "%.8f", baseLR), String.format(java.util.Locale.US, "%.8f", scaledLR));
            }
        } else {
            logger.debug("[TRAIN][GA] Accumulation ignorée (trainSeq={} requestedBatch={})", trainSeqCount, targetBatchRequest);
        }

        // Boucle d'entraînement epoch par epoch
        for (int epoch = 1; epoch <= epochs; epoch++) {
            if (useGradAccum) {
                for (int step = 0; step < gradAccumSteps; step++) {
                    iterator.reset();
                    model.fit(iterator);
                }
            } else {
                iterator.reset();
                model.fit(iterator);
            }
            // Monitoring rapide post-fit (paramètres) pour explosion potentielle sans clipping
            try {
                org.nd4j.linalg.api.ndarray.INDArray paramsEpoch = model.params();
                double maxAbs = paramsEpoch.amaxNumber().doubleValue();
                if (Double.isNaN(maxAbs) || Double.isInfinite(maxAbs)) {
                    logger.error("[TRAIN][MONITOR][CRIT] Paramètres contiennent NaN/Inf après epoch {} => arrêt anticipé (réactiver gradient clipping)", epoch);
                    break;
                }
                if (epoch % 5 == 0) {
                    double l2 = paramsEpoch.norm2Number().doubleValue();
                    if (l2 > 1e6) {
                        logger.warn("[TRAIN][MONITOR] Norme L2 paramètres élevée={} (>1e6) epoch={} (risque explosion gradients)", String.format(java.util.Locale.US, "%.3e", l2), epoch);
                    }
                }
            } catch (Exception eMon) {
                logger.debug("[TRAIN][MONITOR] Skip param check epoch={} cause={}", epoch, eMon.toString());
            }
            // Calcul des losses train & validation après fit (cohérentes avec état courant)
            double trainLoss = model.score(trainDs);
            // === Pénalité platitude ===
            double flatPenalty = 0.0;
            try {
                org.nd4j.linalg.api.ndarray.INDArray predsTrain = model.output(XTrain, false);
                int nPred = (int) predsTrain.size(0);
                double meanPred = 0.0;
                for (int i = 0; i < nPred; i++) {
                    meanPred += predsTrain.getDouble(i);
                }
                meanPred /= nPred > 0 ? nPred : 1;
                double varPred = 0.0;
                for (int i = 0; i < nPred; i++) {
                    double d = predsTrain.getDouble(i) - meanPred;
                    varPred += d * d;
                }
                varPred /= nPred > 0 ? nPred : 1;
                double flatThreshold = 2; // seuil de variance minimale
                double alpha = 8; // force de la pénalité
                if (varPred < flatThreshold) {
                    flatPenalty = alpha * (flatThreshold - varPred);
                    trainLoss += flatPenalty;
                    logger.warn("[TRAIN][FLAT-PENALTY] Pénalité platitude appliquée: varPred={} < seuil={} => penalty={}", String.format("%.6f", varPred), String.format("%.6f", flatThreshold), String.format("%.6f", flatPenalty));
                }
            } catch (Exception e) {
                logger.debug("[TRAIN][FLAT-PENALTY] Erreur calcul pénalité platitude: {}", e.toString());
            }
            // Sécurité NaN sur loss
            if (Double.isNaN(trainLoss) || Double.isInfinite(trainLoss)) {
                logger.error("[TRAIN][MONITOR][CRIT] trainLoss={} (NaN/Inf) epoch {} -> arrêt (réactiver gradient clipping)", trainLoss, epoch);
                break;
            }
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
        if (iterator instanceof AsyncDataSetIterator) {
            try { ((AsyncDataSetIterator) iterator).shutdown(); } catch (Exception e) { logger.warn("[TRAIN][ASYNC][STEP4] Échec shutdown async iterator: {}", e.toString()); }
        }
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
        org.nd4j.linalg.api.ndarray.INDArray input =
            Nd4j.create(seq)                    // Création tenseur initial [1, windowSize, numFeatures]
                .permute(0, 2, 1)              // Permutation: [batch, time, features] -> [batch, features, time]
                .dup('c');                     // Copie contiguë en mémoire pour performance

        // ===== VALIDATION DES DIMENSIONS APRÈS PERMUTATION =====
        // Vérification critique pour détecter les erreurs de shape
        if (input.size(1) != numFeatures || input.size(2) != windowSize) {
            logger.info("[SHAPE][PRED] Incohérence shape input: expected features={} time={} got features={} time={}",
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
        // ===== ADAPTATION SWING PRO (PHASE 8 BIS) : CLAMP DISTRIBUTION LOG-RETURN =====
        // Objectif: éviter des extrapolations extrêmes hors distribution entraînement (mean ± 4σ)
        double predTargetEffective = predTarget;
        if (config.isUseLogReturnTarget() && scalers.labelDistMean != null && scalers.labelDistStd != null && scalers.labelDistStd > 0) {
            double maxStd = 4.0; // bande extrême tolérée
            double low = scalers.labelDistMean - maxStd * scalers.labelDistStd;
            double high = scalers.labelDistMean + maxStd * scalers.labelDistStd;
            if (predTargetEffective < low || predTargetEffective > high) {
                double before = predTargetEffective;
                predTargetEffective = Math.max(low, Math.min(high, predTargetEffective));
                logger.info("[PREDICT][CLAMP] logReturn clamp {} -> {} (mean={} std={} range=[{},{}])",
                        String.format("%.6f", before),
                        String.format("%.6f", predTargetEffective),
                        String.format("%.6f", scalers.labelDistMean),
                        String.format("%.6f", scalers.labelDistStd),
                        String.format("%.6f", low),
                        String.format("%.6f", high));
            }
        }
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
            predicted = referencePrice * Math.exp(predTargetEffective);
            logger.info("[PREDICT][LOG-RETURN] referencePrice={} predTargetEff={} predicted={}",
                    String.format("%.3f", referencePrice),
                    String.format("%.6f", predTargetEffective),
                    String.format("%.3f", predicted));
        } else {
            predicted = predTargetEffective;
        }

        // ===== ADAPTATION SWING PRO (PHASE 10 BIS) : AJUSTEMENTS VOLATILITÉ & DEADZONE =====
        try {
            // ATR% courant pour calibrer l'amplitude utile d'un swing professionnel
            ATRIndicator atrInd = new ATRIndicator(series, 14);
            double atrVal = atrInd.getValue(series.getEndIndex()).doubleValue();
            double atrPct = atrVal > 0 && referencePrice > 0 ? atrVal / referencePrice : 0.0;
            if (Double.isFinite(atrPct) && referencePrice > 0) {
                double deltaPct = (predicted - referencePrice) / referencePrice;
                // Environnement très calme (<0.30% ATR) => compression
                if (atrPct < 0.003) {
                    double scale = Math.max(0.25, atrPct / 0.003);
                    double before = predicted;
                    predicted = referencePrice + deltaPct * scale * referencePrice;
                    logger.info("[PREDICT][ADAPT][LOWVOL] atrPct={} scale={} before={} after={}",
                            String.format("%.5f", atrPct),
                            String.format("%.3f", scale),
                            String.format("%.3f", before),
                            String.format("%.3f", predicted));
                }
                else if (atrPct > 0.02) { // volatilité extrême
                    double scale = Math.min(1.0, 0.02 / atrPct * 1.2);
                    if (scale < 1.0) {
                        double before = predicted;
                        predicted = referencePrice + deltaPct * scale * referencePrice;
                        logger.info("[PREDICT][ADAPT][HIGHVOL] atrPct={} scale={} before={} after={}",
                                String.format("%.5f", atrPct),
                                String.format("%.3f", scale),
                                String.format("%.3f", before),
                                String.format("%.3f", predicted));
                    }
                }
                // Deadzone configurable
                double swingTh = computeSwingTradeThreshold(series, config);
                boolean disableDead = false;
                double dzFactor = 0.5; // défaut legacy
                try { disableDead = config.isDisableDeadzone(); dzFactor = config.getDeadzoneFactor(); } catch (Exception ignored) {}
                if (dzFactor <= 0) dzFactor = 0.5; else if (dzFactor > 0.9) dzFactor = 0.9; // bornes sécurité
                if (!disableDead) {
                    if (Math.abs((predicted - referencePrice) / referencePrice) < swingTh * dzFactor) {
                        double before = predicted;
                        predicted = referencePrice; // neutralise
                        logger.info("[PREDICT][DEADZONE-CFG] deltaPct={} < swingTh*{} (={}) -> neutralisation ({} -> {})",
                                String.format("%.5f", deltaPct),
                                String.format("%.2f", dzFactor),
                                String.format("%.5f", swingTh * dzFactor),
                                String.format("%.3f", before),
                                String.format("%.3f", predicted));
                    }
                } else {
                    logger.trace("[PREDICT][DEADZONE-CFG] deadzone désactivée");
                }
            }
        } catch (Exception e) {
            logger.info("[PREDICT][ADAPT] Skip ajustements swing: {}", e.toString());
        }
        // Sécurité prix non positif (marché fermé ou anomalie) => fallback last
        if (!(predicted > 0)) {
            logger.warn("[PREDICT][SAFETY] predicted<=0 -> fallback referencePrice ({})", String.format("%.3f", referencePrice));
            predicted = referencePrice;
        }
        // ===== Correction : amplification si delta trop faible =====
        double deltaPct = referencePrice > 0 ? (predicted - referencePrice) / referencePrice : 0.0;
        double minDelta = 0.001; // 0.1% minimum
        if (Math.abs(deltaPct) < minDelta && predicted != referencePrice) {
            double before = predicted;
            double amplify = deltaPct > 0 ? minDelta : -minDelta;
            predicted = referencePrice * (1.0 + amplify);
            logger.debug("[PREDICT][AMPLIFY] predicted trop proche de referencePrice : avant={} après={} deltaPct={}",
                String.format("%.3f", before),
                String.format("%.3f", predicted),
                String.format("%.5f", deltaPct));
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
        logger.info("[PREDICT] Prédiction: referencePrice={}, predicted={}, predNorm={}, predTargetEff={}, limitPct={}",
                String.format("%.3f", referencePrice),
                String.format("%.3f", predicted),
                String.format("%.6f", predNorm),
                String.format("%.6f", predTargetEffective),
                limitPct);

        // Retour de la prédiction finale ajustée et limitée
        return predicted;
    }

    /**
     * Alias conservant signature existante.
     */
    public double predictNextCloseWithScalerSet(BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers){
        return predictNextCloseScalarFast(series, config, model, scalers);
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
        if (config.isUseLogReturnTarget()) {
            return referencePrice * Math.exp(predTarget);
        } else {
            return predTarget;
        }
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
        // Améliorations swing: métriques R (risk multiples)
        public double avgR;           // moyenne des R multiples par trade
        public double medianR;        // médiane des R
        public double rWinRate;       // proportion trades R>0
        public double positiveRSkew;  // (mean R positif - |mean R négatif|) / (|mean R négatif|+1e-9)
        public int partialExitTrades; // nb trades avec partial take profit
        // Nouvelle métrique amplitude prédictions (moyenne |pred - close_t| / close_t)
        public double meanAbsPredDelta;
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
            // Ajout du contrôle pour que testEndBar ne dépasse jamais la taille de la série
            if (testEndBar > series.getBarCount()) {
                testEndBar = series.getBarCount();
            }
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

            TradingMetricsV2 metrics = simulateTradingWalkForwardBis(
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
            metrics.meanAbsPredDelta = lastComputedMeanAbsPredDelta.get();

            // Calcul du business score
            double businessScore = computeBusinessScore(
                metrics.profitFactor, metrics.winRate,
                metrics.maxDrawdownPct, metrics.expectancy, config
            );
            // Bonus léger contre sorties trop plates (amplitude moyenne des prédictions)
            // clamp sur 2% (0.02) pour éviter inflation excessive si volatilité extrême
            double ampClamp = 0.02;
            double amp = metrics.meanAbsPredDelta; // déjà calculé lors du MSE
            if (Double.isFinite(amp)) {
                double ampBonus = 1.0 + 0.05 * Math.min(Math.max(amp, 0.0), ampClamp);
                businessScore *= ampBonus;
            }
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

            // ===== LIBÉRATION MÉMOIRE GPU APRÈS CHAQUE SPLIT =====
            try {
                org.nd4j.linalg.factory.Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
            } catch (Exception e) {
                logger.warn("Erreur lors de la libération mémoire GPU : {}", e.getMessage());
            }
            System.gc();
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

    public TradingMetricsV2 simulateTradingWalkForwardBis(
            BarSeries fullSeries,
            int testStartBar,
            int testEndBar,
            MultiLayerNetwork model,
            ScalerSet scalers,
            LstmConfig config
    ) {
        TradingMetricsV2 tm = new TradingMetricsV2();
        boolean inPosition = false;
        double entryPrice = 0.0;
        int numTrades = 0;
        int winTrades = 0;
        double totalProfit = 0.0;
        List<Double> tradeProfits = new ArrayList<>();
        List<Integer> barsHeldList = new ArrayList<>();
        List<Double> positionValues = new ArrayList<>();
        List<Double> tradeRMultiples = new ArrayList<>();
        double[] closes = extractCloseValues(fullSeries);
        int barsInPos = 0;
        double initialRiskPerShare = 0.0;
        double positionSize = 1.0;
        double stopLoss = 0.0;
        double capital = config != null && config.getCapital() > 0 ? config.getCapital() : 1.0;
        double riskPct = config != null ? config.getRiskPct() : 0.01;
        double meanAbsPredDeltaSum = 0.0;
        int meanAbsPredDeltaCount = 0;
        List<Double> equityCurve = new ArrayList<>();
        double currentEquity = capital;
        for (int i = testStartBar; i <= testEndBar; i++) {
            BarSeries subSeries = fullSeries.getSubSeries(0, i + 1);
            TradeStylePrediction pred = predictTradeStyle("", subSeries, config, model, scalers);
            double close = subSeries.getLastBar().getClosePrice().doubleValue();
            double predicted = pred.predictedClose;
            if (Double.isFinite(predicted) && close != 0.0) {
                meanAbsPredDeltaSum += Math.abs(predicted - close) / close;
                meanAbsPredDeltaCount++;
            }
            // --- AJOUT DU CONTROLE DRAWDOWN ---
            if (inPosition) {
                double latentDrawdown = entryPrice > 0 ? (entryPrice - close) / entryPrice : 0.0;
                if ((latentDrawdown >= config.getStopLossPct() || -latentDrawdown >= config.getTakeProfitPct())
                        && !"BUY".equals(pred.action)) {
                    // Vente forcée si drawdown >= 10% et action HOLD ou SELL
                    double profit = (close - entryPrice) * positionSize;
                    totalProfit += profit;
                    tradeProfits.add(profit);
                    numTrades++;
                    if (profit > 0) winTrades++;
                    barsHeldList.add(barsInPos);
                    double r = initialRiskPerShare > 0 ? profit / (initialRiskPerShare * positionSize) : 0.0;
                    tradeRMultiples.add(r);
                    inPosition = false;
                    entryPrice = 0;
                    barsInPos = 0;
                    initialRiskPerShare = 0;
                    positionSize = 1.0;
                    stopLoss = 0.0;
                    // On continue à la prochaine barre
                    currentEquity = capital + totalProfit;
                    equityCurve.add(currentEquity);
                    continue;
                }
            }
            if ("BUY".equals(pred.action)) {
                if (!inPosition) {
                    inPosition = true;
                    entryPrice = close;
                    barsInPos = 0;
                    // Calcul du stop et positionSize
                    double atr = 0.0;
                    try {
                        ATRIndicator atrInd = new ATRIndicator(subSeries, 14);
                        atr = atrInd.getValue(subSeries.getEndIndex()).doubleValue();
                    } catch (Exception e) {}
                    double stopDistance = atr > 0 ? atr : entryPrice * 0.01;
                    stopLoss = entryPrice - stopDistance;
                    initialRiskPerShare = entryPrice - stopLoss;
                    double riskAmount = capital * riskPct;
                    positionSize = initialRiskPerShare > 0 ? riskAmount / initialRiskPerShare : 1.0;
                    positionValues.add(positionSize * entryPrice);
                }
            } else if ("SELL".equals(pred.action)) {
                if (inPosition) {
                    double profit = (close - entryPrice) * positionSize;
                    totalProfit += profit;
                    tradeProfits.add(profit);
                    numTrades++;
                    if (profit > 0) winTrades++;
                    barsHeldList.add(barsInPos);
                    double r = initialRiskPerShare > 0 ? profit / (initialRiskPerShare * positionSize) : 0.0;
                    tradeRMultiples.add(r);
                    inPosition = false;
                    entryPrice = 0;
                    barsInPos = 0;
                    initialRiskPerShare = 0;
                    positionSize = 1.0;
                }
            }
            // Mise à jour de la courbe d'equity à chaque barre
            if (inPosition) {
                // Valeur latente de la position + capital
                currentEquity = capital + (close - entryPrice) * positionSize;
            } else {
                currentEquity = capital + totalProfit;
            }
            equityCurve.add(currentEquity);
            if (inPosition) barsInPos++;
        }
        // Si position ouverte à la fin, on la clôture au dernier prix
        if (inPosition) {
            double close = fullSeries.getBarCount() > testEndBar
                    ? fullSeries.getBar(testEndBar).getClosePrice().doubleValue()
                    : fullSeries.getLastBar().getClosePrice().doubleValue();
            if(testEndBar >= fullSeries.getBarCount()){
                logger.warn("[simulateTradingWalkForwardBis][warn] index testEndBar={} fullSeries.getBarCount={}", testEndBar, fullSeries.getBarCount());
            }
            double profit = (close - entryPrice) * positionSize; // Correction ici
            totalProfit += profit;
            tradeProfits.add(profit);
            numTrades++;
            if (profit > 0) winTrades++;
            barsHeldList.add(barsInPos);
            double r = initialRiskPerShare > 0 ? profit / (initialRiskPerShare * positionSize) : 0.0;
            tradeRMultiples.add(r);

            System.out.println(String.format(
                    "[EXIT] fin Entry=%.4f | Duration=%d | Exit=%.4f | Qty=%.2f | Profit=%.4f",
                    entryPrice, barsInPos, close, positionSize, profit
            ));
        }
        tm.totalProfit = totalProfit;
        tm.numTrades = numTrades;
        tm.winRate = numTrades > 0 ? (double) winTrades / numTrades : 0.0;
        // Calculs réels des métriques principales
        double gains = 0, losses = 0; int win = 0, loss = 0;
        for (double p : tradeProfits) {
            if (p > 0) { gains += p; win++; }
            else { losses += p; loss++; }
        }
        tm.profitFactor = losses != 0 ? gains / Math.abs(losses) : (gains > 0 ? Double.POSITIVE_INFINITY : 0);
        // Correction du calcul du max drawdown
        double maxDrawdown = 0.0;
        double maxEquity = 0.0;
        if (!equityCurve.isEmpty()) {
            for (double equity : equityCurve) {
                if (equity > maxEquity) maxEquity = equity;
                double drawdown = maxEquity - equity;
                if (drawdown > maxDrawdown) maxDrawdown = drawdown;
            }
        }
        tm.maxDrawdownPct = maxEquity != 0 ? maxDrawdown / Math.abs(maxEquity) : 0.0;
        double avgGain = win > 0 ? gains / win : 0, avgLoss = loss > 0 ? Math.abs(losses) / loss : 0;
        tm.expectancy = (win + loss) > 0 ? (tm.winRate * avgGain - (1 - tm.winRate) * avgLoss) : 0;
        double meanRet = tradeProfits.stream().mapToDouble(d -> d).average().orElse(0);
        double stdRet = Math.sqrt(tradeProfits.stream().mapToDouble(d -> { double m = d - meanRet; return m * m; }).average().orElse(0));
        tm.sharpe = stdRet > 0 ? meanRet / stdRet * Math.sqrt(Math.max(1, tradeProfits.size())) : 0;
        double downsideStd = Math.sqrt(tradeProfits.stream().filter(r -> r < meanRet).mapToDouble(r -> { double dr = r - meanRet; return dr * dr; }).average().orElse(0));
        tm.sortino = downsideStd > 0 ? meanRet / downsideStd : 0;
        tm.turnover = numTrades > 0 ? (double) numTrades / 252.0 : 0.0;
        double annualizedReturn = capital > 0 ? tm.totalProfit / capital : tm.totalProfit;
        tm.calmar = tm.maxDrawdownPct > 0 ? annualizedReturn / tm.maxDrawdownPct : 0;
        // Calcul métriques avancées demandées
        int totalTestBars = testEndBar - testStartBar + 1;
        int totalBarsInPosition = barsHeldList.stream().mapToInt(i -> i).sum();
        tm.exposure = totalTestBars > 0 ? (double) totalBarsInPosition / totalTestBars : 0.0;
        tm.avgBarsInPosition = barsHeldList.isEmpty() ? 0.0 : barsHeldList.stream().mapToInt(i -> i).average().orElse(0.0);
        // MSE sur les trades (entry/exit vs close)
        double mseSum = 0.0; int mseCount = 0;
        for (int idx = 0, j = 0; idx < tradeProfits.size(); idx++) {
            // On approxime l'entrée/sortie par le prix d'entrée et de sortie
            // Ici, on ne stocke que le profit, donc on ne peut pas faire mieux sans refactor
            // On laisse à 0 si non calculable
        }
        tm.mse = 0.0; // Non calculable sans refactor
        tm.businessScore = computeBusinessScore(tm.profitFactor, tm.winRate, tm.maxDrawdownPct, tm.expectancy, config);
        tm.contrarianTrades = 0; // Non simulé ici
        tm.normalTrades = numTrades;
        tm.contrarianRatio = 0.0;
        // Position value mean/std
        if (!positionValues.isEmpty()) {
            double sumPV = 0.0;
            for (double v : positionValues) sumPV += v;
            double meanPV = sumPV / positionValues.size();
            double varPV = 0.0;
            for (double v : positionValues) { double d = v - meanPV; varPV += d * d; }
            varPV /= positionValues.size();
            tm.positionValueMean = meanPV;
            tm.positionValueStd = Math.sqrt(varPV);
        } else {
            tm.positionValueMean = 0.0;
            tm.positionValueStd = 0.0;
        }
        // R multiples
        if (!tradeRMultiples.isEmpty()) {
            tm.avgR = tradeRMultiples.stream().mapToDouble(d -> d).average().orElse(0.0);
            double[] sortedR = tradeRMultiples.stream().mapToDouble(d -> d).sorted().toArray();
            tm.medianR = sortedR.length % 2 == 1 ? sortedR[sortedR.length / 2] : (sortedR[sortedR.length / 2 - 1] + sortedR[sortedR.length / 2]) / 2.0;
            long winsR = tradeRMultiples.stream().filter(r -> r > 0).count();
            tm.rWinRate = numTrades > 0 ? (double) winsR / numTrades : 0.0;
            double meanPos = tradeRMultiples.stream().filter(r -> r > 0).mapToDouble(r -> r).average().orElse(0.0);
            double meanNeg = tradeRMultiples.stream().filter(r -> r < 0).mapToDouble(r -> r).average().orElse(0.0);
            tm.positiveRSkew = (meanPos - Math.abs(meanNeg)) / (Math.abs(meanNeg) + 1e-9);
        } else {
            tm.avgR = 0.0;
            tm.medianR = 0.0;
            tm.rWinRate = 0.0;
            tm.positiveRSkew = 0.0;
        }
        tm.partialExitTrades = 0; // Non simulé ici
        tm.meanAbsPredDelta = meanAbsPredDeltaCount > 0 ? meanAbsPredDeltaSum / meanAbsPredDeltaCount : 0.0;
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
        // Pour amplitude
        double sumAbsPredDelta = 0; int countAmp = 0;

        for (int t = testStartBar; t < testEndBar; t++) {
            if (t - window < 1) continue; // si log-return => besoin de t-1
            BarSeries sub = series.getSubSeries(0, t); // exclut t (label à t)
            double pred = predictNextCloseScalarFast(sub, config, model, scalers);
            double actual = closes[t];
            if (Double.isFinite(pred) && Double.isFinite(actual)) {
                double diff = pred - actual;
                se += diff * diff;
                count++;
            }
            // Amplitude par rapport au dernier close disponible (t-1)
            double ref = closes[t-1];
            if (Double.isFinite(pred) && Double.isFinite(ref) && ref != 0) {
                sumAbsPredDelta += Math.abs(pred - ref) / ref;
                countAmp++;
            }
        }
        double mse = count > 0 ? se / count : Double.NaN;
        // Injection amplitude dans dernier split metrics si accessible via thread local? -> Simplifié: on stocke dans ThreadLocal static
        lastComputedMeanAbsPredDelta.set(countAmp>0? sumAbsPredDelta / countAmp : 0.0);
        return mse;
    }

    // ThreadLocal pour laisser computeSplitMse communiquer l'amplitude au caller (évite refactor large signature)
    private static final ThreadLocal<Double> lastComputedMeanAbsPredDelta = ThreadLocal.withInitial(() -> 0.0);

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
            logger.debug("[SEUIL SWING][ATR] rawThreshold={}% (ATR={}, close={}, k={})", String.format("%.4f", rawThreshold * 100), lastATR, lastClose, k);
        } else if ("returns".equalsIgnoreCase(type)) {
            double[] closes = extractCloseValues(series);
            if (closes.length < 3) rawThreshold = 0.005; else {
                double[] logRet = new double[closes.length - 1];
                for (int i = 1; i < closes.length; i++) logRet[i - 1] = Math.log(closes[i] / closes[i - 1]);
                double mean = Arrays.stream(logRet).average().orElse(0);
                double std = Math.sqrt(Arrays.stream(logRet).map(r -> (r - mean) * (r - mean)).sum() / logRet.length);
                rawThreshold = k * std;
                logger.debug("[SEUIL SWING][RETURNS] rawThreshold={}% (std={}, k={})", String.format("%.4f", rawThreshold * 100), std, k);
            }
        } else {
            rawThreshold = 0.01 * k;
        }
        // Bornes dynamiques depuis config
        double minAllowedThreshold = config.getThresholdAtrMin() > 0 ? config.getThresholdAtrMin() : 0.001;
        double maxAllowedThreshold = config.getThresholdAtrMax() > minAllowedThreshold ? config.getThresholdAtrMax() : 0.01;
        double adjustedThreshold = Math.max(minAllowedThreshold, Math.min(rawThreshold, maxAllowedThreshold));
        if (Math.abs(adjustedThreshold - rawThreshold) > 0.0001) {
            logger.info("[SEUIL SWING][ADJUST] Seuil ajusté: {}% -> {}% (bornes: {}%-{}%)", String.format("%.4f", rawThreshold * 100), String.format("%.4f", adjustedThreshold * 100), String.format("%.1f", minAllowedThreshold * 100), String.format("%.1f", maxAllowedThreshold * 100));
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
        if (n == 0) return   0;
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
     * @param series  Série complète
     * @param config  Config LSTM
     * @param model   Modèle potentiellement déjà en mémoire
     * @param scalers Scalers associés
     * @return Objet PreditLsdm (DTO)
     */
    public PreditLsdm getPredit(BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers) {
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

        // Seuil swing de base (ATR ou returns selon config)
        double th = computeSwingTradeThreshold(series, config);
        double predicted = predictNextCloseWithScalerSet(series, config, model, scalers);
        predicted = Math.round(predicted * 1000.0) / 1000.0;

        double[] closes = extractCloseValues(series);
        double lastClose = closes[closes.length - 1];
        double delta = predicted - lastClose;
        double deltaPct = lastClose > 0 ? delta / lastClose : 0.0;

        // ================== Filtres Swing Pro ==================
        // 1. Volatilité (ATR)
        double atrVal = 0, atrPct = 0;
        try {
            ATRIndicator atrInd = new ATRIndicator(series, 14);
            atrVal = atrInd.getValue(series.getEndIndex()).doubleValue();
            if (lastClose > 0) atrPct = atrVal / lastClose;
        } catch (Exception e) { logger.debug("[PREDIT][ATR] skip {}", e.toString()); }

        // 2. Momentum RSI pour filtrer sur-achat / sur-vente
        double rsiVal = 50.0;
        try {
            RSIIndicator rsi = new RSIIndicator(new ClosePriceIndicator(series), 14);
            rsiVal = rsi.getValue(series.getEndIndex()).doubleValue();
        } catch (Exception e) { logger.debug("[PREDIT][RSI] skip {}", e.toString()); }

        // 3. Volume relatif (logique simple: volume courant vs moyenne 20)
        double volRatio = 1.0;
        try {
            VolumeIndicator volInd = new VolumeIndicator(series);
            double curVol = volInd.getValue(series.getEndIndex()).doubleValue();
            double sumVol = 0.0; int count = 0;
            for (int i = Math.max(0, series.getEndIndex() - 19); i <= series.getEndIndex(); i++) {
                sumVol += volInd.getValue(i).doubleValue(); count++;
            }
            double avgVol = count > 0 ? sumVol / count : curVol;
            if (avgVol > 0) volRatio = curVol / avgVol;
        } catch (Exception e) { logger.debug("[PREDIT][VOL] skip {}", e.toString()); }

        // 4. Confiance: distance relative au seuil principal (normalisée)
        double confidence;
        double absDeltaPct = Math.abs(deltaPct);
        if (absDeltaPct <= th) confidence = 0.0; else confidence = Math.min(1.0, (absDeltaPct - th) / (th * 2.0));

        // 5. Ajustements professionnels: neutraliser signaux faibles ou en conditions défavorables
        boolean weaken = false;
        // Sur-achat / sur-vente extrêmes -> éviter poursuite
        if (deltaPct > 0 && rsiVal > 75) weaken = true; // trop sur-acheté
        if (deltaPct < 0 && rsiVal < 25) weaken = true; // trop survendu (on évite vendre)
        // Volume insuffisant => moins de fiabilité
        if (volRatio < 0.7) weaken = true;
        // Volatilité ultra basse => bruit (delta inférieur à 0.6 * seuil ATR dynamique)
        if (atrPct > 0 && absDeltaPct < Math.max(th * 0.75, atrPct * 0.6)) weaken = true;
        // Confiance faible intrinsèque
        if (confidence < 0.20) weaken = true;

        SignalType signal =
            delta > th ? SignalType.UP :
                (delta < -th ? SignalType.DOWN : SignalType.STABLE);

        if (weaken && signal != SignalType.STABLE) {
            // On réduit le signal à STABLE si les conditions pro ne sont pas réunies
            logger.debug("[PREDIT][FILTER] signal={} affaibli -> STABLE (rsi={} volRatio={} atrPct={} conf={})", signal, String.format("%.1f", rsiVal), String.format("%.2f", volRatio), String.format("%.4f", atrPct), String.format("%.2f", confidence));
            signal = SignalType.STABLE;
            confidence = 0.0;
        }

        // 6. Estimation risk / target professionnelle (basée ATR)
        double riskPerShare = atrVal * 1.3; // stop ~1.3 ATR
        double targetPerShare = atrVal * 2.6; // objectif ~2R
        double provisionalTarget = (signal == SignalType.UP)
            ? lastClose + targetPerShare
            : (signal == SignalType.DOWN)
                ? lastClose - targetPerShare
                : lastClose;
        if (provisionalTarget <= 0) provisionalTarget = lastClose;

        String position = analyzePredictionPosition(
            Arrays.copyOfRange(closes, closes.length - config.getWindowSize(), closes.length),
            predicted
        );

        // Enrichissement du champ position pour exposer les métriques utiles au front / supervision
        String posInfo = String.format(
            java.util.Locale.US,
            "%s|dPct=%.3f%%|th=%.3f%%|ATR%%=%.3f%%|RSI=%.1f|VolR=%.2f|conf=%.2f|risk=%.4f|target=%.4f",
            position,
            deltaPct * 100.0,
            th * 100.0,
            atrPct * 100.0,
            rsiVal,
            volRatio,
            confidence,
            riskPerShare,
            provisionalTarget
        );

        String formattedDate = series.getLastBar()
            .getEndTime()
            .format(java.time.format.DateTimeFormatter.ofPattern("dd/MM"));

        logger.info("[PRED-TRADE] win={} last={} pred={} delta={} ({}%) thr={} ({}) signal={} conf={} rsi={} volR={} atrPct={}",
            config.getWindowSize(),
            String.format(java.util.Locale.US, "%.4f", lastClose),
            String.format(java.util.Locale.US, "%.4f", predicted),
            String.format(java.util.Locale.US, "%.4f", delta),
            String.format(java.util.Locale.US, "%.3f", deltaPct * 100.0),
            String.format(java.util.Locale.US, "%.4f", th),
            config.getThresholdType(),
            signal,
            String.format(java.util.Locale.US, "%.2f", confidence),
            String.format(java.util.Locale.US, "%.1f", rsiVal),
            String.format(java.util.Locale.US, "%.2f", volRatio),
            String.format(java.util.Locale.US, "%.3f", atrPct * 100.0)
        );

        return PreditLsdm.builder()
            .lastClose(lastClose)
            .predictedClose(predicted)
            .signal(SignalType.STABLE) // Signal par défaut, peut être ajusté par la logique métier
            .explication(posInfo)
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
                              int totalSeriesTested,
                              int phase_grid,
                              int number_grid,
                              int phase_1_top_n,
                              String phase_1_top_n_label,
                              boolean holdOut, String tuningResult, double ratio) throws IOException {
        if (model == null) return;
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ModelSerializer.writeModel(model, baos, true);
        byte[] modelBytes = baos.toByteArray();

        // Sauvegarde hyperparamètres via repository + JSON
        // hyperparamsRepository.saveHyperparams(symbol, config);

        com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        String hyperparamsJson = mapper.writeValueAsString(config);
        String scalersJson = mapper.writeValueAsString(scalers);
        double rendement = config.getCapital() > 0 ? (sumProfit / config.getCapital()) : 0.0;

        String sql = "REPLACE INTO lstm_models (symbol, model_blob, hyperparams_json, normalization_scope, scalers_json, mse, profit_factor, " +
                "win_rate, max_drawdown, rmse, sum_profit, total_trades, business_score, total_series_tested, rendement, " +
                "phase_grid, number_grid, phase_1_top_n, phase_1_top_n_label, holdOut, tuning_result_json, ratio, updated_date) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, CURRENT_TIMESTAMP)";
        jdbcTemplate.update(sql, symbol, modelBytes, hyperparamsJson, config.getNormalizationScope(),
                scalersJson, mse, profitFactor, winRate, maxDrawdown, rmse, sumProfit, totalTrades, businessScore, totalSeriesTested, rendement,
                phase_grid, number_grid, phase_1_top_n, phase_1_top_n_label, holdOut, tuningResult, ratio);
    }

    /**
     * Charge modèle + scalers (JSON) + hyperparams.
     */
    public LoadedModel loadModelAndScalersFromDb(String symbol, JdbcTemplate jdbcTemplate) throws IOException {

        // Sélection explicite des colonnes nécessaires
        String sql = "SELECT * FROM lstm_models WHERE symbol = ? order by sum_profit desc limit 1";

        try {
            Map<String,Object> result = jdbcTemplate.queryForMap(sql, symbol);

            // Récupération sécurisée des colonnes
            Object modelBlobObj = result.get("model_blob");
            //byte[] modelBlob = modelBlobObj instanceof byte[] ? (byte[]) modelBlobObj : null;

            byte[] modelBlob = null;
            try {
                if (modelBlobObj instanceof byte[]) {
                    modelBlob = (byte[]) modelBlobObj;
                } else if (modelBlobObj instanceof java.sql.Blob) {
                    java.sql.Blob blob = (java.sql.Blob) modelBlobObj;
                    try (InputStream is = blob.getBinaryStream(); ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
                        byte[] buf = new byte[8192];
                        int r;
                        while ((r = is.read(buf)) > 0) baos.write(buf, 0, r);
                        modelBlob = baos.toByteArray();
                    }
                } else if (modelBlobObj instanceof String) {
                    String s = (String) modelBlobObj;
                    try {
                        modelBlob = java.util.Base64.getDecoder().decode(s);
                    } catch (IllegalArgumentException iae) {
                        // Essayer hex
                        try { modelBlob = hexStringToByteArray(s); } catch (Exception ex) { /* ignore */ }
                    }
                }
            } catch (Exception e) {
                logger.warn("[LOAD][MODEL] erreur lecture model_blob type={} : {}", modelBlobObj != null ? modelBlobObj.getClass().getName() : "null", e.getMessage());
            }


            String scalersJson = result.get("scalers_json") != null ? result.get("scalers_json").toString() : null;
            String hyperparams = result.get("hyperparams_json") != null ? result.get("hyperparams_json").toString() : null;
            if(hyperparams == null) throw new IOException("Aucun hyperparamètre pour " + symbol);
            ObjectMapper mapper = new ObjectMapper();
            LstmConfig config = mapper.readValue(hyperparams, LstmConfig.class);
            String tuning_result = result.get("tuning_result_json") != null ? result.get("tuning_result_json").toString() : null;
            LstmTuningService.TuningResult resultTuning = new Gson().fromJson(tuning_result, LstmTuningService.TuningResult.class);

            // Fonctions utilitaires de parsing
            java.util.function.Function<Object, Double> toDouble = (o) -> {
                if (o == null) return Double.NaN;
                if (o instanceof Number) return ((Number) o).doubleValue();
                try { return Double.parseDouble(o.toString()); } catch (Exception e) { return Double.NaN; }
            };
            java.util.function.Function<Object, Integer> toInt = (o) -> {
                if (o == null) return 0;
                if (o instanceof Number) return ((Number) o).intValue();
                try { return Integer.parseInt(o.toString()); } catch (Exception e) { return 0; }
            };

            double rendement = toDouble.apply(result.get("rendement"));
            int totalSerieTested = toInt.apply(result.get("total_series_tested"));
            int totalTrades = toInt.apply(result.get("total_trades"));
            double businnesScore = toDouble.apply(result.get("business_score"));
            double sumProfil = toDouble.apply(result.get("sum_profit"));
            double maxDrawdown = toDouble.apply(result.get("max_drawdown"));
            double winRate = toDouble.apply(result.get("win_rate"));
            double profitFactor = toDouble.apply(result.get("profit_factor"));
            double ratio = toDouble.apply(result.get("ratio"));
            double phaseD = toDouble.apply(result.get("phase_grid"));

            // Restitution du modèle si présent
            MultiLayerNetwork model = null;
            if (modelBlob != null) {
                // Log utile pour debug : taille et premiers octets (vérifier signature ZIP PK..)
                try {
                    int len = modelBlob.length;
                    byte[] head = Arrays.copyOfRange(modelBlob, 0, Math.min(8, len));
                    logger.info("[LOAD][MODEL] model_blob size={} head={}", len, bytesToHex(head));
                } catch (Exception ignored) {}
                try{
                    ByteArrayInputStream bais = new ByteArrayInputStream(modelBlob);
                    model = ModelSerializer.restoreMultiLayerNetwork(bais);
                } catch (Exception e) {
                    logger.warn("Impossible de restaurer le modèle depuis la BDD pour {}: {}", symbol, e.getMessage());
                    try {
                        // Écrire le blob dans un fichier temporaire pour inspection
                        String tmpDir = System.getProperty("java.io.tmpdir");
                        java.nio.file.Path tmpFile = java.nio.file.Files.createTempFile(java.nio.file.Paths.get(tmpDir), "model_blob_", ".bin");
                        java.nio.file.Files.write(tmpFile, modelBlob);
                        logger.warn("[LOAD][MODEL][DIAG] Dump temporaire écrit: {} (size={})", tmpFile.toString(), modelBlob.length);
                    } catch (Exception ex) {
                        logger.warn("[LOAD][MODEL][DIAG] Impossible d'écrire dump temporaire: {}", ex.getMessage());
                    }

                    // Tenter d'énumérer les entrées ZIP pour voir ce que contient le fichier
                    try (java.io.ByteArrayInputStream bais2 = new java.io.ByteArrayInputStream(modelBlob);
                         java.util.zip.ZipInputStream zis = new java.util.zip.ZipInputStream(bais2)) {
                        java.util.zip.ZipEntry ze;
                        List<String> entries = new ArrayList<>();
                        while ((ze = zis.getNextEntry()) != null) {
                            entries.add(String.format("%s (size=%d)", ze.getName(), ze.getSize()));
                        }
                        logger.warn("[LOAD][MODEL][DIAG] Zip entries: {}", String.join(", ", entries));
                    } catch (Exception ex) {
                        logger.warn("[LOAD][MODEL][DIAG] Impossible de lister entrées ZIP: {}", ex.getMessage());
                    }

                    // Essayer de restaurer en tant que ComputationGraph (au cas où le modèle enregistré serait un ComputationGraph)
                    try (ByteArrayInputStream bais3 = new ByteArrayInputStream(modelBlob)) {
                        try {
                            org.deeplearning4j.nn.graph.ComputationGraph cg = ModelSerializer.restoreComputationGraph(bais3);
                            if (cg != null) {
                                logger.warn("[LOAD][MODEL][DIAG] Le blob contient un ComputationGraph (pas un MultiLayerNetwork). Conversion impossible automatique.");
                            }
                        } catch (Exception ex) {
                            logger.debug("[LOAD][MODEL][DIAG] restoreComputationGraph échoué: {}", ex.getMessage());
                        }
                    } catch (Exception ignored) {}
                }
            }

            // Parse des scalers JSON si présent
            ScalerSet scalers = null;
            if (scalersJson != null && !scalersJson.isBlank()) {
                try {
                    scalers = mapper.readValue(scalersJson, ScalerSet.class);
                } catch (Exception e) {
                    logger.warn("Impossible de parser scalers_json pour {} : {}", symbol, e.getMessage());
                }
            }

            logger.info("Chargé modèle+scalers pour {} (scalers={})", symbol, scalers != null);

            // Construction de l'objet LoadedModel renseigné
            LoadedModel lm = new LoadedModel();
            lm.model = model;
            lm.scalers = scalers;
            lm.config = config;
            lm.resultTuning = resultTuning;
            lm.rendement = Double.isFinite(rendement) ? rendement : 0.0;
            lm.totalSerieTested = totalSerieTested;
            lm.totalTrades = totalTrades;
            lm.businnesScore = Double.isFinite(businnesScore) ? businnesScore : 0.0;
            lm.sumProfil = Double.isFinite(sumProfil) ? sumProfil : 0.0;
            lm.maxDrawdown = Double.isFinite(maxDrawdown) ? maxDrawdown : 0.0;
            lm.winRate = Double.isFinite(winRate) ? winRate : 0.0;
            lm.profitFactor = Double.isFinite(profitFactor) ? profitFactor : 0.0;
            lm.ratio = Double.isFinite(ratio) ? ratio : 0.0;
            lm.phase = (int) Math.round(Double.isFinite(phaseD) ? phaseD : 0.0);

            return lm;

        } catch (EmptyResultDataAccessException e) {
            throw new IOException("Modèle non trouvé");
        }
    }

    // Utilitaire: convertit une chaine hex en tableau d'octets
    private static byte[] hexStringToByteArray(String s) {
        s = s.replaceAll("\\s+", "");
        int len = s.length();
        if (len % 2 != 0) throw new IllegalArgumentException("Invalid hex string");
        byte[] data = new byte[len / 2];
        for (int i = 0; i < len; i += 2) {
            data[i / 2] = (byte) ((Character.digit(s.charAt(i), 16) << 4)
                    + Character.digit(s.charAt(i+1), 16));
        }
        return data;
    }

    // Utilitaire: affiche des octets en hex pour debug
    private static String bytesToHex(byte[] bytes) {
        if (bytes == null) return "null";
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) sb.append(String.format("%02X", b));
        return sb.toString();
    }

    /**
     * Wrapper pour retour groupé.
     */
    public static class LoadedModel {
        public MultiLayerNetwork model;
        public ScalerSet scalers;
        public double rendement;
        public int totalSerieTested;
        public int totalTrades;
        public double businnesScore;
        public double sumProfil;
        public double maxDrawdown;
        public double winRate;
        public double profitFactor;
        public double ratio;
        public LstmTuningService.TuningResult resultTuning;
        public LstmConfig config;
        public int phase;
        public LoadedModel(){}
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
        double m=0,v=0; int n=norm.length; for(double d: norm)m+=d; m = n>0? m/n:0; for(double d: norm)v += (d-m)*(d-m); v = n>0? v/n:0; double std = Math.sqrt(v);
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

    /**
     * Met à jour le buffer circulaire des deltas (absolus) utilisés pour calculer
     * un percentile adaptatif (entryThreshold). IMPORTANT:
     *  - count = nombre total d'éléments déjà insérés (peut être < buffer.length au début)
     *  - index = position courante de réécriture lorsque le buffer est plein
     *  - La logique d'incrément (count/index) est gérée dans l'appelant pour éviter
     *    toute création d'objet ou retour multiple (performance loop trading).
     *  - On stocke la valeur ABS(rawDelta) pour que le percentile reflète l'amplitude.
     *  - Si les paramètres sont incohérents, on sécurise sans throw afin de ne pas casser la simulation.
     */
    private void updateDeltaBuffer(double[] buffer, double rawDelta, int count, int index) {
        if (buffer == null || buffer.length == 0) return;
        int capacity = buffer.length;
        int pos;
        if (count < 0) count = 0; // sécurité
        if (count < capacity) {
            // Phase de remplissage linéaire
            pos = count;
        } else {
            // Phase circulaire
            if (index < 0) index = 0;
            pos = index % capacity;
        }
        if (pos < 0 || pos >= capacity) pos = capacity - 1; // garde-fou
        double v = Math.abs(rawDelta);
        if (!Double.isFinite(v)) v = 0.0;
        buffer[pos] = v;
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

    /**
     * Test de performance et d'invalidation du cache de extractFeatureMatrix.
     * Affiche les temps d'exécution et la validité du cache.
     */
    public static void testFeatureMatrixCache(BarSeries fullbars) {
        // Reset instrumentation
        LstmFeatureMatrixCache.resetStats();
        String symbol = "TESTSYM";
        List<String> features = Arrays.asList("close", "high", "low", "volume", "rsi_14", "sma_20", "realized_vol");
        LstmTradePredictor predictor = new LstmTradePredictor(null, null);
        // Construire une sous-série initiale (au moins 501 barres dans fullbars attendu)
        int initialBars = Math.min(500, fullbars.getBarCount() - 2); // garde 1+ barres pour ajout
        BarSeries series = new BaseBarSeriesBuilder().withName(symbol).build();
        for (int i = 0; i < initialBars; i++) series.addBar(fullbars.getBar(i));

        // Helpers checksum
        java.util.function.ToDoubleFunction<double[][]> checksum = (m) -> {
            double s = 0; if (m==null) return s; for (double[] row: m) for (double v: row) s += v; return s; };

        // 1) Premier appel (MISS attendu)
        long t0 = System.nanoTime();
        double[][] m1 = predictor.extractFeatureMatrix(series, features);
        long t1 = System.nanoTime();
        LstmFeatureMatrixCache.CacheStats s1 = LstmFeatureMatrixCache.getStats();
        double dt1 = (t1 - t0)/1e6;
        System.out.println("[CACHE-TEST] 1er appel: ms="+String.format(java.util.Locale.US, "%.3f", dt1)+" stats="+s1);
        if (s1.misses()!=1 || s1.hits()!=0) System.out.println("[CACHE-TEST][WARN] attendu misses=1 hits=0");

        // 2) Second appel (HIT attendu)
        long t2a = System.nanoTime();
        double[][] m2 = predictor.extractFeatureMatrix(series, features);
        long t2b = System.nanoTime();
        LstmFeatureMatrixCache.CacheStats s2 = LstmFeatureMatrixCache.getStats();
        double dt2 = (t2b - t2a)/1e6;
        System.out.println("[CACHE-TEST] 2e appel: ms="+String.format(java.util.Locale.US, "%.3f", dt2)+" stats="+s2);
        if (s2.hits()!=1 || s2.misses()!=1) System.out.println("[CACHE-TEST][WARN] attendu hits=1 misses=1");

        // Validations contenu identique (checksum + dimensions)
        if (m1.length!=m2.length || m1[0].length!=m2[0].length) System.out.println("[CACHE-TEST][ERR] Dimensions diff m1/m2");
        double cks1 = checksum.applyAsDouble(m1);
        double cks2 = checksum.applyAsDouble(m2);
        if (Math.abs(cks1-cks2) > 1e-9) System.out.println("[CACHE-TEST][ERR] checksum diff entre m1 et m2 => cache incohérent");

        // 3) Ajout d'une nouvelle barre (doit invalider: clé barCount & lastBarEndTime changent)
        if (initialBars+1 < fullbars.getBarCount()) {
            series.addBar(fullbars.getBar(initialBars+1));
        } else {
            System.out.println("[CACHE-TEST][WARN] Impossible d'ajouter une barre (fullbars insuffisant)");
        }
        long t3a = System.nanoTime();
        double[][] m3 = predictor.extractFeatureMatrix(series, features);
        long t3b = System.nanoTime();
        LstmFeatureMatrixCache.CacheStats s3 = LstmFeatureMatrixCache.getStats();
        double dt3 = (t3b - t3a)/1e6;
        System.out.println("[CACHE-TEST] 3e appel après ajout barre: ms="+String.format(java.util.Locale.US, "%.3f", dt3)+" stats="+s3);
        boolean invalidationOk = (s3.misses()==2 && s3.hits()==1);
        if (!invalidationOk) System.out.println("[CACHE-TEST][ERR] Invalidation non détectée (attendu misses=2 hits=1)");
        if (m3.length != m2.length + 1) System.out.println("[CACHE-TEST][ERR] m3.length="+m3.length+" attendu="+(m2.length+1));
        double cks3 = checksum.applyAsDouble(m3);
        if (Math.abs(cks3-cks2) < 1e-9) System.out.println("[CACHE-TEST][WARN] checksum m3 ~= m2 (valeurs peu changées ou cache mal invalidé)");

        // 4) Quatrième appel (HIT attendu sur nouvelle clé)
        long t4a = System.nanoTime();
        double[][] m4 = predictor.extractFeatureMatrix(series, features);
        long t4b = System.nanoTime();
        LstmFeatureMatrixCache.CacheStats s4 = LstmFeatureMatrixCache.getStats();
        double dt4 = (t4b - t4a)/1e6;
        System.out.println("[CACHE-TEST] 4e appel: ms="+String.format(java.util.Locale.US, "%.3f", dt4)+" stats="+s4);
        if (s4.hits()!=2 || s4.misses()!=2) System.out.println("[CACHE-TEST][ERR] attendu hits=2 misses=2");
        double cks4 = checksum.applyAsDouble(m4);
        if (Math.abs(cks4-cks3) > 1e-9) System.out.println("[CACHE-TEST][ERR] m4 diff de m3 (cache hit attendu)");

        // Ratios de temps (indicatif)
        System.out.println("[CACHE-TEST][SUMMARY] t1(ms)="+String.format(java.util.Locale.US, "%.2f", dt1)+
                " t2="+String.format(java.util.Locale.US, "%.2f", dt2)+
                " t3="+String.format(java.util.Locale.US, "%.2f", dt3)+
                " t4="+String.format(java.util.Locale.US, "%.2f", dt4)+
                " | invalidationOK="+invalidationOk+
                " | checksums c1="+String.format(java.util.Locale.US, "%.4f", cks1)+
                " c3="+String.format(java.util.Locale.US, "%.4f", cks3));
    }


    // ==== NOUVELLE PREDICTION STYLE TRADING (sans modification des méthodes existantes) ====
    /**
     * Prédiction avancée mimant la logique d'entrée de simulateTradingWalkForward pour un snapshot.
     * Fournit: prix prédit, tendance (UP/DOWN/STABLE) et action (BUY/HOLD/SELL) + métriques décision.
     * Entrée LONG only; SELL correspond à signal de faiblesse/éviter achat (ou sortie potentielle si en position externe).
     */
    public TradeStylePrediction predictTradeStyle(String symbol,
                                                  BarSeries series,
                                                  LstmConfig config,
                                                  MultiLayerNetwork model,
                                                  ScalerSet scalers) {

        TradeStylePrediction out = new TradeStylePrediction();
        try {
            int barCount = series.getBarCount();
            int window = config.getWindowSize();
            if (barCount <= window + 2) {
                double lc = series.getLastBar().getClosePrice().doubleValue();
                out.lastClose = lc; out.predictedClose = lc; out.tendance = "STABLE"; out.action = "HOLD"; out.comment = "insuffisant"; return out;
            }
            // Assurer modèle & scalers
            boolean rebuiltModel = false;
            boolean rebuiltScalers = false;
            if (model == null) {
                rebuiltModel = true;
                logger.warn("[PREDICT-TRADE][WARN] Modèle null, initialisation par défaut (non entraîné)");
            }
            model = ensureModelWindowSize(model, config.getFeatures().size(), config);
            if (scalers == null || scalers.featureScalers == null || scalers.featureScalers.size() != config.getFeatures().size() || scalers.labelScaler == null) {
                rebuiltScalers = true;
                logger.warn("[PREDICT-TRADE][WARN] Scalers null/incomplets, reconstruction sur série complète (risque de prédiction collée au dernier close)");
                scalers = rebuildScalers(series, config);
            }
            if (rebuiltModel || rebuiltScalers) {
                out.comment = "Modèle ou scalers reconstruits, prédiction peu fiable (souvent égale au dernier close)";
            }
            // Prix actuel
            double lastClose = series.getLastBar().getClosePrice().doubleValue();
            out.lastClose = lastClose;
            // Prédiction principale
            java.util.Random rand = new java.util.Random();
            double predicted = predictNextCloseScalarFast(series, config, model, scalers);
            out.predictedClose = predicted;
            double rawDeltaPct = (lastClose > 0) ? (predicted - lastClose) / lastClose : 0.0;

            // Correction : gestion NaN ou modèle non entraîné pour cas contrarien
            if (!Double.isFinite(predicted)) {
                // Vérifie si la série est baissière
                boolean isDownSeries = true;
                for (int i = 1; i < series.getBarCount(); i++) {
                    double prev = series.getBar(i-1).getClosePrice().doubleValue();
                    double curr = series.getBar(i).getClosePrice().doubleValue();
                    if (curr >= prev) { isDownSeries = false; break; }
                }
                if (isDownSeries) {
                    out.tendance = "DOWN";
                    out.contrarianAdjusted = true;
                    out.contrarianReason = "Série baissière, modèle non fiable";
                    out.comment = (out.comment == null ? "" : out.comment + "; ") + "Prédiction NaN, série baissière => contrarian DOWN";
                }
            }

            // ATR & RSI & Volume (comme simulation)
            double atrBar = 0, atrPct = 0; double rsiVal = 50; double volRatio = 1.0;
            try {
                ATRIndicator atrInd = new ATRIndicator(series, 14);
                atrBar = atrInd.getValue(series.getEndIndex()).doubleValue();
                if (lastClose > 0) atrPct = atrBar / lastClose;
            } catch (Exception e) { logger.debug("[PREDIT][ATR] skip {}", e.toString()); }

            // 2. Momentum RSI pour filtrer sur-achat / sur-vente
            try {
                RSIIndicator rsi = new RSIIndicator(new ClosePriceIndicator(series), 14);
                rsiVal = rsi.getValue(series.getEndIndex()).doubleValue();
            } catch (Exception e) { logger.debug("[PREDIT][RSI] skip {}", e.toString()); }

            // 3. Volume relatif (logique simple: volume courant vs moyenne 20)
            try {
                VolumeIndicator volInd = new VolumeIndicator(series);
                double curVol = volInd.getValue(series.getEndIndex()).doubleValue();
                double sumVol = 0.0; int count = 0;
                for (int i = Math.max(0, series.getEndIndex() - 19); i <= series.getEndIndex(); i++) {
                    sumVol += volInd.getValue(i).doubleValue(); count++;
                }
                double avgVol = count > 0 ? sumVol / count : curVol;
                if (avgVol > 0) volRatio = curVol / avgVol;
            } catch (Exception e) { logger.debug("[PREDIT][VOL] skip {}", e.toString()); }

        // Seuil ATR adaptatif (min/max)
        double cfgMinTh = config.getThresholdAtrMin(); if (!(cfgMinTh > 0)) cfgMinTh = 0.001; if (cfgMinTh < 1e-4) cfgMinTh = 1e-4;
        double cfgMaxTh = config.getThresholdAtrMax(); if (!(cfgMaxTh > cfgMinTh)) cfgMaxTh = Math.max(cfgMinTh*2, 0.01);
        double atrAdaptiveThreshold = Math.min(cfgMaxTh, Math.max(cfgMinTh, atrPct));

        // Paramètres d'entrée avancés (mêmes noms que simulation)
        double basePercentileQ = Math.min(0.95, Math.max(0.50, config.getEntryPercentileQuantile()));
        double deltaFloor = Math.max(0.0002, config.getEntryDeltaFloor());
        boolean orLogic = config.isEntryOrLogic();
        double aggressivenessBoost = config.getAggressivenessBoost() > 0 ? config.getAggressivenessBoost() : 1.0;

        // Distribution historique des deltas prédits (approx) sur N dernières barres pour percentile
        int lookbackForPercentile = Math.min(100, barCount - window - 2); // réduit à 100 pour accélérer
        List<Double> absDeltas = new ArrayList<>();
        /*if (lookbackForPercentile > 20) {
                Collections.sort(absDeltas);
                int start = barCount - lookbackForPercentile - 1;
                double[] closes = extractCloseValues(series);
                // Échantillonnage : une barre sur deux
                for (int b = start + window; b < barCount - 1; b += 2) { // step=2 pour accélérer
                    try {
                        BarSeries sub = series.getSubSeries(0, b + 1);
                        double prevClose = closes[b];
                        double predB = predictNextCloseScalarFast(sub, config, model, scalers);
                        if (prevClose > 0 && Double.isFinite(predB)) {
                            double d = Math.abs((predB - prevClose) / prevClose);
                            if (Double.isFinite(d)) absDeltas.add(d);
                        }
                    } catch (Exception ignore) { }
                }
            }*/
            double entryPercentileThreshold;
            if (absDeltas.size() >= 32) {
                int idx = (int)Math.floor(basePercentileQ * (absDeltas.size()-1));
                if (idx < 0) idx = 0; if (idx >= absDeltas.size()) idx = absDeltas.size()-1;
                entryPercentileThreshold = Math.max(absDeltas.get(idx), deltaFloor);
            } else {
                entryPercentileThreshold = deltaFloor;
            }
            out.percentileThreshold = entryPercentileThreshold;

            // Signal strength + contrarian éventuel (logique proche simulation)
            double signalStrength = rawDeltaPct * aggressivenessBoost;
            // Construire rsiValues pour appel evaluateContrarian
            double[] rsiValues = new double[barCount];
            try {
                RSIIndicator rsiAll = new RSIIndicator(new ClosePriceIndicator(series), 14);
                for (int i=0;i<barCount;i++) rsiValues[i] = rsiAll.getValue(i).doubleValue();
            } catch (Exception e){ Arrays.fill(rsiValues, rsiVal); }
            ContrarianDecision contrarian = evaluateContrarian(signalStrength, lastClose, rsiValues, barCount-1, series.getSubSeries(Math.max(0, barCount-50), barCount));
            if (contrarian.active) {
                signalStrength = contrarian.adjustedSignalStrength;
                out.contrarianAdjusted = true;
                out.contrarianReason = contrarian.reason;
            }
            out.signalStrength = signalStrength;

            // Filtres RSI/Volume (mêmes que simulation pour l'entrée)
            boolean rsiFilter = rsiVal > config.getRsiOverboughtLimit();
            boolean volumeFilter = !(volRatio >= config.getVolumeMinRatio());

            boolean enter = !rsiFilter && !volumeFilter && (
                    orLogic ? (signalStrength > entryPercentileThreshold || signalStrength > atrAdaptiveThreshold)
                            : (signalStrength > entryPercentileThreshold && signalStrength > atrAdaptiveThreshold)
            );

            // Tendance brute (au-dessus / en-dessous seuil principal)
            String tendance;
            if (rawDeltaPct > atrAdaptiveThreshold) tendance = "UP"; else if (rawDeltaPct < -atrAdaptiveThreshold) tendance = "DOWN"; else tendance = "STABLE";
            out.tendance = tendance;

            // Action décisionnelle
            String action;
            double sellThreshold = Math.max(0.001, config.getThresholdAtrMin() * 0.5);
            if (enter && signalStrength > 0) action = "BUY";
            else if (rawDeltaPct < -sellThreshold) action = "SELL"; // signal de faiblesse
            else action = "HOLD";
            out.action = action;
            logger.info("[DEBUG][TRADE] bar={} action={} rawDeltaPct={} threshold={}", barCount, action, rawDeltaPct, atrAdaptiveThreshold);

            out.deltaPct = rawDeltaPct;
            out.atrPct = atrPct;
            out.rsi = rsiVal;
            out.volumeRatio = volRatio;
            out.thresholdAtrAdaptive = atrAdaptiveThreshold;
            out.entryLogicOr = orLogic;
            out.rsiFiltered = rsiFilter;
            out.volumeFiltered = volumeFilter;
            out.aggressivenessBoost = aggressivenessBoost;
            out.symbol = symbol;
            out.windowSize = window;
            out.comment = String.format(java.util.Locale.US,
                    "rawDelta=%.4f%% sig=%.4f thrATR=%.4f thrPct=%.4f rsi=%.1f volR=%.2f enter=%s",
                    rawDeltaPct*100, signalStrength, atrAdaptiveThreshold, entryPercentileThreshold, rsiVal, volRatio, enter);

            logger.info("[PRED-TRADE] win={} last={} pred={} delta={} ({}%) thr={} ({}) signal={} conf={} rsi={} volR={} atrPct={}",
                    symbol,
                    String.format(java.util.Locale.US, "%.4f", lastClose),
                    String.format(java.util.Locale.US, "%.4f", predicted),
                    String.format(java.util.Locale.US, "%.4f", rawDeltaPct*100),
                    tendance, action,
                    String.format(java.util.Locale.US, "%.5f", signalStrength),
                    String.format(java.util.Locale.US, "%.5f", atrAdaptiveThreshold),
                    String.format(java.util.Locale.US, "%.5f", entryPercentileThreshold),
                    String.format(java.util.Locale.US, "%.1f", rsiVal),
                    String.format(java.util.Locale.US, "%.2f", volRatio),
                    String.format(java.util.Locale.US, "%.3f", atrPct * 100.0)
            );

        } catch (Exception e) {
            out.action = out.action == null? "HOLD": out.action; out.tendance = out.tendance==null?"STABLE":out.tendance; out.comment = "error:"+e.getMessage();
            logger.warn("[PRED-TRADE][ERR] {}", e.toString());
        }
        return out;
    }

    /**
     * Version optimisée/minimale : retourne juste le prix prédit sans limitation, logs, ni ajustements.
     * Prérequis : model, config et scalers sont valides et cohérents.
     */
    public double predictNextCloseScalarFast(BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers) {
        List<String> features = config.getFeatures();
        int windowSize = config.getWindowSize();
        if (series.getBarCount() <= windowSize) {
            throw new IllegalArgumentException("Pas assez de barres: " + series.getBarCount() + " <= " + windowSize);
        }
        double[][] matrix = extractFeatureMatrix(series, features);
        int numFeatures = features.size();
        double[][] normMatrix = new double[matrix.length][numFeatures];
        for (int f = 0; f < numFeatures; f++) {
            double[] col = new double[matrix.length];
            for (int i = 0; i < matrix.length; i++) {
                col[i] = matrix[i][f];
            }
            double[] normCol = scalers.featureScalers.get(features.get(f)).transform(col);
            for (int i = 0; i < matrix.length; i++) {
                normMatrix[i][f] = normCol[i];
            }
        }
        double[][][] seq = new double[1][windowSize][numFeatures];
        for (int j = 0; j < windowSize; j++) {
            System.arraycopy(normMatrix[normMatrix.length - windowSize + j], 0, seq[0][j], 0, numFeatures);
        }
        org.nd4j.linalg.api.ndarray.INDArray input = Nd4j.create(seq).permute(0, 2, 1).dup('c');
        double predNorm = model.output(input).getDouble(0);
        double predTarget = scalers.labelScaler.inverse(predNorm);
        double[] closes = extractCloseValues(series);
        double referencePrice = closes[closes.length - 1];
        if (config.isUseLogReturnTarget()) {
            return referencePrice * Math.exp(predTarget);
        } else {
            return predTarget;
        }
    }


}
