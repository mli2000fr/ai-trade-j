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
import org.deeplearning4j.nn.conf.inputs.InputType;

@Service
public class LstmTradePredictor {

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

        // Sélection dynamique de l'updater (optimiseur) selon chaîne.
        builder.updater(
            "adam".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.Adam(learningRate)
                : "rmsprop".equalsIgnoreCase(optimizer) ? new org.nd4j.linalg.learning.config.RmsProp(learningRate)
                : new org.nd4j.linalg.learning.config.Sgd(learningRate)
        );

        // Régularisations
        builder.l1(l1).l2(l2);

        // Activation des workspaces mémoire (optimisation Dl4J)
        builder.trainingWorkspaceMode(WorkspaceMode.ENABLED)
               .inferenceWorkspaceMode(WorkspaceMode.ENABLED);

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();

        // Paramètres dynamiques
        int nLayers = config != null ? config.getNumLstmLayers() : 1;
        boolean bidir = config != null && config.isBidirectional();
        boolean attention = config != null && config.isAttention();

        // Empilement des couches LSTM
        for (int i = 0; i < nLayers; i++) {
            int inSize = (i == 0) ? inputSize : (bidir ? lstmNeurons * 2 : lstmNeurons);

            // Construction d'une couche LSTM basique
            LSTM.Builder lstmBuilder = new LSTM.Builder()
                .nIn(inSize)
                .nOut(lstmNeurons)
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
                // Dropout appliqué seulement entre couches (pas sur la dernière récurrente)
                if (dropoutRate > 0.0) {
                    listBuilder.layer(new DropoutLayer.Builder()
                        .dropOut(dropoutRate)
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
        int denseOut = Math.max(16, lstmNeurons / 4);

        // Couche Dense de projection vers une dimension plus compacte
        listBuilder.layer(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
            .nIn(finalRecurrentSize)
            .nOut(denseOut)
            .activation(Activation.RELU)
            .build());

        // Sélection dynamique de la fonction de perte / activation finale
        Activation outAct;
        LossFunctions.LossFunction outLoss;
        if (outputSize == 1 && !classification) {
            outAct = Activation.IDENTITY;
            outLoss = LossFunctions.LossFunction.MSE;
        } else if (classification) {
            outAct = Activation.SOFTMAX;
            outLoss = LossFunctions.LossFunction.MCXENT;
        } else {
            outAct = Activation.IDENTITY;
            outLoss = LossFunctions.LossFunction.MSE;
        }

        listBuilder.layer(new OutputLayer.Builder(outLoss)
            .nIn(denseOut)
            .nOut(outputSize)
            .activation(outAct)
            .build());

        // IMPORTANT: On force explicitement le type d'entrée
        // Ici on travaille conceptuellement avec du recurrent(inputSize)
        listBuilder.setInputType(InputType.recurrent(inputSize));

        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
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
     * Extrait une matrice de features depuis une série TA4J.
     * Chaque feature est indexée dans l'ordre fourni par la liste features.
     *
     * @param series   Série temporelle de barres (TA4J)
     * @param features Liste ordonnée des features à extraire (ex: ["close","rsi","sma"])
     * @return Matrice double[barCount][featuresCount]
     */
    public double[][] extractFeatureMatrix(BarSeries series, List<String> features) {
        int n = series.getBarCount();
        int fCount = features.size();
        double[][] M = new double[n][fCount];
        if (n == 0) return M;

        // Indicateurs de base
        ClosePriceIndicator close = new ClosePriceIndicator(series);
        HighPriceIndicator high = new HighPriceIndicator(series);
        LowPriceIndicator low = new LowPriceIndicator(series);
        VolumeIndicator vol = new VolumeIndicator(series);

        // Instanciations conditionnelles (calculées seulement si nécessaires)
        RSIIndicator rsi = features.contains("rsi") ? new RSIIndicator(close, 14) : null;
        SMAIndicator sma14 = features.contains("sma") ? new SMAIndicator(close, 14) : null;
        EMAIndicator ema14 = features.contains("ema") ? new EMAIndicator(close, 14) : null;
        MACDIndicator macd = features.contains("macd") ? new MACDIndicator(close, 12, 26) : null;
        ATRIndicator atr = features.contains("atr") ? new ATRIndicator(series, 14) : null;
        StochasticOscillatorKIndicator stoch = features.contains("stochastic") ? new StochasticOscillatorKIndicator(series, 14) : null;
        CCIIndicator cci = features.contains("cci") ? new CCIIndicator(series, 20) : null;
        StandardDeviationIndicator sd20 =
            (features.contains("bollinger_high") || features.contains("bollinger_low"))
                ? new StandardDeviationIndicator(close, 20) : null;
        SMAIndicator sma20 =
            (features.contains("bollinger_high") || features.contains("bollinger_low"))
                ? new SMAIndicator(close, 20) : null;

        boolean needMomentum = features.contains("momentum");

        // Parcours de chaque barre
        for (int i = 0; i < n; i++) {
            double closeVal = close.getValue(i).doubleValue();
            double highVal = high.getValue(i).doubleValue();
            double lowVal = low.getValue(i).doubleValue();
            ZonedDateTime t = series.getBar(i).getEndTime();

            for (int f = 0; f < fCount; f++) {
                String feat = features.get(f);
                double val = 0.0;

                // Switch exhaustif : toute feature inconnue => fallback sur close (comportement actuel conservé)
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
                    case "momentum" -> val =
                        (needMomentum && i >= 10)
                            ? (close.getValue(i).doubleValue() - close.getValue(i - 10).doubleValue())
                            : 0.0;
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
                    case "day_of_week" -> val = t.getDayOfWeek().getValue(); // 1=Lundi .. 7=Dimanche
                    case "month" -> val = t.getMonthValue();
                    default -> val = closeVal; // fallback inert sans changer logique existante
                }

                // Nettoyage valeurs invalides
                if (Double.isNaN(val) || Double.isInfinite(val)) val = 0.0;
                M[i][f] = val;
            }
        }
        return M;
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
        public TrainResult(MultiLayerNetwork m, ScalerSet s){this.model=m;this.scalers=s;}
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
            case "rsi", "momentum", "stochastic", "cci", "macd" -> "zscore";
            default -> "minmax";
        };
    }

    /**
     * Entraîne un modèle LSTM pour prédire la prochaine clôture (ou log-return).
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
     * @param series Série d'entraînement
     * @param config Configuration LSTM
     * @param unused Paramètre non utilisé (maintenu pour compat compat)
     * @return Résultat contenant modèle + scalers
     */
    public TrainResult trainLstmScalarV2(BarSeries series, LstmConfig config, Object unused) {
        List<String> features = config.getFeatures();
        if (features == null || features.isEmpty()) {
            logger.error("[TRAIN] Liste de features vide/null -> fallback ['close']");
            features = java.util.List.of("close");
            config.setFeatures(new java.util.ArrayList<>(features));
        }

        int windowSize = config.getWindowSize();
        int numFeatures = features.size();
        if (numFeatures <= 0) {
            throw new IllegalStateException("numFeatures=0 après fallback, config=" + config.getWindowSize());
        }

        int barCount = series.getBarCount();
        // Pas assez de données pour constituer au moins une séquence complète
        if (barCount <= windowSize + 1) return new TrainResult(null, null);

        // 1) Matrice de features + vecteur des prix
        double[][] matrix = extractFeatureMatrix(series, features);
        double[] closes = extractCloseValues(series);

        // 2) Construction des séquences d'entrée (sliding window)
        int numSeq = barCount - windowSize - 1;
        double[][][] inputSeq = new double[numSeq][windowSize][numFeatures];
        double[] labelSeq = new double[numSeq];

        for (int i = 0; i < numSeq; i++) {
            for (int j = 0; j < windowSize; j++) {
                System.arraycopy(matrix[i + j], 0, inputSeq[i][j], 0, numFeatures);
            }
            // Label: soit log-return soit prix direct (selon config)
            if (config.isUseLogReturnTarget()) {
                double prev = closes[i + windowSize - 1];
                double next = closes[i + windowSize];
                labelSeq[i] = Math.log(next / prev);
            } else {
                labelSeq[i] = closes[i + windowSize];
            }
        }

        // 3) Apprentissage des scalers
        ScalerSet scalers = new ScalerSet();
        for (int f = 0; f < numFeatures; f++) {
            double[] col = new double[numSeq + windowSize];
            for (int i = 0; i < numSeq + windowSize; i++) col[i] = matrix[i][f];
            FeatureScaler.Type type =
                getFeatureNormalizationType(features.get(f)).equals("zscore")
                    ? FeatureScaler.Type.ZSCORE
                    : FeatureScaler.Type.MINMAX;
            FeatureScaler scaler = new FeatureScaler(type);
            scaler.fit(col);
            scalers.featureScalers.put(features.get(f), scaler);
        }
        FeatureScaler labelScaler = new FeatureScaler(FeatureScaler.Type.MINMAX);
        labelScaler.fit(labelSeq);
        scalers.labelScaler = labelScaler;

        // 4) Normalisation des séquences
        double[][][] normSeq = new double[numSeq][windowSize][numFeatures];
        for (int i = 0; i < numSeq; i++) {
            for (int j = 0; j < windowSize; j++) {
                for (int f = 0; f < numFeatures; f++) {
                    normSeq[i][j][f] =
                        scalers.featureScalers
                            .get(features.get(f))
                            .transform(new double[]{inputSeq[i][j][f]})[0];
                }
            }
        }

        // 5) Conversion en INDArray + permutation: [batch, time, features] -> [batch, features, time]
        org.nd4j.linalg.api.ndarray.INDArray X = Nd4j.create(normSeq);
        X = X.permute(0, 2, 1).dup('c');

        if (X.size(1) != numFeatures || X.size(2) != windowSize) {
            logger.warn("[SHAPE][TRAIN] Incohérence shape après permute: expected features={} time={} got features={} time={}",
                numFeatures, windowSize, X.size(1), X.size(2));
        }

        double[] normLabels = scalers.labelScaler.transform(labelSeq);
        org.nd4j.linalg.api.ndarray.INDArray y = Nd4j.create(normLabels, new long[]{numSeq, 1});

        // 6) Initialisation modèle
        int effectiveFeatures = (int) X.size(1);
        if (effectiveFeatures != numFeatures) {
            logger.warn("[INIT][ADAPT] numFeatures déclaré={} mais tensor features={} => reconstruction modèle",
                numFeatures, effectiveFeatures);
        }

        MultiLayerNetwork model = initModel(
            effectiveFeatures,
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

        logger.debug("[TRAIN] X shape={} (batch={} features={} time={}) y shape={} expectedFeatures={} lstmNeurons={}",
            Arrays.toString(X.shape()), X.size(0), X.size(1), X.size(2),
            Arrays.toString(y.shape()), numFeatures, config.getLstmNeurons());

        // 7) Création DataSetIterator (batching)
        org.nd4j.linalg.dataset.DataSet ds = new org.nd4j.linalg.dataset.DataSet(X, y);
        org.nd4j.linalg.dataset.api.iterator.DataSetIterator iterator =
            new ListDataSetIterator<>(ds.asList(), config.getBatchSize());

        // 8) Boucle d'entraînement (simple, sans early stopping)
        int epochs = config.getNumEpochs();
        long t0 = System.currentTimeMillis();
        double bestScore = Double.POSITIVE_INFINITY;

        for (int epoch = 1; epoch <= epochs; epoch++) {
            iterator.reset();
            model.fit(iterator);
            double score = model.score();
            if (score < bestScore) bestScore = score;

            // Logging périodique + mesure mémoire
            if (epoch == 1 || epoch == epochs || epoch % Math.max(1, epochs / 10) == 0) {
                Runtime rt = Runtime.getRuntime();
                long used = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024);
                long max = rt.maxMemory() / (1024 * 1024);
                long elapsed = System.currentTimeMillis() - t0;
                logger.info("[TRAIN][EPOCH] {}/{} score={} best={} elapsedMs={} memUsedMB={}/{}",
                    epoch,
                    epochs,
                    String.format("%.6f", score),
                    String.format("%.6f", bestScore),
                    elapsed,
                    used,
                    max
                );
            }
        }

        return new TrainResult(model, scalers);
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
     * Prédit la prochaine clôture (ou log-return reconverti en prix).
     *
     * @param series  Série complète (la dernière barre = point courant)
     * @param config  Configuration (fenêtre, normalisation, etc.)
     * @param model   Modèle LSTM déjà entraîné
     * @param scalers Scalers adaptés au modèle
     * @return Prix prédit (ajusté).
     */
    public double predictNextCloseScalarV2(BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers) {
        List<String> features = config.getFeatures();
        int windowSize = config.getWindowSize();

        if (series.getBarCount() <= windowSize)
            throw new IllegalArgumentException("Pas assez de barres");

        // Reconstruction des scalers si incohérents (sécurité)
        if (scalers == null
            || scalers.featureScalers == null
            || scalers.featureScalers.size() != features.size()
            || scalers.labelScaler == null) {
            logger.warn("[PREDICT] Scalers null/incomplets -> rebuild");
            scalers = rebuildScalers(series, config);
        }

        double[][] matrix = extractFeatureMatrix(series, features);
        int numFeatures = features.size();

        // Normalisation colonne par colonne
        double[][] normMatrix = new double[matrix.length][numFeatures];
        for (int f = 0; f < numFeatures; f++) {
            double[] col = new double[matrix.length];
            for (int i = 0; i < matrix.length; i++) col[i] = matrix[i][f];
            double[] normCol = scalers.featureScalers.get(features.get(f)).transform(col);
            for (int i = 0; i < matrix.length; i++) normMatrix[i][f] = normCol[i];
        }

        // Construction dernière séquence
        double[][][] seq = new double[1][windowSize][numFeatures];
        for (int j = 0; j < windowSize; j++)
            System.arraycopy(normMatrix[normMatrix.length - windowSize + j], 0, seq[0][j], 0, numFeatures);

        // Permutation vers [1, features, time]
        org.nd4j.linalg.api.ndarray.INDArray input =
            Nd4j.create(seq).permute(0, 2, 1).dup('c');

        if (input.size(1) != numFeatures || input.size(2) != windowSize) {
            logger.warn("[SHAPE][PRED] Incohérence shape input: expected features={} time={} got features={} time={}",
                numFeatures, windowSize, input.size(1), input.size(2));
        }

        // Forward pass
        double predNorm = model.output(input).getDouble(0);

        // Inversion normalisation
        double predTarget = scalers.labelScaler.inverse(predNorm);

        double lastClose = series.getLastBar().getClosePrice().doubleValue();

        // Si on a entraîné sur log-return => reconversion en prix
        double predicted = config.isUseLogReturnTarget()
            ? lastClose * Math.exp(predTarget)
            : predTarget;

        // Limitation relative (anti explosion)
        double limitPct = config.getLimitPredictionPct();
        if (limitPct > 0) {
            double min = lastClose * (1 - limitPct);
            double max = lastClose * (1 + limitPct);
            if (predicted < min) predicted = min;
            else if (predicted > max) predicted = max;
        }
        return predicted;
    }

    /**
     * Alias conservant signature existante.
     */
    public double predictNextCloseWithScalerSet(String symbol, BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers){
        return predictNextCloseScalarV2(series, config, model, scalers);
    }

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
     * Contient les métriques de trading sur un split (simulation).
     * Les champs sont agrégés plus tard.
     */
    public static class TradingMetricsV2 implements Serializable {
        public double totalProfit, profitFactor, winRate, maxDrawdownPct, expectancy, sharpe, sortino, exposure, turnover, avgBarsInPosition, mse, businessScore, calmar;
        public int numTrades;
    }

    /**
     * Résultat global d'un walk-forward avec statistiques agrégées.
     */
    public static class WalkForwardResultV2 implements Serializable {
        public List<TradingMetricsV2> splits = new ArrayList<>();
        public double meanMse, meanBusinessScore, mseVariance, mseInterModelVariance;
    }

    /**
     * Exécute une validation walk-forward (multi-splits).
     * @param series Série complète
     * @param config Config LSTM
     * @return Résultat avec métriques par split
     */
    public WalkForwardResultV2 walkForwardEvaluate(BarSeries series, LstmConfig config) {
        WalkForwardResultV2 result = new WalkForwardResultV2();
        int splits = Math.max(2, config.getWalkForwardSplits());
        int windowSize = config.getWindowSize();
        int totalBars = series.getBarCount();

        if (totalBars < windowSize + 50) return result;

        int splitSize = (totalBars - windowSize) / splits;
        double sumMse = 0, sumBusiness = 0;
        int mseCount = 0, businessCount = 0;
        List<Double> mseList = new ArrayList<>();

        for (int s = 1; s <= splits; s++) {
            int testEndBar = (s == splits) ? totalBars : windowSize + s * splitSize;
            int testStartBar = windowSize + (s - 1) * splitSize + config.getEmbargoBars();

            // Vérifie qu'il y a assez de barres pour test
            if (testStartBar + windowSize + 5 >= testEndBar) continue;

            BarSeries trainSeries = series.getSubSeries(0, testStartBar);

            // Entraînement sur le segment passé
            TrainResult tr = trainLstmScalarV2(trainSeries, config, null);
            if (tr.model == null) continue;

            // Simulation + MSE
            TradingMetricsV2 metrics = simulateTradingWalkForward(
                series,
                trainSeries.getBarCount(),
                testStartBar,
                testEndBar,
                tr.model,
                tr.scalers,
                config
            );
            metrics.mse = computeSplitMse(series, testStartBar, testEndBar, tr.model, tr.scalers, config);

            // Calcul d'un score "business" (pondération heuristique)
            double pfAdj = Math.min(metrics.profitFactor, config.getBusinessProfitFactorCap());
            double expPos = Math.max(metrics.expectancy, 0.0);
            double denom = 1.0 + Math.pow(Math.max(metrics.maxDrawdownPct, 0.0), config.getBusinessDrawdownGamma());
            metrics.businessScore = (expPos * pfAdj * metrics.winRate) / (denom + 1e-9);

            result.splits.add(metrics);

            if (Double.isFinite(metrics.mse)) {
                sumMse += metrics.mse;
                mseCount++;
                mseList.add(metrics.mse);
            }
            if (Double.isFinite(metrics.businessScore)) {
                sumBusiness += metrics.businessScore;
                businessCount++;
            }
        }

        result.meanMse = mseCount > 0 ? sumMse / mseCount : Double.NaN;
        result.meanBusinessScore = businessCount > 0 ? sumBusiness / businessCount : Double.NaN;

        if (mseList.size() > 1) {
            double mean = result.meanMse;
            double var = mseList.stream().mapToDouble(m -> (m - mean) * (m - mean)).sum() / mseList.size();
            result.mseVariance = var;
            result.mseInterModelVariance = var; // alias simple
        }
        return result;
    }

    /**
     * Simule une stratégie basique "swing" sur un intervalle de test
     * en utilisant les prédictions successives du modèle.
     *
     * @return TradingMetricsV2 métriques remplies.
     */
    public TradingMetricsV2 simulateTradingWalkForward(
        BarSeries fullSeries,
        int trainBarCount,
        int testStartBar,
        int testEndBar,
        MultiLayerNetwork model,
        ScalerSet scalers,
        LstmConfig config
    ) {
        TradingMetricsV2 tm = new TradingMetricsV2();

        double equity = 0, peak = 0, trough = 0;
        boolean inPos = false, longPos = false;
        double entry = 0;
        int barsInPos = 0;
        int horizon = Math.max(3, config.getHorizonBars());

        List<Double> tradePnL = new ArrayList<>();
        List<Double> tradeReturns = new ArrayList<>();
        List<Integer> barsInPosList = new ArrayList<>();

        double timeInPos = 0;
        int positionChanges = 0;
        double capital = config.getCapital();
        double riskPct = config.getRiskPct();
        double sizingK = config.getSizingK();
        double feePct = config.getFeePct();
        double slippagePct = config.getSlippagePct();
        double positionSize = 0;
        double entrySpread = 0;

        // Boucle sur les barres de test
        for (int bar = testStartBar; bar < testEndBar - 1; bar++) {
            BarSeries sub = fullSeries.getSubSeries(0, bar + 1);
            if (sub.getBarCount() <= config.getWindowSize()) continue;

            double threshold = computeSwingTradeThreshold(sub, config);
            ATRIndicator atrInd = new ATRIndicator(sub, 14);
            double atr = atrInd.getValue(sub.getEndIndex()).doubleValue();

            if (!inPos) {
                double predicted = predictNextCloseScalarV2(sub, config, model, scalers);
                double lastClose = sub.getLastBar().getClosePrice().doubleValue();

                double up = lastClose * (1 + threshold);
                double down = lastClose * (1 - threshold);

                // Signal d'entrée
                if (predicted > up || predicted < down) {
                    inPos = true;
                    longPos = predicted > up;
                    entry = lastClose;
                    barsInPos = 0;
                    positionChanges++;

                    // Calcul taille position (ATR risk-based)
                    positionSize = atr > 0 ? capital * riskPct / (atr * sizingK) : 0.0;
                    entrySpread = computeMeanSpread(sub);
                }
            } else {
                barsInPos++;
                timeInPos++;

                double current = fullSeries.getBar(bar).getClosePrice().doubleValue();
                double stop = entry * (1 - threshold * (longPos ? 1 : -1));
                double target = entry * (1 + 2 * threshold * (longPos ? 1 : -1));

                boolean exit = false;
                double pnl = 0;

                // Gestion long / short symétrique
                if (longPos) {
                    if (current <= stop || current >= target) {
                        pnl = (current - entry) * positionSize;
                        exit = true;
                    }
                } else {
                    if (current >= stop || current <= target) {
                        pnl = (entry - current) * positionSize;
                        exit = true;
                    }
                }

                // Sortie si horizon dépassé (prise de profit / limitation du temps)
                if (!exit && barsInPos >= horizon) {
                    pnl = longPos ? (current - entry) * positionSize : (entry - current) * positionSize;
                    exit = true;
                }

                if (exit) {
                    // Coûts de trading
                    double cost = entrySpread * positionSize
                        + slippagePct * entry * positionSize
                        + feePct * entry * positionSize;
                    pnl -= cost;

                    tradePnL.add(pnl);
                    tradeReturns.add(pnl / (entry * Math.max(positionSize, 1e-9)));
                    barsInPosList.add(barsInPos);

                    equity += pnl;
                    if (equity > peak) {
                        peak = equity;
                        trough = equity;
                    } else if (equity < trough) {
                        trough = equity;
                    }

                    inPos = false;
                    longPos = false;
                    entry = 0;
                    barsInPos = 0;
                    positionSize = 0;
                    entrySpread = 0;
                }
            }
        }

        // Agrégation métriques
        double gains = 0, losses = 0;
        int win = 0, loss = 0;
        for (double p : tradePnL) {
            if (p > 0) { gains += p; win++; }
            else if (p < 0) { losses += p; loss++; }
        }

        tm.totalProfit = gains + losses;
        tm.numTrades = tradePnL.size();
        tm.profitFactor = losses != 0 ? gains / Math.abs(losses) : (gains > 0 ? Double.POSITIVE_INFINITY : 0);
        tm.winRate = tm.numTrades > 0 ? (double) win / tm.numTrades : 0.0;

        double ddAbs = peak - trough;
        tm.maxDrawdownPct = peak != 0 ? ddAbs / Math.abs(peak) : 0.0;

        double avgGain = win > 0 ? gains / win : 0;
        double avgLoss = loss > 0 ? Math.abs(losses) / loss : 0;
        tm.expectancy = (win + loss) > 0 ? (tm.winRate * avgGain - (1 - tm.winRate) * avgLoss) : 0;

        double meanRet = tradeReturns.stream().mapToDouble(d -> d).average().orElse(0);
        double stdRet = Math.sqrt(tradeReturns.stream().mapToDouble(d -> {
            double m = d - meanRet;
            return m * m;
        }).average().orElse(0));

        tm.sharpe = stdRet > 0 ? meanRet / stdRet * Math.sqrt(Math.max(1, tradeReturns.size())) : 0;
        double downsideStd = Math.sqrt(tradeReturns.stream().filter(r -> r < 0).mapToDouble(r -> {
            double dr = r - meanRet;
            return dr * dr;
        }).average().orElse(0));
        tm.sortino = downsideStd > 0 ? meanRet / downsideStd : 0;

        tm.exposure = 0; // Non implémenté ici
        tm.turnover = 0; // Non implémenté ici
        tm.avgBarsInPosition = barsInPosList.stream().mapToInt(i -> i).average().orElse(0);

        double capitalBase = config.getCapital() > 0 ? config.getCapital() : 1.0;
        tm.calmar = tm.maxDrawdownPct > 0 ? ((tm.totalProfit / capitalBase) / tm.maxDrawdownPct) : 0.0;

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
            double actual = config.isUseLogReturnTarget()
                ? Math.log(closes[t] / closes[t - 1])
                : closes[t];
            double err = pred - actual;
            se += err * err;
            count++;
        }
        return count > 0 ? se / count : Double.NaN;
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

    /**
     * Variante simple (booléenne) de la détection + retrain.
     */
    public boolean checkDriftAndRetrain(BarSeries series, LstmConfig config, MultiLayerNetwork model, ScalerSet scalers) {
        return !checkDriftAndRetrainWithReport(series, config, model, scalers, "").isEmpty();
    }

    /* =========================================================
     *              THRESHOLD & SPREAD UTILITAIRES
     * ========================================================= */

    /**
     * Calcule un seuil de swing trading relatif basé soit sur ATR, soit sur variance des returns.
     *
     * @return seuil (en pourcentage relatif, ex: 0.01 = 1%)
     */
    public double computeSwingTradeThreshold(BarSeries series, LstmConfig config) {
        double k = config.getThresholdK();
        String type = config.getThresholdType();

        if ("ATR".equalsIgnoreCase(type)) {
            ATRIndicator atr = new ATRIndicator(series, 14);
            double lastATR = atr.getValue(series.getEndIndex()).doubleValue();
            double lastClose = series.getLastBar().getClosePrice().doubleValue();
            double th = k * lastATR / (lastClose == 0 ? 1 : lastClose);
            logger.info("[SEUIL SWING] ATR%={}", th);
            return th;
        } else if ("returns".equalsIgnoreCase(type)) {
            double[] closes = extractCloseValues(series);
            if (closes.length < 3) return 0;
            double[] logRet = new double[closes.length - 1];
            for (int i = 1; i < closes.length; i++)
                logRet[i - 1] = Math.log(closes[i] / closes[i - 1]);
            double mean = Arrays.stream(logRet).average().orElse(0);
            double std = Math.sqrt(Arrays.stream(logRet).map(r -> (r - mean) * (r - mean)).sum() / logRet.length);
            return k * std;
        }
        // Fallback simple constant
        return 0.01 * k;
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
     */
    public void saveModelToDb(String symbol, MultiLayerNetwork model, JdbcTemplate jdbcTemplate, LstmConfig config, ScalerSet scalers) throws IOException {
        if (model == null) return;
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ModelSerializer.writeModel(model, baos, true);
        byte[] modelBytes = baos.toByteArray();

        // Sauvegarde hyperparamètres via repository + JSON
        hyperparamsRepository.saveHyperparams(symbol, config);

        com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        String hyperparamsJson = mapper.writeValueAsString(config);
        String scalersJson = mapper.writeValueAsString(scalers);

        String sql = "REPLACE INTO lstm_models (symbol, model_blob, hyperparams_json, normalization_scope, scalers_json, updated_date) VALUES (?,?,?,?,?, CURRENT_TIMESTAMP)";
        jdbcTemplate.update(sql, symbol, modelBytes, hyperparamsJson, config.getNormalizationScope(), scalersJson);
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

        // Label scaler basé sur close ou log-return si demandé
        double[] closes = extractCloseValues(series);
        if (config.isUseLogReturnTarget() && closes.length > 1) {
            double[] lr = new double[closes.length - 1];
            for(int i = 1; i < closes.length; i++)
                lr[i - 1] = Math.log(closes[i] / closes[i - 1]);
            FeatureScaler lab = new FeatureScaler(FeatureScaler.Type.MINMAX);
            lab.fit(lr);
            set.labelScaler = lab;
        } else {
            FeatureScaler lab = new FeatureScaler(FeatureScaler.Type.MINMAX);
            lab.fit(closes);
            set.labelScaler = lab;
        }

        logger.warn("[SCALERS][REBUILD] Reconstruction ad-hoc des scalers (peut diverger de l'entraînement initial).");
        return set;
    }
}
