package com.app.backend.trade.controller;


import com.app.backend.trade.lstm.LstmConfig;
import com.app.backend.trade.lstm.LstmTradePredictor;
import com.app.backend.trade.lstm.LstmTuningService;
import com.app.backend.trade.model.*;
import com.app.backend.trade.util.TradeUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Controller;
import org.ta4j.core.BarSeries;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Controller
public class LstmHelper {

    /**
     * Helper / façade autour des services LSTM.
     *
     * Rôles :
     * - Récupération des données (BarSeries) depuis la base (table daily_value)
     * - Lancement d'entraînements (tuning + sauvegarde hyperparamètres)
     * - Chargement du modèle + scalers et exécution des prédictions
     * - Détection de drift + éventuel retrain automatique + reporting BD
     * - Sauvegarde de l’historique des signaux (table signal_lstm)
     *
     * Très important :
     * - NE PAS modifier la logique métier ici (risque régression)
     * - Les méthodes utilisent JdbcTemplate directement : pas de transaction explicite ici
     * - File driftReports : structure thread-safe (ConcurrentLinkedQueue) => safe pour ajouts concurrents
     *
     * Maintenance (débutants) :
     * - Lire chaque Javadoc
     * - En cas d’évolution, préférer ajouter de nouvelles méthodes plutôt que modifier celles existantes
     * - Toujours tester après changement (drift + prediction)
     */

    // -------------- Dépendances injectées --------------
    // Accès SQL simplifié
    private final JdbcTemplate jdbcTemplate;
    // Service d'inférence et gestion des modèles LSTM
    private final LstmTradePredictor lstmTradePredictor;
    // Service de génération / tuning d'hyperparamètres
    private final LstmTuningService lstmTuningService;

    // Logger standard SLF4J
    private static final Logger logger = LoggerFactory.getLogger(LstmHelper.class);

    // File concurrente en mémoire pour conserver les rapports de drift (accès thread-safe)
    private final java.util.concurrent.ConcurrentLinkedQueue<LstmTradePredictor.DriftReportEntry> driftReports =
            new java.util.concurrent.ConcurrentLinkedQueue<>();

    /**
     * Constructeur standard avec injection Spring.
     * @param jdbcTemplate accès base
     * @param lstmTradePredictor prédiction LSTM + drift
     * @param lstmTuningService tuning / hyperparamètres
     */
    public LstmHelper(JdbcTemplate jdbcTemplate,
                      LstmTradePredictor lstmTradePredictor,
                      LstmTuningService lstmTuningService) {
        this.jdbcTemplate = jdbcTemplate;
        this.lstmTradePredictor = lstmTradePredictor;
        this.lstmTuningService = lstmTuningService;
    }

    /**
     * Récupération d'une série temporelle (BarSeries) pour un symbole.
     *
     * Détails :
     * - Si limit == null : on prend tout l'historique ascendant (ORDER BY date ASC)
     * - Si limit > 0 : on prend les N dernières lignes (ORDER BY date DESC LIMIT N), puis on réinverse pour retrouver ordre chronologique
     * - Mapping des colonnes BD -> objet DailyValue -> transformation en BarSeries via TradeUtils.mapping
     *
     * ATTENTION :
     * - NE PAS changer les noms de colonnes SQL
     * - Inversion de liste OBLIGATOIRE pour conserver cohérence temporelle
     *
     * @param symbol symbole (clé)
     * @param limit nombre maximum de lignes (null = tout)
     * @return BarSeries utilisable pour entraînement/prédiction
     */
    public BarSeries getBarBySymbol(String symbol, Integer limit) {
        // Construction de la requête selon la présence de limit
        String sql = "SELECT date, open, high, low, close, volume, number_of_trades, volume_weighted_average_price " +
                "FROM daily_value WHERE symbol = ? ORDER BY date ASC";
        if (limit != null && limit > 0) {
            // IMPORTANT : on prend les dernières lignes (DESC + LIMIT), puis on inversera ensuite
            sql = "SELECT date, open, high, low, close, volume, number_of_trades, volume_weighted_average_price " +
                    "FROM daily_value WHERE symbol = ? ORDER BY date DESC LIMIT " + limit;
        }

        // Exécution + mapping ligne -> DailyValue
        List<DailyValue> results = jdbcTemplate.query(sql, new Object[]{symbol}, (rs, rowNum) -> {
            return DailyValue.builder()
                    .date(rs.getDate("date").toString())
                    .open(rs.getString("open"))
                    .high(rs.getString("high"))
                    .low(rs.getString("low"))
                    .close(rs.getString("close"))
                    .volume(rs.getString("volume"))
                    .numberOfTrades(rs.getString("number_of_trades"))
                    .volumeWeightedAveragePrice(rs.getString("volume_weighted_average_price"))
                    .build();
        });

        // Si on a utilisé l'ordre DESC (limit > 0) : on réinverse pour retrouver ordre croissant (chronologique)
        if (limit != null && limit > 0) {
            Collections.reverse(results);
        }
        return TradeUtils.mapping(results);
    }

    /**
     * Lance un entraînement (tuning) pour un symbole unique.
     *
     * Actuellement :
     * - useRandomGrid=true forcé (random search)
     * - Taille de grille 10 (hardcodée)
     *
     * NOTE :
     * - NE PAS modifier la stratégie sans validation métier
     *
     * @param symbol symbole ciblé
     */
    public void trainLstm(String symbol) {
        boolean useRandomGrid = true; // Stratégie actuelle : random search
        List<LstmConfig> grid;
        if (useRandomGrid) {
            grid = lstmTuningService.generateRandomSwingTradeGrid(10);
        } else {
            grid = lstmTuningService.generateSwingTradeGrid();
        }
        // Délègue réellement au service
        lstmTuningService.tuneAllSymbols(Arrays.asList(symbol), grid, jdbcTemplate, sym -> getBarBySymbol(sym, null));
    }

    /**
     * Récupère une prédiction LSTM pour un symbole.
     *
     * Étapes métier :
     * 1. Vérifie si une prédiction du jour existe déjà en base (cache logique) -> si oui : retour direct
     * 2. Charge la configuration (hyperparamètres) enregistrée
     * 3. Charge modèle + scalers depuis BD (best effort)
     * 4. Vérifie drift si modèle présent (peut déclencher retrain interne)
     * 5. Exécute la prédiction
     * 6. Sauvegarde le signal du jour en BD
     *
     * Points critiques :
     * - Si config == null : retourne un objet "neutre" (aucune position)
     * - Drift : insertion best-effort en table lstm_drift_report (peut ne pas exister)
     * - Retrain : si au moins un rapport indique retrained=true => on sauvegarde le modèle mis à jour
     *
     * @param symbol symbole
     * @return PreditLsdm objet contenant signal + prix prédits + métadonnées
     * @throws IOException si erreur IO interne (propagée depuis prédicteur)
     */
    public PreditLsdm getPredit(String symbol) throws IOException {
        // 1. Vérifier si une prédiction du jour existe déjà (évite recalcul)
        PreditLsdm preditLsdmDb = this.getPreditFromDB(symbol);
        if (preditLsdmDb != null) {
            return preditLsdmDb;
        }

        // 2. Charger la config (hyperparamètres) sauvegardée
        LstmConfig config = lstmTuningService.hyperparamsRepository.loadHyperparams(symbol);
        if (config == null) {
            // Log informatif + retour neutre (IMPORTANT : ne pas changer le contenu renvoyé)
            logger.info("Hyperparamètres existants trouvés pour {}. Ignorer le tuning.", symbol);
            return PreditLsdm.builder()
                    .lastClose(0)
                    .predictedClose(0)
                    .signal(SignalType.NONE)
                    .position("")
                    .lastDate("")
                    .build();
        }

        // 3. Charger modèle + scalers (si échec : on continue quand même => modèle null => prédiction fallback)
        LstmTradePredictor.LoadedModel loaded = null;
        try {
            loaded = lstmTradePredictor.loadModelAndScalersFromDb(symbol, jdbcTemplate);
        } catch (Exception e) {
            logger.warn("Impossible de charger le modèle/scalers depuis la base : {}", e.getMessage());
        }
        MultiLayerNetwork model = loaded != null ? loaded.model : null;
        LstmTradePredictor.ScalerSet scalers = loaded != null ? loaded.scalers : null;

        // 4. Charger les données prix (BarSeries complet)
        BarSeries series = getBarBySymbol(symbol, null);

        // 5. Vérification drift + éventuel retrain + reporting
        if (model != null && scalers != null) {
            try {
                java.util.List<LstmTradePredictor.DriftReportEntry> reports =
                        lstmTradePredictor.checkDriftAndRetrainWithReport(series, config, model, scalers, symbol);

                if (!reports.isEmpty()) {
                    for (LstmTradePredictor.DriftReportEntry r : reports) {
                        // Ajout mémoire
                        driftReports.add(r);

                        // Insertion BD (best-effort) - si la table n'existe pas : simple log debug
                        try {
                            String sql = "INSERT INTO lstm_drift_report (event_date, symbol, feature, drift_type, kl, mean_shift, mse_before, mse_after, retrained) VALUES (?,?,?,?,?,?,?,?,?)";
                            jdbcTemplate.update(sql,
                                    java.sql.Timestamp.from(r.eventDate),
                                    r.symbol,
                                    r.feature,
                                    r.driftType,
                                    r.kl,
                                    r.meanShift,
                                    r.mseBefore,
                                    r.mseAfter,
                                    r.retrained
                            );
                        } catch (Exception ex) {
                            logger.debug("Table lstm_drift_report absente ou insertion échouée: {}", ex.getMessage());
                        }

                        // Log détaillé (utile monitoring)
                        logger.info("[DRIFT-REPORT] symbol={} feature={} type={} kl={} meanShift={}σ mseBefore={} mseAfter={} retrained={}",
                                r.symbol, r.feature, r.driftType, r.kl, r.meanShift, r.mseBefore, r.mseAfter, r.retrained);
                    }

                    // Si au moins un retrain a eu lieu => sauvegarde du modèle
                    boolean retrained = reports.stream().anyMatch(rr -> rr.retrained);
                    if (retrained) {
                        try {
                            lstmTradePredictor.saveModelToDb(symbol, model, jdbcTemplate, config, scalers);
                        } catch (Exception e) {
                            logger.warn("Erreur lors de la sauvegarde du modèle/scalers après retrain : {}", e.getMessage());
                        }
                    }
                }
            } catch (Exception e) {
                logger.warn("Erreur pendant le drift reporting: {}", e.getMessage());
            }
        }

        // 6. Exécution de la prédiction (utilise model/scalers si présents)
        PreditLsdm preditLsdm = lstmTradePredictor.getPredit(symbol, series, config, model, scalers);

        // 7. Sauvegarde du signal du jour
        saveSignalHistory(symbol, preditLsdm);

        return preditLsdm;
    }

    /**
     * Sauvegarde le signal quotidien en base.
     *
     * IMPORTANT :
     * - Utilise la date du dernier jour de bourse (méthode utilitaire)
     * - On insère systématiquement (pas de upsert)
     *
     * @param symbol symbole
     * @param preditLsdm objet contenant signal + prix
     */
    public void saveSignalHistory(String symbol, PreditLsdm preditLsdm) {
        java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(java.time.LocalDate.now());
        String insertSql = "INSERT INTO signal_lstm (symbol, signal_lstm, price_lstm, price_clo, position_lstm, lstm_created_at) VALUES (?, ?, ?, ?, ?, ?)";
        jdbcTemplate.update(insertSql,
                symbol,
                preditLsdm.getSignal().name(),
                preditLsdm.getPredictedClose(),
                preditLsdm.getLastClose(),
                preditLsdm.getPosition(),
                java.sql.Date.valueOf(lastTradingDay));
    }

    /**
     * Récupération d'une prédiction déjà existante en base.
     *
     * Processus :
     * - Lit la dernière ligne (ORDER BY lstm_created_at DESC LIMIT 1)
     * - Vérifie si la date stockée correspond au dernier jour de cotation
     *   -> si oui : renvoie l'objet reconstitué
     *   -> sinon : retourne null (ce qui force un recalcul)
     *
     * Sécurité :
     * - Si le signal en base a une valeur non mappée => warning + type=null
     * - En cas d'exception SQL => log + null
     *
     * @param symbol symbole
     * @return PreditLsdm ou null si pas à jour
     */
    public PreditLsdm getPreditFromDB(String symbol) {
        String sql = "SELECT * FROM signal_lstm WHERE symbol = ? ORDER BY lstm_created_at DESC LIMIT 1";
        try {
            return jdbcTemplate.query(sql, ps -> ps.setString(1, symbol), rs -> {
                if (rs.next()) {
                    String signalStr = rs.getString("signal_lstm");
                    double priceLstm = rs.getDouble("price_lstm");
                    double priceClo = rs.getDouble("price_clo");
                    String positionLstm = rs.getString("position_lstm");
                    java.sql.Date lastDate = rs.getDate("lstm_created_at");

                    // Conversion robuste du type de signal
                    SignalType type;
                    try {
                        type = SignalType.valueOf(signalStr);
                    } catch (Exception e) {
                        logger.warn("SignalType inconnu en base: {}", signalStr);
                        type = null; // On garde la sémantique existante
                    }

                    java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(java.time.LocalDate.now());
                    java.time.LocalDate lastKnown = lastDate.toLocalDate();

                    // Si déjà à jour, on renvoie sinon null pour forcer recalcul
                    if (lastKnown.isEqual(lastTradingDay) || lastKnown.isAfter(lastTradingDay)) {
                        String dateSavedStr = lastKnown.format(java.time.format.DateTimeFormatter.ofPattern("dd/MM"));
                        return PreditLsdm.builder()
                                .signal(type)
                                .lastClose(priceClo)
                                .lastDate(dateSavedStr)
                                .predictedClose(priceLstm)
                                .position(positionLstm)
                                .build();
                    } else {
                        return null;
                    }
                }
                return null;
            });
        } catch (Exception e) {
            logger.warn("Erreur SQL getPreditFromDB pour {}: {}", symbol, e.getMessage());
            return null;
        }
    }

    /**
     * Récupère la liste des symboles filtrés via une table d'analyse de stratégie.
     *
     * Conditions :
     * - avg_pnl > 0
     * - profit_factor > 1
     * - win_rate > 0.5
     * - max_drawdown < 0.2
     * - sharpe_ratio > 1
     * - rendement > 0.05
     *
     * @param sort colonne de tri (fallback: score_swing_trade)
     * @return liste de symboles ordonnée
     */
    public List<String> getSymbolFitredFromTabSingle(String sort) {
        String orderBy = sort == null ? "score_swing_trade" : sort;
        String sql = "select symbol from best_in_out_single_strategy s where s.avg_pnl > 0 AND s.profit_factor > 1 AND s.win_rate > 0.5 AND s.max_drawdown < 0.2 AND s.sharpe_ratio > 1 AND s.rendement > 0.05";
        sql += " ORDER BY " + orderBy + " DESC";
        return jdbcTemplate.queryForList(sql, String.class);
    }

    // Méthode conservée (signature legacy)
    public void tuneAllSymbols() {
        tuneAllSymbols(true, 1);
    }

    /**
     * Lance le tuning sur tous les symboles filtrés.
     *
     * Paramètres testés (grille de base) :
     * - horizons : 3,5,10
     * - couches LSTM : 1,2,3
     * - batchSizes : 64,128,256
     * - bidirectionnel : true/false
     * - attention : true/false
     *
     * Mode :
     * - useRandomGrid = random search (sélection aléatoire sur grille)
     * - sinon grid exhaustive
     *
     * @param useRandomGrid active random search
     * @param randomGridSize nombre de configs tirées si random
     */
    public void tuneAllSymbols(boolean useRandomGrid, int randomGridSize) {
        List<String> symbols = getSymbolFitredFromTabSingle("score_swing_trade");

        // Valeurs testées (ne pas modifier sans validation)
        int[] horizonBars = {3, 5, 10};
        int[] numLstmLayers = {1, 2, 3};
        int[] batchSizes = {64, 128, 256};
        boolean[] bidirectionals = {true, false};
        boolean[] attentions = {true, false};

        for (String symbol : symbols) {
            // Features dynamiques selon type de symbole
            //List<String> features = getFeaturesForSymbol(symbol);
            List<LstmConfig> grid;
            if (useRandomGrid) {
                grid = lstmTuningService.generateRandomSwingTradeGrid(
                        randomGridSize, horizonBars, numLstmLayers, batchSizes, bidirectionals, attentions
                );
            } else {
                grid = lstmTuningService.generateSwingTradeGrid(horizonBars, numLstmLayers, batchSizes, bidirectionals, attentions
                );
            }
            // Appel pour ce symbole uniquement (singleton list)
            lstmTuningService.tuneAllSymbols(Collections.singletonList(symbol), grid, jdbcTemplate, s -> getBarBySymbol(s, null));
        }
    }

    /**
     * Détermine dynamiquement la liste des features à utiliser selon le symbole.
     *
     * Règles simples :
     * - Crypto (BTC*, ETH*) : momentum + volatilité + MACD
     * - Indices (CAC*, SPX*) : focus sur moyennes + bandes
     * - Tech US (AAPL, MSFT) : set enrichi
     * - Sinon : set large par défaut
     *
     * @param symbol symbole
     * @return liste de noms de features (doivent être reconnues côté pipeline)
     */
    public List<String> getFeaturesForSymbol(String symbol) {
        if (symbol.startsWith("BTC") || symbol.startsWith("ETH")) {
            return Arrays.asList("close", "volume", "rsi", "macd", "atr", "momentum", "day_of_week", "month");
        } else if (symbol.startsWith("CAC") || symbol.startsWith("SPX")) {
            return Arrays.asList("close", "sma", "ema", "rsi", "atr", "bollinger_high", "bollinger_low", "month");
        } else if (symbol.startsWith("AAPL") || symbol.startsWith("MSFT")) {
            return Arrays.asList("close", "volume", "rsi", "sma", "ema", "macd", "atr",
                    "bollinger_high", "bollinger_low", "stochastic", "cci", "momentum", "day_of_week", "month");
        }
        return Arrays.asList("close", "volume", "rsi", "sma", "ema", "macd", "atr",
                "bollinger_high", "bollinger_low", "stochastic", "cci", "momentum", "day_of_week", "month");
    }

    /**
     * Retourne le rapport des exceptions rencontrées durant le tuning (collecté côté service).
     * @return liste immuable de rapports
     */
    public List<LstmTuningService.TuningExceptionReportEntry> getTuningExceptionReport() {
        return lstmTuningService.getTuningExceptionReport();
    }

    /**
     * Accès lecture aux rapports de drift accumulés en mémoire depuis le démarrage.
     * @return copie (liste indépendante)
     */
    public java.util.List<LstmTradePredictor.DriftReportEntry> getDriftReports() {
        return new java.util.ArrayList<>(driftReports);
    }

}
