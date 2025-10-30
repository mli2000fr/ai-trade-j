package com.app.backend.trade.controller;


import com.app.backend.trade.lstm.LstmConfig;
import com.app.backend.trade.lstm.LstmTradePredictor;
import com.app.backend.trade.lstm.LstmTuningService;
import com.app.backend.trade.model.*;
import com.app.backend.trade.strategy.BestInOutStrategy;
import com.app.backend.trade.strategy.ParamsOptim;
import com.app.backend.trade.util.TradeUtils;
import com.google.gson.Gson;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Controller;
import org.ta4j.core.BarSeries;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.app.backend.trade.lstm.LstmDataAuditService;

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
    // Service d'audit des données (ajouté)
    private final LstmDataAuditService lstmDataAuditService;

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
        this.lstmDataAuditService = new LstmDataAuditService(this);
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


    public List<String> getBestModel(Integer limit, String sort, Boolean topClassement) {
        String orderBy = (sort == null || sort.isBlank()) ? "business_score" : sort;
        String orderBySup = "";
        if(sort.equals("classement")){
            orderBySup = " stm.top ASC ";
            orderBy = " business_score ";
            topClassement = true;
        }else{
            orderBySup = " CASE WHEN lm."+ orderBy +" IS NULL THEN 1 ELSE 0 END, lm."+ orderBy +" DESC";
        }
        String sql_tous_sym = "SELECT aa.symbol, stm.top" +
                " FROM alpaca_asset aa" +
                " LEFT JOIN trade_ai.swing_trade_metrics stm ON aa.symbol = stm.symbol";
        String sql_sym_top_classement = "SELECT stm.symbol, stm.top" +
                " FROM trade_ai.swing_trade_metrics stm";
        String prefix = (topClassement != null && topClassement) ? sql_sym_top_classement : sql_tous_sym;
        String condTousSym = (topClassement != null && topClassement) ? "" : " WHERE aa.status = 'active' AND aa.eligible = true AND aa.filtre_out = false";
        String sql = prefix +
                " LEFT JOIN (" +
                " SELECT symbol, rendement, business_score" +
                " FROM (" +
                " SELECT symbol, rendement, business_score," +
                " ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY "+ orderBy +" DESC) as rn" +
                " FROM trade_ai.lstm_models" +
                " ) t" +
                " WHERE t.rn = 1" +
                " ) lm ON stm.symbol = lm.symbol " +
                condTousSym +
                " ORDER BY" + orderBySup;
        if (limit != null && limit > 0) {
            sql += " LIMIT " + limit;
        }
        List<String> results = jdbcTemplate.query(sql, (rs, rowNum) -> {
            return rs.getString("symbol");
        });
        return results;
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
    public PreditLsdm getPredit(String symbol, String index) {
        // 1. Vérifier si une prédiction du jour existe déjà (évite recalcul)
        PreditLsdm preditLsdmDb = this.getPreditFromDB(symbol, index);
        if (preditLsdmDb != null) {
            return preditLsdmDb;
        }

        // 3. Charger modèle + scalers + config (si échec : on continue quand même => modèle null => prédiction fallback)
        LstmTradePredictor.LoadedModel loaded = null;
        try {
            loaded = lstmTradePredictor.loadModelAndScalersFromDb(symbol, index, jdbcTemplate);
        } catch (Exception e) {
            logger.warn("Impossible de charger le modèle/scalers depuis la base : {}", e.getMessage());
        }
        MultiLayerNetwork model = loaded != null ? loaded.model : null;
        LstmTradePredictor.ScalerSet scalers = loaded != null ? loaded.scalers : null;
        LstmConfig config = loaded != null ? loaded.config : null;

        // 4. Charger les données prix (BarSeries complet)
        BarSeries series = getBarBySymbol(symbol, null);


        // 6. Exécution de la prédiction (utilise model/scalers si présents)
        PreditLsdm preditLsdm = PreditLsdm.builder().build();

        if (loaded == null) {
            preditLsdm.setSignal(SignalType.NONE);
            return preditLsdm;
        }

        if(loaded.phase == 0 ){
            preditLsdm = lstmTradePredictor.getPredit(series, config, model, scalers);
        }else{
            TradeStylePrediction tradeStylePrediction = lstmTradePredictor.predictTradeStyle(symbol, series, config, model, scalers);

            preditLsdm.setLastClose(tradeStylePrediction.lastClose);
            preditLsdm.setPredictedClose(tradeStylePrediction.predictedClose);
            preditLsdm.setSignal(tradeStylePrediction.action.equals("BUY") ? SignalType.BUY :
                    tradeStylePrediction.action.equals("SELL") ? SignalType.SELL : SignalType.NONE);
            preditLsdm.setLastDate(java.time.LocalDate.now().format(java.time.format.DateTimeFormatter.ofPattern("dd/MM")));
            preditLsdm.setPosition(tradeStylePrediction.tendance);
            preditLsdm.setExplication(tradeStylePrediction.comment);
            tradeStylePrediction.loadedModel = loaded;
        }
        loaded.model = null;
        loaded.scalers = null;
        loaded.config = null;
        preditLsdm.setLoadedModel(loaded);

        // 7. Sauvegarde du signal du jour
        saveSignalHistory(symbol, index, preditLsdm);
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
    public void saveSignalHistory(String symbol, String tri, PreditLsdm preditLsdm) {
        java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(java.time.LocalDate.now());
        String insertSql = "INSERT INTO signal_lstm (symbol, tri, signal_lstm, price_lstm, price_clo, position_lstm, lstm_created_at, result_tuning) VALUES (?, ?, ?, ?, ?, ?, ?, ?)";
        jdbcTemplate.update(insertSql,
                symbol,
                preditLsdm.getSignal().name(),
                tri,
                preditLsdm.getPredictedClose(),
                preditLsdm.getLastClose(),
                preditLsdm.getPosition(),
                java.sql.Date.valueOf(lastTradingDay),
                preditLsdm.getLoadedModel() != null ? new Gson().toJson(preditLsdm.getLoadedModel()) : null);
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
    public PreditLsdm getPreditFromDB(String symbol, String tri) {
        String sql = "SELECT * FROM signal_lstm WHERE symbol = ? AND tri = ? ORDER BY lstm_created_at DESC LIMIT 1";
        try {
            return jdbcTemplate.query(sql, ps -> {
                ps.setString(1, symbol);
                ps.setString(2, tri);
            }, rs -> {
                if (rs.next()) {
                    String signalStr = rs.getString("signal_lstm");
                    double priceLstm = rs.getDouble("price_lstm");
                    double priceClo = rs.getDouble("price_clo");
                    String positionLstm = rs.getString("position_lstm");
                    java.sql.Date lastDate = rs.getDate("lstm_created_at");
                    String tuning_result =  rs.getString("result_tuning");
                    LstmTradePredictor.LoadedModel loadedModel = new Gson().fromJson(tuning_result, LstmTradePredictor.LoadedModel.class);

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
                                .loadedModel(loadedModel)
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

    public List<String> getSymbolTopClassement() {
        String sql = "select symbol from swing_trade_metrics ORDER BY top ASC";
        return jdbcTemplate.queryForList(sql, String.class);
    }

    // Méthode conservée (signature legacy)
    public void tuneAllSymbols() {
        tuneAllSymbols(true, 150);
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
        //List<String> symbols = getSymbolFitredFromTabSingle("rendement");

        List<String> symbols = getSymbolTopClassement();

        // Valeurs testées (ne pas modifier sans validation)
        int[] horizonBars = {3, 5, 10};
        int[] numLstmLayers = {1, 2, 3};
        int[] batchSizes = {64, 128, 256};
        boolean[] bidirectionals = {true, false};
        boolean[] attentions = {true, false};

        List<LstmConfig> grid;
        if (useRandomGrid) {
            // Si budget limité (<=60 configs), utiliser la stratégie optimisée exploration/exploitation
            if (randomGridSize <= 200) {
                grid = lstmTuningService.generateRandomSwingTradeGridOptimized(randomGridSize);
            } else {
                grid = lstmTuningService.generateRandomSwingTradeGrid(
                        randomGridSize, horizonBars, numLstmLayers, batchSizes, bidirectionals, attentions
                );
            }
        } else {
            grid = lstmTuningService.generateSwingTradeGrid(horizonBars, numLstmLayers, batchSizes, bidirectionals, attentions
            );
        }
        lstmTuningService.tuneAllSymbols(symbols, grid, jdbcTemplate, s -> getBarBySymbol(s, null));
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

    /**
     * Lance un audit complet des données pour tous les symboles filtrés.
     * @param windowSize taille de la fenêtre d'entrée
     * @param horizonBars horizon de prédiction
     * @return liste des symboles valides
     */
    public List<String> auditAllSymbols(int windowSize, int horizonBars) {
        return lstmDataAuditService.auditAllSymbols(windowSize, horizonBars);
    }

}
