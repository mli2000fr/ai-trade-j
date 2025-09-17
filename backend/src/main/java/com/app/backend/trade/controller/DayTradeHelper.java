package com.app.backend.trade.controller;

import com.app.backend.trade.model.DailyValue;
import com.app.backend.trade.service.AlpacaService;
import com.app.backend.trade.util.TradeConstant;
import com.app.backend.trade.util.TradeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Controller;

import java.util.ArrayList;
import java.util.List;

@Controller
public class DayTradeHelper {


    private static final Logger logger = LoggerFactory.getLogger(DayTradeHelper.class);

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Autowired
    private AlpacaService alpacaService;

    public int alimenteDBMinValue(String symbol) throws InterruptedException {
        int totalInsert = this.updatMinValue(symbol);
        logger.info("Insertion min_value size{}", totalInsert);
        return totalInsert;
    }

    public void insertDailyValue(String symbol, List<DailyValue> listeValues){
        for(DailyValue dv : listeValues){
            this.insertDailyValue(symbol, dv);
        }
    }

    public  int updatMinValue(String symbol) throws InterruptedException {

        // 1. Chercher la date la plus récente pour ce symbol dans la table min_value
        String sql = "SELECT MAX(date) FROM min_value WHERE symbol = ?";
        java.sql.Date lastDate = null;
        try {
            lastDate = jdbcTemplate.query(sql, ps -> ps.setString(1, symbol), rs -> {
                if (rs.next()) {
                    return rs.getDate(1);
                }
                return null;
            });
        } catch (Exception e) {
            logger.warn("Aucune date trouvée pour le symbole {} dans min_value ou erreur SQL: {}", symbol, e.getMessage());
        }
        String dateStart;
        java.time.LocalDateTime todayDateTime = java.time.LocalDateTime.now();
        java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(todayDateTime.toLocalDate());
        if (lastDate == null) {
            // Si aucune ligne trouvée, on prend la date de start par défaut
            dateStart = TradeUtils.getStartDate(TradeConstant.HISTORIQUE_MIN_VALUE);
            // S'assurer du format RFC3339
            if (!dateStart.endsWith("Z")) {
                dateStart = dateStart + "T00:00:00Z";
            }
        } else {
            java.time.LocalDateTime lastKnown = lastDate.toLocalDate().atStartOfDay();
            // Si la dernière date connue est le dernier jour de cotation, la base est à jour
            if (lastKnown.toLocalDate().isEqual(lastTradingDay) || lastKnown.toLocalDate().isAfter(lastTradingDay)) {
                return -1; // Base à jour
            }
            // Sinon, on ajoute une minute à la date la plus récente
            java.time.LocalDateTime nextMinute = lastKnown.plusMinutes(1);
            dateStart = nextMinute.toString().replace("T", "T") + ":00Z"; // format RFC3339
        }
        int compteur = 0;
        String currentStart = dateStart;
        while (true) {
            // S'assurer que currentStart est bien au format RFC3339
            if (!currentStart.endsWith("Z")) {
                if (currentStart.length() == 16) { // YYYY-MM-DDTHH:mm
                    currentStart = currentStart + ":00Z";
                } else if (currentStart.length() == 19) { // YYYY-MM-DDTHH:mm:ss
                    currentStart = currentStart + "Z";
                }
            }
            List<DailyValue> batch = this.alpacaService.getHistoricalBarsJsonDaysMin(symbol, currentStart);
            if (batch == null || batch.isEmpty()) {
                break;
            }
            compteur += batch.size();
            logger.info("Récupération batch de {} bougies pour {} à partir de {} (total {})", batch.size(), symbol, currentStart, compteur);
            // Insérer les valeurs
            this.insertDailyValue(symbol, batch);
            // Si moins de 1000 bougies, on a tout récupéré
            if (batch.size() < 1000) {
                break;
            }
            // Mettre à jour la date de départ pour la prochaine requête
            DailyValue last = batch.get(batch.size() - 1);
            String lastDateStr = last.getDate();
            if (lastDateStr == null || lastDateStr.length() < 16) {
                break;
            }
            java.time.LocalDateTime lastBatchDateTime = java.time.LocalDateTime.parse(lastDateStr.substring(0, 16)); // format YYYY-MM-DDTHH:mm
            if (lastBatchDateTime.toLocalDate().isEqual(lastTradingDay) || lastBatchDateTime.toLocalDate().isAfter(lastTradingDay)) {
                break;
            }
            if(compteur > 15000) {
                logger.warn("Trop de bougies insérées pour {}: {}, arrêt du processus pour éviter surcharge", symbol, compteur);
                break;
            }
            currentStart = lastBatchDateTime.plusMinutes(1).toString().replace("T", "T") + ":00Z";
            try {
                Thread.sleep(200); // Pause pour éviter de surcharger l'API
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        return compteur;
    }

    public void insertDailyValue(String symbol, DailyValue dailyValue) {
        // Conversion de la date (ex: "2025-09-16T09:31:00Z" ou "2025-09-16") en java.sql.Timestamp
        java.sql.Timestamp sqlTimestamp = null;
        if (dailyValue.getDate() != null && !dailyValue.getDate().isEmpty()) {
            String dateStr = dailyValue.getDate();
            // Extraction de la partie date et heure (YYYY-MM-DDTHH:mm:ss)
            if (dateStr.length() >= 19) {
                // Format complet avec heure
                dateStr = dateStr.substring(0, 19).replace('T', ' ');
            } else if (dateStr.length() >= 10) {
                // Format date seule
                dateStr = dateStr.substring(0, 10) + " 00:00:00";
            }
            try {
                sqlTimestamp = java.sql.Timestamp.valueOf(dateStr);
            } catch (Exception e) {
                logger.warn("Format de date inattendu pour DailyValue: {}", dailyValue.getDate());
            }
        }
        String sql = "INSERT INTO min_value (symbol, date, open, high, low, close, volume, number_of_trades, volume_weighted_average_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";
        jdbcTemplate.update(sql,
                symbol,
                sqlTimestamp,
                dailyValue.getOpen(),
                dailyValue.getHigh(),
                dailyValue.getLow(),
                dailyValue.getClose(),
                dailyValue.getVolume(),
                dailyValue.getNumberOfTrades(),
                dailyValue.getVolumeWeightedAveragePrice()
        );
    }
}
