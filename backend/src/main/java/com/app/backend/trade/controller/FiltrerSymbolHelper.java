package com.app.backend.trade.controller;

import com.app.backend.trade.model.DailyValue;
import com.app.backend.trade.util.TradeUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Controller;
import org.ta4j.core.BarSeries;

import java.util.Collections;
import java.util.List;

@Controller
public class FiltrerSymbolHelper {

    @Autowired
    private JdbcTemplate jdbcTemplate;

    public List<String> getAllAssetSymbolsEligibleFromDb() {
        String sql = "SELECT symbol FROM trade_ai.alpaca_asset WHERE status = 'active' and eligible = true and filtre_out = false ORDER BY symbol ASC;";
        return jdbcTemplate.queryForList(sql, String.class);
    }

    public BarSeries getBarBySymbol(String symbol) {
        // Construction de la requête selon la présence de limit
        String sql = "SELECT date, open, high, low, close, volume, number_of_trades, volume_weighted_average_price " +
                "FROM daily_value WHERE symbol = ? ORDER BY date ASC";

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

        return TradeUtils.mapping(results);
    }

    // Vérifie si le symbole a déjà été calculé et stocké dans la table swing_trade_metrics
    public boolean existsInSwingTradeMetrics(String symbol) {
        String sql = "SELECT COUNT(1) FROM swing_trade_metrics WHERE symbol = ?";
        Integer count = jdbcTemplate.queryForObject(sql, new Object[]{symbol}, Integer.class);
        return count != null && count > 0;
    }

    // Retourne les top N symboles les plus adaptés au swing trade
    public void triBestSwingTradeSymbols() {
        List<String> symbols = getAllAssetSymbolsEligibleFromDb();
        if (symbols == null || symbols.isEmpty()) throw new RuntimeException("aucun symbole trouvé pour le swing trade");

        class SymbolScore {
            String symbol;
            double score;
            SymbolScore(String symbol, double score) {
                this.symbol = symbol;
                this.score = score;
            }
        }

        List<SymbolScore> scored = new java.util.ArrayList<>();
        for (String symbol : symbols) {
            // Vérification si déjà calculé
            if (existsInSwingTradeMetrics(symbol)) continue;
            BarSeries series = getBarBySymbol(symbol);
            if (series == null || series.isEmpty() || series.getBarCount() < 30) continue;
            double sumAbs = 0;
            double sum = 0;
            double[] variations = new double[series.getBarCount()];
            double volumeSum = 0;
            for (int i = 0; i < series.getBarCount(); i++) {
                double open = series.getBar(i).getOpenPrice().doubleValue();
                double close = series.getBar(i).getClosePrice().doubleValue();
                double variation = close - open;
                variations[i] = variation;
                sum += variation;
                sumAbs += Math.abs(variation);
                volumeSum += series.getBar(i).getVolume().doubleValue();
            }
            double mean = sum / series.getBarCount();
            double variance = 0;
            for (double v : variations) variance += Math.pow(v - mean, 2);
            double stddev = Math.sqrt(variance / series.getBarCount());
            double ratioTendance = sumAbs == 0 ? 0 : sum / sumAbs;
            double volumeMoyen = volumeSum / series.getBarCount();
            if (volumeMoyen < 10000) continue; // seuil à ajuster selon le marché
            // Exclure les symboles à tendance négative (pas de trade à découvert)
            if (ratioTendance <= 0) continue;
            // Score combiné : pondérer volatilité et ratio de tendance
            double score = stddev * Math.abs(ratioTendance);
            // Sauvegarde des métriques dans la base
            saveSwingTradeMetric(symbol, stddev, ratioTendance, volumeMoyen, score);
            scored.add(new SymbolScore(symbol, score));
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) { } finally { }
        }
        this.triScore();
    }

    // Sauvegarde les métriques swing trade pour un symbole
    public void saveSwingTradeMetric(String symbol, double volatilite, double ratioTendance, double volumeMoyen, double score) {
        String sql = "REPLACE INTO swing_trade_metrics (symbol, volatilite, ratio_tendance, volume_moyen, score, date_calcul) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)";
        jdbcTemplate.update(sql, symbol, volatilite, ratioTendance, volumeMoyen, score);
    }

    // Trie les symboles par score décroissant et met à jour la colonne 'top' selon le classement
    public void triScore() {
        String selectSql = "SELECT symbol, score FROM swing_trade_metrics ORDER BY score DESC";
        List<java.util.Map<String, Object>> rows = jdbcTemplate.queryForList(selectSql);
        int rank = 1;
        for (java.util.Map<String, Object> row : rows) {
            String symbol = (String) row.get("symbol");
            String updateSql = "UPDATE swing_trade_metrics SET top = ? WHERE symbol = ?";
            jdbcTemplate.update(updateSql, rank, symbol);
            rank++;
        }
    }
}
