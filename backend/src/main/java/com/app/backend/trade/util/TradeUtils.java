package com.app.backend.trade.util;

import com.app.backend.model.RiskResult;
import com.app.backend.trade.model.DailyValue;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.List;

/**
 * Classe utilitaire pour fonctions diverses liées au trading (logs, dates, nettoyage HTML, etc.).
 */
public class TradeUtils {
    /**
     * Affiche un message de log dans la console (préfixé).
     * @param message message à afficher
     */
    public static void log(String message) {
        System.out.println("[LOG] " + message);
    }

    /**
     * Retourne la date du jour moins un nombre de jours donné, au format YYYY-MM-DD.
     * @param histo nombre de jours à soustraire
     * @return date au format yyyy-MM-dd
     */
    public static String getStartDate(int histo) {
        LocalDate date = LocalDate.now().minusDays(histo);
        return date.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
    }
    public static String getDateToDay() {
        return LocalDate.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
    }

    /**
     * Lit le contenu d'un fichier de ressources et le retourne sous forme de String.
     * @param path chemin du fichier dans les ressources
     * @return contenu du fichier
     */
    public static String readResourceFile(String path) {
        try (java.io.InputStream is = Thread.currentThread().getContextClassLoader().getResourceAsStream(path)) {
            if (is == null) {
                throw new RuntimeException("Fichier de ressource non trouvé : " + path);
            }
            java.util.Scanner scanner = new java.util.Scanner(is, java.nio.charset.StandardCharsets.UTF_8.name());
            String content = scanner.useDelimiter("\\A").hasNext() ? scanner.next() : "";
            scanner.close();
            return content;
        } catch (Exception e) {
            throw new RuntimeException("Erreur lors de la lecture du fichier de ressource : " + path, e);
        }
    }

    /**
     * Supprime toutes les balises HTML d'une chaîne de caractères et ne garde que le texte.
     * @param htmlContent contenu HTML à nettoyer
     * @return texte sans balises HTML
     */
    public static String stripHtmlTags(String htmlContent) {
        if (htmlContent == null || htmlContent.isEmpty()) {
            return htmlContent;
        }

        // Remplacer les entités HTML communes
        String cleaned = htmlContent
                .replaceAll("&amp;", "&")
                .replaceAll("&lt;", "<")
                .replaceAll("&gt;", ">")
                .replaceAll("&quot;", "\"")
                .replaceAll("&#8216;", "'")
                .replaceAll("&#8217;", "'")
                .replaceAll("&#8220;", "\"")
                .replaceAll("&#8221;", "\"")
                .replaceAll("&nbsp;", " ");

        // Supprimer toutes les balises HTML
        cleaned = cleaned.replaceAll("<[^>]+>", "");

        // Nettoyer les espaces multiples et les retours à la ligne excessifs
        cleaned = cleaned
                .replaceAll("\\s+", " ")  // Remplacer plusieurs espaces par un seul
                .replaceAll("\\n\\s*\\n", "\n\n")  // Limiter à maximum 2 retours à la ligne consécutifs
                .trim();  // Supprimer les espaces en début et fin

        return cleaned;
    }

    /**
     * Conversion Map<String, Double> en JSON
     */
    public static String convertMapToJson(java.util.Map<String, Double> map) {
        if (map == null) return null;
        com.google.gson.JsonObject jsonObj = new com.google.gson.JsonObject();
        map.forEach(jsonObj::addProperty);
        return new com.google.gson.Gson().toJson(jsonObj);
    }

    /**
     * Conversion Map<String, RiskResult> en JSON
     */
    public static String convertDetailedResultsToJson(java.util.Map<String, RiskResult> detailedResults) {
        if (detailedResults == null) return null;
        com.google.gson.JsonObject jsonObj = new com.google.gson.JsonObject();
        detailedResults.forEach((key, result) -> {
            com.google.gson.JsonObject resultObj = new com.google.gson.JsonObject();
            resultObj.addProperty("rendement", result.rendement);
            resultObj.addProperty("maxDrawdown", result.maxDrawdown);
            resultObj.addProperty("tradeCount", result.tradeCount);
            resultObj.addProperty("winRate", result.winRate);
            resultObj.addProperty("avgPnL", result.avgPnL);
            resultObj.addProperty("profitFactor", result.profitFactor);
            resultObj.addProperty("avgTradeBars", result.avgTradeBars);
            resultObj.addProperty("maxTradeGain", result.maxTradeGain);
            resultObj.addProperty("maxTradeLoss", result.maxTradeLoss);
            jsonObj.add(key, resultObj);
        });
        return new com.google.gson.Gson().toJson(jsonObj);
    }

    /**
     * Conversion JSON en Map<String, Double>
     */
    public static java.util.Map<String, Double> convertJsonToPerformanceMap(String json) {
        if (json == null) return new java.util.HashMap<>();
        try {
            com.google.gson.JsonObject jsonObj = new com.google.gson.JsonParser().parse(json).getAsJsonObject();
            java.util.Map<String, Double> map = new java.util.HashMap<>();
            jsonObj.entrySet().forEach(entry ->
                map.put(entry.getKey(), entry.getValue().getAsDouble()));
            return map;
        } catch (Exception e) {
            log("Erreur conversion JSON vers Map<String, Double>: " + e.getMessage());
            return new java.util.HashMap<>();
        }
    }

    /**
     * Mapping List<DailyValue> vers BarSeries
     */
    public static org.ta4j.core.BarSeries mapping(List<DailyValue> listeValues) {
        org.ta4j.core.BarSeries series = new org.ta4j.core.BaseBarSeries();
        for (com.app.backend.trade.model.DailyValue dailyValue : listeValues) {
            try {
                java.time.ZonedDateTime dateTime;
                if (dailyValue.getDate().length() == 10) {
                    dateTime = java.time.LocalDate.parse(dailyValue.getDate())
                            .atStartOfDay(java.time.ZoneId.systemDefault());
                } else {
                    dateTime = java.time.ZonedDateTime.parse(dailyValue.getDate());
                }
                series.addBar(
                        dateTime,
                        Double.parseDouble(dailyValue.getOpen()),
                        Double.parseDouble(dailyValue.getHigh()),
                        Double.parseDouble(dailyValue.getLow()),
                        Double.parseDouble(dailyValue.getClose()),
                        Double.parseDouble(dailyValue.getVolume())
                );
            } catch (Exception e) {
                log("Erreur conversion DailyValue en BarSeries pour la date " + dailyValue.getDate() + ": " + e.getMessage());
            }
        }
        return series;
    }

    /**
     * Parse les paramètres JSON selon le type de stratégie
     */
    public static Object parseStrategyParams(String name, String json) {
        com.google.gson.Gson gson = new com.google.gson.Gson();
        switch (name) {
            case "Improved Trend":
                return gson.fromJson(json, com.app.backend.trade.strategy.StrategieBackTest.ImprovedTrendFollowingParams.class);
            case "SMA Crossover":
                return gson.fromJson(json, com.app.backend.trade.strategy.StrategieBackTest.SmaCrossoverParams.class);
            case "RSI":
                return gson.fromJson(json, com.app.backend.trade.strategy.StrategieBackTest.RsiParams.class);
            case "Breakout":
                return gson.fromJson(json, com.app.backend.trade.strategy.StrategieBackTest.BreakoutParams.class);
            case "MACD":
                return gson.fromJson(json, com.app.backend.trade.strategy.StrategieBackTest.MacdParams.class);
            case "Mean Reversion":
                return gson.fromJson(json, com.app.backend.trade.strategy.StrategieBackTest.MeanReversionParams.class);
            default:
                return null;
        }
    }

    public static Object parseStrategyName(String name) {
        switch (name) {
            case "Improved Trend":
                return "ImprovedTrendFollowingStrategy";
            case "SMA Crossover":
                return "SmaCrossoverStrategy";
            case "RSI":
                return "RsiStrategy";
            case "Breakout":
                return "BreakoutStrategy";
            case "MACD":
                return "MacdStrategy";
            case "Mean Reversion":
                return "MeanReversionStrategy";
            default:
                return null;
        }
    }

    public static double calculerScoreSwingTrade(RiskResult r) {
        double poidsRendement = 2.0;
        double poidsWinRate = 1.5;
        double poidsProfitFactor = 1.0;
        double poidsDrawdown = 2.0;
        double poidsAvgPnL = 1.0;
        double scorePF = r.profitFactor > 10 ? 10 : r.profitFactor; // Limiter le score du profit factor à 10 max
        return (r.rendement * poidsRendement)
                + (r.winRate * poidsWinRate)
                + (scorePF * poidsProfitFactor)
                - (r.maxDrawdown * poidsDrawdown)
                + (r.avgPnL * poidsAvgPnL);
    }
}
