package com.app.backend.trade.util;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

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
}
