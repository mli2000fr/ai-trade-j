package com.app.backend.trade.util;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

public class TradeUtils {


    public static void log(String message) {
        System.out.println("[LOG] " + message);
    }

    // Retourne la date du jour moins 90 jours au format YYYY-MM-DD
    public static String getStartDate(int histo) {
        LocalDate date = LocalDate.now().minusDays(histo);
        return date.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
    }

    // Lit le contenu d'un fichier de ressources et le retourne sous forme de String
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
}
