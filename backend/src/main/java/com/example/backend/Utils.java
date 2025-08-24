package com.example.backend;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

public class Utils {


    public static final int HISTO = 180; // Nombre de jours d'historique à récupérer

    public static void log(String message) {
        System.out.println("[LOG] " + message);
    }

    // Retourne la date du jour moins 90 jours au format YYYY-MM-DD
    public static String getDateMoins90Jours() {
        LocalDate date = LocalDate.now().minusDays(HISTO);
        return date.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
    }
}

