package com.example.backend.util;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

public class Utils {


    public static void log(String message) {
        System.out.println("[LOG] " + message);
    }

    // Retourne la date du jour moins 90 jours au format YYYY-MM-DD
    public static String getDateMoins90Jours() {
        LocalDate date = LocalDate.now().minusDays(Constant.HISTO);
        return date.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
    }
}

