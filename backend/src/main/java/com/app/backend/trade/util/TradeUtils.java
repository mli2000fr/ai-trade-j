package com.app.backend.trade.util;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

public class TradeUtils {


    public static void log(String message) {
        System.out.println("[LOG] " + message);
    }

    // Retourne la date du jour moins 90 jours au format YYYY-MM-DD
    public static String getDateMoins90Jours() {
        LocalDate date = LocalDate.now().minusDays(TradeConstant.HISTO);
        return date.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
    }
}

