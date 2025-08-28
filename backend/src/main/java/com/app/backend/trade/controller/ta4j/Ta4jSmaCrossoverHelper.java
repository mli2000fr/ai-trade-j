package com.app.backend.trade.controller.ta4j;

import org.ta4j.core.*;
import org.ta4j.core.indicators.SMAIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.num.DecimalNum;
import org.ta4j.core.num.Num;

import java.time.ZonedDateTime;
import java.time.Duration;
import java.util.List;

public class Ta4jSmaCrossoverHelper {

    /**
     * Crée une série temporelle TA4J à partir d'une liste de prix de clôture.
     */
    public static BarSeries createSeries(List<Double> closePrices) {
        BarSeries series = new BaseBarSeriesBuilder().withName("my_series").build();
        ZonedDateTime endTime = ZonedDateTime.now();

        for (int i = 0; i < closePrices.size(); i++) {
            Double price = closePrices.get(i);
            // Créer une barre avec le prix de clôture, en utilisant le même prix pour open, high, low
            ZonedDateTime barEndTime = endTime.minusDays(closePrices.size() - i - 1);
            Num priceNum = DecimalNum.valueOf(price);
            Num volume = DecimalNum.valueOf(1000);

            // Création d'une barre simple avec tous les prix identiques
            series.addBar(barEndTime, priceNum, priceNum, priceNum, priceNum, volume);
        }
        return series;
    }

    /**
     * Applique la stratégie SMA crossover sur la série et retourne les signaux d'achat/vente.
     * Retourne une liste de signaux (index, type)
     */
    public static List<String> runSmaCrossoverStrategy(BarSeries series, int shortPeriod, int longPeriod) {
        ClosePriceIndicator closePrice = new ClosePriceIndicator(series);
        SMAIndicator shortSma = new SMAIndicator(closePrice, shortPeriod);
        SMAIndicator longSma = new SMAIndicator(closePrice, longPeriod);

        // Détection manuelle des croisements
        List<String> signals = new java.util.ArrayList<>();
        boolean inPosition = false;
        for (int i = 1; i < series.getBarCount(); i++) {
            Num prevShort = shortSma.getValue(i - 1);
            Num prevLong = longSma.getValue(i - 1);
            Num currShort = shortSma.getValue(i);
            Num currLong = longSma.getValue(i);
            // Croisement à la hausse (achat)
            if (!inPosition && prevShort.isLessThan(prevLong) && currShort.isGreaterThan(currLong)) {
                signals.add("BUY at index " + i);
                inPosition = true;
            }
            // Croisement à la baisse (vente)
            else if (inPosition && prevShort.isGreaterThan(prevLong) && currShort.isLessThan(currLong)) {
                signals.add("SELL at index " + i);
                inPosition = false;
            }
        }
        return signals;
    }
}
