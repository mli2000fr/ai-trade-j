package com.app.backend.trade.service;

import com.app.backend.trade.util.TradeUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpStatus;

@Service
public class TwelveDataService {
    @Value("${twelvedata.api.key}")
    private String apiKey;

    @Value("${twelvedata.api.url}")
    private String apiUrl;


    // SMA (Simple Moving Average)
    public String getSMA(String symbol) {
        return getIndicator("sma", symbol, "interval=1day&time_period=12&start_date=" + TradeUtils.getDateMoins90Jours());
    }

    // RSI (Relative Strength Index)
    public String getRSI(String symbol) {
        return getIndicator("rsi", symbol, "interval=1day&time_period=14&start_date=" + TradeUtils.getDateMoins90Jours());
    }

    // MACD (Moving Average Convergence Divergence)
    public String getMACD(String symbol) {
        return getIndicator("macd", symbol, "interval=1day&short_period=12&long_period=26&signal_period=9&start_date=" + TradeUtils.getDateMoins90Jours());
    }

    // ATR (Average True Range)
    public String getATR(String symbol) {
        return getIndicator("atr", symbol, "interval=1day&time_period=14&start_date=" + TradeUtils.getDateMoins90Jours());
    }

    // Données de l'action (Time Series)
    public String getDataAction(String symbol) {
        return getIndicator("time_series", symbol, "interval=1day&start_date=" + TradeUtils.getDateMoins90Jours());
    }

    // Données fondamentales (Fundamental Data) les fondamentaux (bénéfices, prévisions, valorisation, actualités produits)
    public String getFundamental(String symbol) {
        return getIndicator("fundamentals", symbol, null);
    }



    private String getIndicator(String function, String symbol, String params) {
        RestTemplate restTemplate = new RestTemplate();
        String url = apiUrl + function + "?symbol=" + symbol + "&apikey=" + apiKey + (params != null ? "&" + params : "");
        TradeUtils.log("Appel Twelve Data API (" + function + "): " + url);
        try {
            ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);
            if (response.getStatusCode() == HttpStatus.OK) {
                String responseBody = response.getBody();
                TradeUtils.log("Réponse Twelve Data API (" + function + "): " + responseBody);
                return responseBody != null ? responseBody : "Aucune donnée reçue de l'API Twelve Data.";
            } else {
                return "Erreur lors de la récupération des données : " + response.getStatusCode();
            }
        } catch (Exception e) {
            return "Exception callTwelveDataApi: " + e.getMessage();
        }
    }
}
