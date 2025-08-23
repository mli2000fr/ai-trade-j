package com.example.backend;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpStatus;

@Service
public class TwelveDataService {
    @Value("${twelvedata.api.key}")
    private String apiKey;
    private final String BASE_URL = "https://api.twelvedata.com/";

    // SMA (Simple Moving Average)
    public String getSMA(String symbol) {
        return getIndicator("sma", symbol, "interval=1day&time_period=12");
    }

    // RSI (Relative Strength Index)
    public String getRSI(String symbol) {
        return getIndicator("rsi", symbol, "interval=1day&time_period=14");
    }

    // MACD (Moving Average Convergence Divergence)
    public String getMACD(String symbol) {
        return getIndicator("macd", symbol, "interval=1day&short_period=12&long_period=26&signal_period=9");
    }

    // ATR (Average True Range)
    public String getATR(String symbol) {
        return getIndicator("atr", symbol, "interval=1day&time_period=14");
    }

    // Données de l'action (Time Series)
    public String getDataAction(String symbol) {
        return getIndicator("time_series", symbol, "interval=1day");
    }

    // Données fondamentales (Fundamental Data) les fondamentaux (bénéfices, prévisions, valorisation, actualités produits)
    public String getFundamental(String symbol) {
        return getIndicator("fundamentals", symbol, null);
    }

    private String getIndicator(String function, String symbol, String params) {
        RestTemplate restTemplate = new RestTemplate();
        String url = BASE_URL + function + "?symbol=" + symbol + "&apikey=" + apiKey + (params != null ? "&" + params : "");
        System.out.println("Appel Twelve Data API (" + function + "): " + url);
        try {
            ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);
            if (response.getStatusCode() == HttpStatus.OK) {
                String responseBody = response.getBody();
                System.out.println("Réponse Twelve Data API (" + function + "): " + responseBody);
                return responseBody != null ? responseBody : "Aucune donnée reçue de l'API Twelve Data.";
            } else {
                return "Erreur lors de la récupération des données : " + response.getStatusCode();
            }
        } catch (Exception e) {
            return "Exception : " + e.getMessage();
        }
    }
}
