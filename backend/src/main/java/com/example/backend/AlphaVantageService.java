package com.example.backend;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpStatus;

@Service
public class AlphaVantageService {
    @Value("${alphavantage.api.key}")
    private String apiKey;
    private final String BASE_URL = "https://www.alphavantage.co/query?function=";
    private final String SURFIX_URL = "&interval=daily&time_period=12&series_type=close";

    //SMA (Simple Moving Average) Tendances globales (simple).
    public String getSMA(String symbol) {
        return getInfosAction("SMA", symbol, SURFIX_URL);
    }

    //RSI (Relative Strength Index) Zones extrêmes (surachat/survente).
    public String getRSI(String symbol) {
        return getInfosAction("RSI", symbol, SURFIX_URL);
    }

    //MACD (Moving Average Convergence Divergence) Force et changement de tendance.
    public String getMACD(String symbol) {
        return getInfosAction("MACD", symbol, SURFIX_URL);
    }

    //Volatilité (via ATR = Average True Range) la taille des mouvements possibles.
    public String getATR(String symbol) {
        return getInfosAction("ATR", symbol, SURFIX_URL);
    }

    //Volatilité (Moving Average Convergence Divergence)
    public String getDataAction(String symbol) {
        return getInfosAction("TIME_SERIES_DAILY", symbol, SURFIX_URL);
    }

    // Données fondamentales (Fundamental Data)
    public String getFundamental(String symbol) {
        return getInfosAction("OVERVIEW", symbol, null);
    }

    public String getInfosAction(String functionName, String symbol, String surfix) {
        RestTemplate restTemplate = new RestTemplate();
        String url = BASE_URL + functionName + "&symbol=" + symbol + "&apikey=" + apiKey + (surfix == null ? "" : surfix);
        System.out.println("call l'API Alpha Vantage ("+functionName+"): " + url);
        try {
            ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);
            if (response.getStatusCode() == HttpStatus.OK) {
                String responseBody =  response.getBody();
                System.out.println("Réponse de l'API Alpha Vantage ("+functionName+"): " + responseBody);
                return responseBody != null ? responseBody : "Aucune donnée reçue de l'API Alpha Vantage.";
            } else {
                return "Erreur lors de la récupération des données : " + response.getStatusCode();
            }
        } catch (Exception e) {
            return "Exception : " + e.getMessage();
        }
    }
}
