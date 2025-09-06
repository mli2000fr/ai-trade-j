package com.app.backend.trade.service;

import com.app.backend.trade.model.OrderRequest;
import com.app.backend.trade.util.TradeUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpStatus;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Service pour interagir avec l'API Twelve Data (indicateurs techniques et fondamentaux).
 */
@Service
public class TwelveDataService {
    @Value("${twelvedata.api.key}")
    private String apiKey;

    @Value("${twelvedata.api.url}")
    private String apiUrl;

    @Value("${twelvedata.api.history.limit}")
    private int limit;

    /**
     * EMA 20
     */
    public String getEMA20(String symbol)  {
        return getEMA(symbol, 20, 100);
    }
    /**
     * EMA 50
     */
    public String getEMA50(String symbol)  {
        return getEMA(symbol, 50, 100);
    }
    /**
     * EMA (Exponential Moving Average)
     */
    public String getEMA(String symbol, int periode, int nombre)  {
        try {
            return getIndicator("ema", symbol, "interval=1day&time_period="+String.valueOf(periode)+"&start_date=" + TradeUtils.getStartDate(nombre));
        }catch (Exception e){
            return getIndicator("ema", symbol, "interval=1day&time_period="+String.valueOf(periode));

        }
    }

    public String getSMA200(String symbol)  {
        return getSMA(symbol, 200, 400);
    }

    /**
     * SMA (Simple Moving Average)
     */
    public String getSMA(String symbol, int periode, int nombre)  {
        try {
            return getIndicator("sma", symbol, "interval=1day&time_period=" + String.valueOf(periode) + "&start_date=" + TradeUtils.getStartDate(nombre));
        }catch (Exception e){
            return getIndicator("sma", symbol, "interval=1day&time_period=" + String.valueOf(periode / 2));
        }
    }
    /**
     * RSI (Relative Strength Index)
     */
    public String getRSI(String symbol)  {
        try {
            return getIndicator("rsi", symbol, "interval=1day&time_period=14&start_date=" + TradeUtils.getStartDate(limit)); //60
        }catch (Exception e){
            return getIndicator("rsi", symbol, "interval=1day&time_period=14&start_date=" + TradeUtils.getStartDate(20)); //60
        }
    }
    /**
     * MACD (Moving Average Convergence Divergence)
     */
    public String getMACD(String symbol)  {
        try {
            return getIndicator("macd", symbol, "interval=1day&short_period=12&long_period=26&signal_period=9&start_date=" + TradeUtils.getStartDate(limit)); //60
        }catch (Exception e){
            return getIndicator("macd", symbol, "interval=1day&short_period=12&long_period=26&signal_period=9&start_date=" + TradeUtils.getStartDate(20)); //60
        }
    }
    /**
     * ATR (Average True Range)
     */
    public String getATR(String symbol)  {
        try {
            return getIndicator("atr", symbol, "interval=1day&time_period=14&start_date=" + TradeUtils.getStartDate(limit)); //60
        }catch (Exception e){
            return getIndicator("atr", symbol, "interval=1day&time_period=14&start_date=" + TradeUtils.getStartDate(20)); //60
        }
    }
    /**
     * Données de l'action (Time Series)
     */
    public String getDataAction(String symbol)  {
        try {
            return getIndicator("time_series", symbol, "interval=1day&start_date=" + TradeUtils.getStartDate(limit));
        }catch (Exception e){
            return getIndicator("time_series", symbol, "interval=1day&start_date=" + TradeUtils.getStartDate(20));
        }
    }
    /**
     * Données fondamentales (Fundamental Data)
     */
    public String getFundamental(String symbol)  {
        return getIndicator("fundamentals", symbol, null);
    }
    /**
     * Appelle l'API Twelve Data pour un indicateur donné.
     * @param function nom de la fonction (sma, rsi, etc.)
     * @param symbol symbole de l'action
     * @param params paramètres supplémentaires
     * @return réponse JSON de l'API
     */
    private String getIndicator(String function, String symbol, String params)  {
        RestTemplate restTemplate = new RestTemplate();
        String url = apiUrl + function + "?symbol=" + symbol + "&apikey=" + apiKey + (params != null ? "&" + params : "");
        TradeUtils.log("Appel Twelve Data API (" + function + "): " + url);
        ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);
        if (response.getStatusCode() == HttpStatus.OK) {
            String responseBody = response.getBody();
            TradeUtils.log("Réponse Twelve Data API (" + function + "): " + responseBody);
            if(responseBody == null || (responseBody != null && responseBody.contains("\"status\":\"error\""))) {
                throw new RuntimeException("Erreur Twelve Data API: " + responseBody);
            }
            return responseBody;
        } else {
            throw new RuntimeException("Erreur lors de la récupération des données : " + response.getStatusCode());
        }
    }
}
