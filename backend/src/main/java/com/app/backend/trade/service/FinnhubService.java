package com.app.backend.trade.service;

import com.app.backend.trade.util.TradeConstant;
import com.app.backend.trade.util.TradeUtils;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Iterator;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

/**
 * Service pour interagir avec l'API Finnhub (données financières et statistiques).
 */
@Service
public class FinnhubService {

    @Value("${finnhub.api.key}")
    private String apiKey;

    @Value("${finnhub.api.url}")
    private String apiUrl;


    @Value("${finnhub.api.history.limit}")
    private int limit;

    private static final String CONTENJ_VIDE = "aucune information trouvée";

    /**
     * Appelle l'API Finnhub pour un endpoint donné.
     */
    private String callFinnhubApi(String endpoint, String params)  {
        try{
            RestTemplate restTemplate = new RestTemplate();
            String url = apiUrl + endpoint + "?" + params + "&token=" + apiKey;
            TradeUtils.log("Appel Finnhub API (" + endpoint + "): " + url);
            String reponse = restTemplate.getForObject(url, String.class);
            TradeUtils.log("Réponse Finnhub API (" + endpoint + "): " + reponse);
            return reponse;
        } catch (Exception e) {
            throw new RuntimeException("Exception callFinnhubApi : " + e.getMessage());
        }
    }

    /**
     * Données financières (financialData).
     */
    public String getFinancialData(String symbol)  {
        String response = callFinnhubApi("stock/financials-reported", "symbol=" + symbol);
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(response);
            // Finnhub retourne généralement un objet avec un tableau "data" contenant les rapports
            if (root.has("data") && root.get("data").isArray() && root.get("data").size() > 0) {
                ArrayNode dataArray = (ArrayNode) root.get("data");
                // On prend uniquement le premier (dernier rapport)
                ArrayNode lastReportArray = mapper.createArrayNode();
                lastReportArray.add(dataArray.get(0));
                ((ObjectNode) root).set("data", lastReportArray);
                return mapper.writeValueAsString(root);
            } else {
                return response;
            }
        } catch (Exception e) {
            throw new RuntimeException("getFinancialData : " + e.getMessage());
        }
    }

    /**
     * Statistiques clés par défaut (defaultKeyStatistics).
     */
    public String getDefaultKeyStatistics(String symbol)  {
        String response = callFinnhubApi("stock/metric", "symbol=" + symbol + "&metric=all");
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(response);
            // Supprimer la partie 'series' si elle existe
            if (root.has("series") && root instanceof ObjectNode) {
                ((ObjectNode) root).remove("series");
            }
            return mapper.writeValueAsString(root);
        } catch (Exception e) {
            throw new RuntimeException("getDefaultKeyStatistics : " + e.getMessage());
        }
    }

    // Résultats (earnings)
    public String getEarnings(String symbol)  {
        return callFinnhubApi("stock/earnings", "symbol=" + symbol);
    }

    // News (actualités)
    public String getNews(String symbol, String keywords)  {
        String params;
        String functionNews;
         // Si un symbole est fourni, on récupère les news de la semaine passée pour ce symbole
        if (symbol != null && !symbol.trim().isEmpty()) {
            java.time.LocalDate toDate = java.time.LocalDate.now();
            java.time.LocalDate fromDate = toDate.minusDays(7);
            params = String.format("symbol=%s&from=%s&to=%s", symbol, fromDate, toDate);
            functionNews = "company-news";
        } else {
            params = "category=general";
            functionNews = "news";
        }
        if (keywords != null && !keywords.trim().isEmpty()) {
            params += "&q=" + keywords.trim();
        }
        return callFinnhubApi(functionNews, params);
    }

    // News générales (actualités générales)
    public String getGeneralNews()  {
        return callFinnhubApi("company-news", "category=general");
    }
}
