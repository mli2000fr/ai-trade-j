package com.app.backend.trade.service;

import com.app.backend.trade.util.TradeUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpStatus;

import java.util.Map;

/**
 * Service pour interagir avec l'API EODHD (actualités financières).
 */
@Service
public class EodhdService {
    @Value("${eodhd.api.key}")
    private String apiKey;

    @Value("${eodhd.api.url}")
    private String apiUrl;

    @Value("${eodhd.api.news.limit}")
    private int limit;


    /**
     * Récupère les news pour un symbole donné (ou toutes les news si symbol est null).
     * @param symbol symbole de l'action (optionnel)
     * @return JSON filtré des news
     */
    public String getNews(String symbol)  {
        RestTemplate restTemplate = new RestTemplate();
        StringBuilder urlBuilder = new StringBuilder(apiUrl);
        if (!apiUrl.endsWith("/")) {
            urlBuilder.append("/");
        }
        urlBuilder.append("news?limit=").append(limit)
                  .append("&api_token=").append(apiKey);
        String url = urlBuilder.toString();
        if (symbol != null && !symbol.trim().isEmpty()) {
            url += "&s=" + symbol.trim();
        }
        TradeUtils.log("Appel EODHD API (news): " + url);
        ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);
        if (response.getStatusCode() == HttpStatus.OK) {
            String responseBody = response.getBody();
            TradeUtils.log("Réponse EODHD API (news): " + responseBody);
            // Supprimer le champ 'content' de chaque news
            if (responseBody != null) {
                // Utilisation d'une manipulation JSON simple
                try {
                    com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                    java.util.List<java.util.Map<String, Object>> newsList = mapper.readValue(responseBody, java.util.List.class);
                    for (java.util.Map<String, Object> news : newsList) {
                        news.remove("content");
                        news.remove("link");
                        Map<String, Object> sentiment = (Map<String, Object>) news.get("sentiment");
                        double neg = 0.0, neu = 0.0, pos = 0.0;
                        if (sentiment != null) {
                            neg = sentiment.get("neg") != null ? ((Number) sentiment.get("neg")).doubleValue() : 0.0;
                            neu = sentiment.get("neu") != null ? ((Number) sentiment.get("neu")).doubleValue() : 0.0;
                            pos = sentiment.get("pos") != null ? ((Number) sentiment.get("pos")).doubleValue() : 0.0;
                        }
                        news.put("sentimentInterpreted", interpretSentiment(neg, neu, pos));
                    }
                    // Retirer les news neutres
                    newsList.removeIf(news -> "neutral".equals(news.get("sentimentInterpreted")));
                    return mapper.writeValueAsString(newsList);
                } catch (Exception jsonEx) {
                    throw new RuntimeException("Erreur lors du traitement JSON : " + jsonEx.getMessage());
                }
            } else {
                throw new RuntimeException("Aucune donnée reçue de l'API EODHD.");
            }
        } else {
            throw new RuntimeException("Erreur lors de la récupération des données : " + response.getStatusCode());
        }
    }

    /**
     * Interprète le score de sentiment en une catégorie lisible.
     */
    public String interpretSentiment(double neg, double neu, double pos) {
        // 1. Cas neutre dominant
        if (neu >= 0.7 && pos < 0.2 && neg < 0.2) {
            return "neutral";
        }
        // 2. Cas positif dominant
        if (pos > neg && pos >= 0.2) {
            return "positive";
        }
        // 3. Cas négatif dominant
        if (neg > pos && neg >= 0.2) {
            return "negative";
        }
        // 4. Par défaut
        return "neutral";
    }
}
