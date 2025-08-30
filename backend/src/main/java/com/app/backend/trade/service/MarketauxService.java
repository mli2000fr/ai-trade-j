package com.app.backend.trade.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.ResponseEntity;

/**
 * Service pour interagir avec l'API Marketaux (actualités financières).
 */
@Service
public class MarketauxService {
    @Value("${marketaux.api.key}")
    private String apiKey;

    @Value("${marketaux.api.url}")
    private String apiUrl;

    /**
     * Récupère les news pour un symbole donné.
     * @param symbol symbole de l'action
     * @return JSON des news ou message d'erreur
     */
    public String getNews(String symbol) {
        try {
            RestTemplate restTemplate = new RestTemplate();
            String url = apiUrl + "/v1/news/all?limit=50&symbols=" + symbol + "&api_token=" + apiKey;
            ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);
            return response.getBody();
        } catch (Exception e) {
            return "Erreur lors de la récupération des news Marketaux : " + e.getMessage();
        }
    }
}
