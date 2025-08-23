package com.example.backend;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

@Service
public class GoogleCustomSearchService {
    @Value("${google.api.key}")
    private String apiKey;

    @Value("${google.api.cx}")
    private String cx;

    private final RestTemplate restTemplate = new RestTemplate();

    public String search(String query) {
        String url = UriComponentsBuilder.fromHttpUrl("https://www.googleapis.com/customsearch/v1")
                .queryParam("key", apiKey)
                .queryParam("cx", cx)
                .queryParam("q", query)
                .toUriString();
        try {
            ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);
            return response.getBody();
        } catch (Exception e) {
            return "{\"error\":\"" + e.getMessage() + "\"}";
        }
    }
}

