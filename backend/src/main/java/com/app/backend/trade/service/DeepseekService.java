package com.app.backend.trade.service;

import com.app.backend.trade.model.Agent;
import com.app.backend.trade.model.AgentEntity;
import com.app.backend.trade.model.AgentResponse;
import com.app.backend.trade.repository.DeepseekRepository;
import com.app.backend.trade.util.TradeUtils;
import org.json.JSONObject;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import java.time.LocalDateTime;

@Service
public class DeepseekService {
    @Value("${deepseek.api.url}")
    private String apiUrl;


    @Value("${deepseek.api.key}")
    private String apiKey;

    @Value("${deepseek.api.model}")
    private String apiModel;

    private final DeepseekRepository deepseekRepository;

    public DeepseekService(DeepseekRepository deepseekRepository) {
        this.deepseekRepository = deepseekRepository;
    }

    /**
     * Envoie un prompt à Deepseek et retourne la réponse.
     * @param prompt texte à envoyer à l'API
     * @return réponse de Deepseek ou message d'erreur
     */
    public AgentResponse askDeepseek(String prompt) {
        RestTemplate restTemplate = new RestTemplate();
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setBearerAuth(apiKey);
        // Construction du body JSON
        JSONObject body = new JSONObject();
        body.put("model", apiModel);
        body.put("messages", new org.json.JSONArray()
            .put(new JSONObject().put("role", "user").put("content", prompt)));
        HttpEntity<String> entity = new HttpEntity<>(body.toString(), headers);
        try {
            TradeUtils.log("askDeepseek Prompt JSON : " + body.toString());
            ResponseEntity<String> response = restTemplate.postForEntity(apiUrl, entity, String.class);
            JSONObject json = new JSONObject(response.getBody());
            String res = json.getJSONArray("choices")
                .getJSONObject(0)
                .getJSONObject("message")
                .getString("content");
            TradeUtils.log("askDeepseek Réponse Deepseek: " + res);
            AgentEntity gpt = new AgentEntity(Agent.DEEPSEEK.getName(), LocalDateTime.now(), prompt, res);
            deepseekRepository.save(gpt);
            return new AgentResponse(gpt.getId(), res, null);
        } catch (Exception e) {
            return new AgentResponse(null, null, "Erreur d'appel à l'API Deepseek : " + e.getMessage());
        }
    }
}
