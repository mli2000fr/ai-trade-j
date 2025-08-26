package com.app.backend.trade.service;

import com.app.backend.trade.model.ChatGptResponse;
import com.app.backend.trade.util.TradeUtils;
import com.app.backend.trade.model.DataAction;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;
import java.util.HashMap;
import java.util.Map;

@Service
public class ChatGptService {
    @Value("${openai.api.key}")
    private String apiKey;

    @Value("${openai.api.url}")
    private String apiUrl;

    @Value("${openai.api.model}")
    private String apiModel;

    @Value("${openai.api.prompt.version.analyse.action}")
    private String promptVersionAnalyseAction;

    @Value("${openai.api.prompt.version.analyse.id}")
    private String promptVersionAnalyseId;



    @Value("${openai.api.prompt.version.test.action}")
    private String promptVersionTestAction;

    @Value("${openai.api.prompt.version.test.id}")
    private String promptVersionTestId;

    public ChatGptResponse askChatGpt(String prompt) {
        RestTemplate restTemplate = new RestTemplate();
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setBearerAuth(apiKey);

        Map<String, Object> message = new HashMap<>();
        message.put("role", "user");
        message.put("content", prompt);

        Map<String, Object> body = new HashMap<>();
        body.put("model", apiModel);
        body.put("messages", new Object[]{message});

        HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);
        try {
            ResponseEntity<Map> response = restTemplate.postForEntity(apiUrl, request, Map.class);
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                Object choices = response.getBody().get("choices");
                if (choices instanceof java.util.List && !((java.util.List<?>) choices).isEmpty()) {
                    Object first = ((java.util.List<?>) choices).get(0);
                    if (first instanceof Map) {
                        Object messageObj = ((Map<?, ?>) first).get("message");
                        if (messageObj instanceof Map) {
                            Object content = ((Map<?, ?>) messageObj).get("content");
                            if (content != null) {
                                return new ChatGptResponse(content.toString(), null);
                            }
                        }
                    }
                }
            }
            return new ChatGptResponse(null, "Erreur lors de la communication avec ChatGPT.");
        } catch (RestClientException e) {
            return new ChatGptResponse(null, "Erreur d'appel à l'API OpenAI : " + e.getMessage());
        }
    }

    // Analyse une action en interrogeant Alpha Vantage puis ChatGPT avec un délai spécifique
    public String test(String message) {


        Map<String, Object> variables = new HashMap<>();
        variables.put("message", message);
        Map<String, Object> promptTemplate = new HashMap<>();
        promptTemplate.put("id", promptVersionTestId);
        promptTemplate.put("version", promptVersionTestAction);
        promptTemplate.put("variables", variables);
        String prompt = promptTemplate.toString();
        TradeUtils.log("Prompt JSON : " + prompt);

        ChatGptResponse response = askChatGpt(prompt);
        if (response.getError() != null) {
            return "Erreur lors de l'analyse : " + response.getError();
        }
        return response.getMessage();
    }

    // Analyse une action en interrogeant Alpha Vantage puis ChatGPT avec un délai spécifique
    public String getAnalyseAction(DataAction dataAction) {

        Map<String, Object> variables = getStringObjectMap(dataAction);
        Map<String, Object> promptTemplate = new HashMap<>();
        promptTemplate.put("id", promptVersionAnalyseId);
        promptTemplate.put("version", promptVersionAnalyseAction);
        promptTemplate.put("variables", variables);
        String prompt = promptTemplate.toString();
        TradeUtils.log("Prompt JSON : " + prompt);

        ChatGptResponse response = askChatGpt(prompt);
        if (response.getError() != null) {
            return "Erreur lors de l'analyse : " + response.getError();
        }
        return response.getMessage();
    }

    private static Map<String, Object> getStringObjectMap(DataAction dataAction) {
        Map<String, Object> variables = new HashMap<>();
        variables.put("symbole", dataAction.getSymbol());
        variables.put("data_value_daily", dataAction.getData());
        variables.put("delai_mois", dataAction.getDelai());
        variables.put("data_sma", dataAction.getSma());
        variables.put("data_rsi", dataAction.getRsi());
        variables.put("data_macd", dataAction.getMacd());
        variables.put("data_atr", dataAction.getAtr());
        variables.put("data_financial", dataAction.getFinancial());
        variables.put("data_statistics", dataAction.getStatistics());
        variables.put("data_earnings", dataAction.getEarnings());
        variables.put("data_montant", dataAction.getMontant());
        variables.put("data_news", dataAction.getNews());
        return variables;
    }
}
