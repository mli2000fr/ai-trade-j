package com.example.backend;

import com.example.backend.model.DataAction;
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

    private static final String API_URL = "https://api.openai.com/v1/chat/completions";
    private static final String VERSION_TEMPLATE_PROMPT = "10";

    public ChatGptResponse askChatGpt(String prompt) {
        RestTemplate restTemplate = new RestTemplate();
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setBearerAuth(apiKey);

        Map<String, Object> message = new HashMap<>();
        message.put("role", "user");
        message.put("content", prompt);

        Map<String, Object> body = new HashMap<>();
        body.put("model", "gpt-5-mini");
        body.put("messages", new Object[]{message});

        HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);
        try {
            ResponseEntity<Map> response = restTemplate.postForEntity(API_URL, request, Map.class);
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

    public String callFunction(FunctionCallRequest req, AlphaVantageService alphaVantageService) {
        if ("getInfosAction".equals(req.getFunctionName())) {
            String symbol = req.getArguments().get("symbol");
            String functionName = req.getArguments().get("functionName");
            if (symbol == null || symbol.isEmpty()) {
                return "Argument 'symbol' manquant ou vide.";
            }
            Utils.log("Appel de la fonction getInfosAction avec le symbole : " + symbol + " et la fonction : " + functionName);

            if (functionName == null || functionName.isEmpty()) {
                functionName = "TIME_SERIES_INTRADAY"; // valeur par défaut
            }
            return alphaVantageService.getInfosAction(functionName, symbol, null);
        }
        return "Fonction inconnue : " + req.getFunctionName();
    }

    // Analyse une action en interrogeant Alpha Vantage puis ChatGPT
    public String getAnalyseAction(String symbol, AlphaVantageService alphaVantageService) {
        String data = alphaVantageService.getDataAction(symbol);
        String prompt = "Voici les données boursières TIME_SERIES_DAILY pour le symbole " + symbol + " :\n"
            + data + "\n\nAnalyse ces données et donne une synthèse sur la tendance, les points forts/faibles et un avis sur l'action.";
        ChatGptResponse response = askChatGpt(prompt);
        if (response.getError() != null) {
            return "Erreur lors de l'analyse : " + response.getError();
        }
        return response.getMessage();
    }

    public String getAnalyseActionWithAlphaVantage(String symbol, int delai, AlphaVantageService alphaVantageService){
        String data = alphaVantageService.getDataAction(symbol);
        String sma = alphaVantageService.getSMA(symbol);
        String rsi = alphaVantageService.getRSI(symbol);
        String macd = alphaVantageService.getMACD(symbol);
        String atr = alphaVantageService.getATR(symbol);
        String fundamental = alphaVantageService.getInfosAction("OVERVIEW", symbol, null);
        return getAnalyseAction( symbol,  delai,  data,  sma,  rsi,  macd,  atr, fundamental);
    }

    public String getAnalyseActionWithTwelveData(String symbol, int delai, TwelveDataService twelveDataService){
        String data = twelveDataService.getDataAction(symbol);
        String sma = twelveDataService.getSMA(symbol);
        String rsi = twelveDataService.getRSI(symbol);
        String macd = twelveDataService.getMACD(symbol);
        String atr = twelveDataService.getATR(symbol);
        String fundamental = twelveDataService.getFundamental(symbol);
        return getAnalyseAction( symbol,  delai,  data,  sma,  rsi,  macd,  atr, fundamental );
    }

    // Analyse une action en interrogeant Alpha Vantage puis ChatGPT avec un délai spécifique
    public String getAnalyseAction(String symbol, int delai, String data, String sma, String rsi, String macd, String atr, String fundamental) {

        Map<String, Object> variables = new HashMap<>();
        variables.put("symbole", symbol);
        variables.put("data_finance", data);
        variables.put("delai_mois", delai);
        variables.put("data_sma", sma);
        variables.put("data_rsi", rsi);
        variables.put("data_macd", macd);
        variables.put("data_atr", atr);
        variables.put("data_fundamental", fundamental);
        Map<String, Object> promptTemplate = new HashMap<>();
        promptTemplate.put("id", "pmpt_68a9a60f9e2081968f356abd0b036e710a1847164c537613");
        promptTemplate.put("version", VERSION_TEMPLATE_PROMPT);
        promptTemplate.put("variables", variables);
        String prompt = promptTemplate.toString();
        Utils.log("Prompt JSON : " + prompt);

        ChatGptResponse response = askChatGpt(prompt);
        if (response.getError() != null) {
            return "Erreur lors de l'analyse : " + response.getError();
        }
        return response.getMessage();
    }

    public String getAnalyseAction(DataAction dataAction) {

        Map<String, Object> variables = getStringObjectMap(dataAction);
        Map<String, Object> promptTemplate = new HashMap<>();
        promptTemplate.put("id", "pmpt_68a9a60f9e2081968f356abd0b036e710a1847164c537613");
        promptTemplate.put("version", VERSION_TEMPLATE_PROMPT);
        promptTemplate.put("variables", variables);
        String prompt = promptTemplate.toString();
        Utils.log("Prompt JSON : " + prompt);

        if(true){
            return "Prompt JSON : " + prompt;
        }

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
        return variables;
    }
}
