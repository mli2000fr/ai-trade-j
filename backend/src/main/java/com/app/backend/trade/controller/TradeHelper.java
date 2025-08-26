package com.app.backend.trade.controller;

import com.app.backend.trade.model.ChatGptResponse;
import com.app.backend.trade.model.DataAction;
import com.app.backend.trade.model.Portfolio;
import com.app.backend.trade.service.AlpacaService;
import com.app.backend.trade.service.ChatGptService;
import com.app.backend.trade.util.TradeUtils;
import com.google.gson.Gson;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;

import java.util.HashMap;
import java.util.Map;

@Controller
public class TradeHelper {

    @Value("${openai.api.prompt.version.analyse.action}")
    private String promptVersionAnalyseAction;

    @Value("${openai.api.prompt.version.analyse.id}")
    private String promptVersionAnalyseId;


    @Value("${openai.api.prompt.version.trade.action}")
    private String promptVersionTradeAction;

    @Value("${openai.api.prompt.version.trade.id}")
    private String promptVersionTradeId;

    private final AlpacaService alpacaService;
    private final ChatGptService chatGptService;

    @Autowired
    public TradeHelper(AlpacaService alpacaService,
                       ChatGptService chatGptService) {
        this.alpacaService = alpacaService;
        this.chatGptService = chatGptService;
    }
    public Portfolio getPortfolio() {
        Portfolio portfolio = alpacaService.getPortfolio();
        return portfolio;
    }

    // Analyse une action en interrogeant Alpha Vantage puis ChatGPT avec un délai spécifique
    public String getAnalyseAction(DataAction dataAction) {

        String promptTemplate = TradeUtils.readResourceFile("prompt/prompt_analyse.json");
        Map<String, Object> variables = getStringObjectMap(dataAction);

        String prompt = promptTemplate;
        for (Map.Entry<String, Object> entry : variables.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();
            prompt = prompt.replace("{{" + key + "}}", value != null ? value.toString() : "");
        }
        TradeUtils.log("Prompt JSON : " + prompt);

        ChatGptResponse response = chatGptService.askChatGpt(prompt);
        if (response.getError() != null) {
            return "Erreur lors de l'analyse : " + response.getError();
        }
        return response.getMessage();
    }

    public String trade(DataAction dataAction, Portfolio portfolio) {
        // Lecture du prompt depuis le fichier
        String promptTemplate = TradeUtils.readResourceFile("prompt/prompt_trade.json");

        Map<String, Object> variables = getStringObjectMap(dataAction);
        // Conversion du portefeuille en JSON avec Gson
        String portfolioJson = new Gson().toJson(portfolio);
        variables.put("data_portefeuille", portfolioJson);
        // Remplacement des variables {{...}} par leur valeur
        String prompt = promptTemplate;
        for (Map.Entry<String, Object> entry : variables.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();
            prompt = prompt.replace("{{" + key + "}}", value != null ? value.toString() : "");
        }
        TradeUtils.log("Prompt JSON : " + prompt);
        ChatGptResponse response = chatGptService.askChatGpt(prompt);
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
