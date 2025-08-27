package com.app.backend.trade.controller;

import com.app.backend.trade.model.ChatGptResponse;
import com.app.backend.trade.model.InfosAction;
import com.app.backend.trade.model.Portfolio;
import com.app.backend.trade.model.alpaca.Order;
import com.app.backend.trade.service.*;
import com.app.backend.trade.util.TradeUtils;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Controller
public class TradeHelper {

    private final AlpacaService alpacaService;
    private final ChatGptService chatGptService;
    private final TwelveDataService twelveDataService;
    private final FinnhubService finnhubService;
    private final EodhdService eodhdService;

    @Autowired
    public TradeHelper(AlpacaService alpacaService,
                       ChatGptService chatGptService,
                       FinnhubService finnhubService,
                       TwelveDataService twelveDataService,
                       EodhdService eodhdService
                       ) {
        this.alpacaService = alpacaService;
        this.chatGptService = chatGptService;
        this.twelveDataService = twelveDataService;
        this.finnhubService = finnhubService;
        this.eodhdService = eodhdService;
    }
    public Portfolio getPortfolio() {
        Portfolio portfolio = alpacaService.getPortfolio();
        return portfolio;
    }

    // Classe interne pour parser la réponse de l'IA
    private static class OrderRequest {
        String symbol;
        Integer qty;
        Integer quantity;
        String side;
        String action;
        Double priceLimit;
        Double price_limit;
        Double stopLoss;
        Double stop_loss;
        Double takeProfit;
        Double take_profit;

        // Méthode utilitaire pour normaliser les champs
        void normalize() {
            if (side == null && action != null) side = action;
            if (qty == null && quantity != null) qty = quantity;
            if (priceLimit == null && price_limit != null) priceLimit = price_limit;
            if (stopLoss == null && stop_loss != null) stopLoss = stop_loss;
            if (takeProfit == null && take_profit != null) takeProfit = take_profit;
        }
    }

    // Analyse une action en interrogeant Alpha Vantage puis ChatGPT avec un délai spécifique
    public String getAnalyseAction(String symbol) throws Exception {

        String promptTemplate = TradeUtils.readResourceFile("prompt/prompt_analyse.json");

        InfosAction infosAction = this.getInfosAction(symbol,true);

        Map<String, Object> variables = getStringObjectMap(infosAction);

        String prompt = getPromptWithValues(promptTemplate, variables);

        ChatGptResponse response = chatGptService.askChatGpt(prompt);
        if (response.getError() != null) {
            return "Erreur lors de l'analyse : " + response.getError();
        }

        return response.getMessage();
    }

    public String tradeAIAuto(List<String> symbols) throws Exception {
        if(symbols == null || symbols.isEmpty()){
            throw new Exception("Aucun symbole fourni pour tradeAIAuto.");
        }

        // Lecture du prompt depuis le fichier
        String promptEntete = TradeUtils.readResourceFile("prompt/prompt_trade_auto_entete.json");
        String promptPied = TradeUtils.readResourceFile("prompt/prompt_trade_auto_pied.json");
        String promptSymbol = TradeUtils.readResourceFile("prompt/prompt_trade_auto_symbol.json");

        Portfolio portfolio = this.getPortfolio();
        String portfolioJson = new Gson().toJson(portfolio);

        String joinedSymbols = String.join(",", symbols);
        StringBuilder promptFinal = new StringBuilder(promptEntete.replace("{{symbols}}", joinedSymbols)
                .replace("{{data_portfolio}}", portfolioJson));

        for(String symbol : symbols){
            symbol = symbol.trim();
            if(symbol.isEmpty()){
                continue;
            }
            InfosAction infosAction = this.getInfosAction(symbol, true);

            Map<String, Object> variables = getStringObjectMap(infosAction);

            promptFinal.append(getPromptWithValues(promptSymbol, variables));
            try {
                Thread.sleep(61000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Thread interrompu pendant la temporisation de 61s", e);
            }
        }
        promptFinal.append(promptPied);
        ChatGptResponse response = chatGptService.askChatGpt(promptFinal.toString());

        if (response.getError() != null) {
            return "Erreur lors de l'analyse : " + response.getError();
        }
        String[] parts = response.getMessage() != null ? response.getMessage().split("===") : new String[0];
        String responseOrder = parts.length > 0 ? parts[0].trim() : "";
        try {
            Type listType = new TypeToken<List<OrderRequest>>(){}.getType();
            List<OrderRequest> orders = new Gson().fromJson(responseOrder, listType);
            if (orders != null) {
                for (OrderRequest order : orders) {
                    order.normalize();
                    if (order != null && order.symbol != null && order.qty != null && order.qty > 0 && order.side != null) {
                        Order epageorder = alpacaService.placeOrder(order.symbol, order.qty, order.side, order.priceLimit, order.stopLoss, order.takeProfit);
                    }
                }
            }
        } catch (Exception e) {
            return "Erreur de parsing de l'ordre : " + e.getMessage();
        }


        return response.getMessage();
    }

    public String tradeAI(String symbol) throws Exception {

        // Lecture du prompt depuis le fichier
        String promptTemplate = TradeUtils.readResourceFile("prompt/prompt_trade.json");

        InfosAction infosAction = this.getInfosAction(symbol, true);

        Map<String, Object> variables = getStringObjectMap(infosAction);

        String prompt = getPromptWithValues(promptTemplate, variables);

        ChatGptResponse response = chatGptService.askChatGpt(prompt);
        if (response.getError() != null) {
            return "Erreur lors de l'analyse : " + response.getError();
        }
        String[] parts = response.getMessage() != null ? response.getMessage().split("===") : new String[0];
        String responseOrder = parts.length > 0 ? parts[0].trim() : "";
        OrderRequest order;
        try {
            order = new Gson().fromJson(responseOrder, OrderRequest.class);
            order.normalize();
        } catch (Exception e) {
            return "Erreur de parsing de l'ordre : " + e.getMessage();
        }
        if (order != null && order.symbol != null && order.qty != null && order.qty > 0 && order.side != null) {
            Order orderAlpaca = alpacaService.placeOrder(order.symbol, order.qty, order.side, order.priceLimit, order.stopLoss, order.takeProfit);
        }
        return response.getMessage();
    }

    private String getPromptWithValues(String promptTemplate,  Map<String, Object> variables) throws Exception {
        String prompt = promptTemplate;
        for (Map.Entry<String, Object> entry : variables.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();
            prompt = prompt.replace("{{" + key + "}}", value != null ? value.toString() : "");
        }
        TradeUtils.log("Prompt JSON : " + prompt);
        if(prompt.indexOf("{{") != -1){
            throw new Exception("Attention, certaines variables n'ont pas été remplacées dans le prompt.");
        }
        return prompt;
    }

    private InfosAction getInfosAction(String symbol, boolean withPortfolio) throws Exception {
        String portfolioJson = null;
        if(withPortfolio){
            Portfolio portfolio = this.getPortfolio();
            portfolioJson = new Gson().toJson(portfolio);
        }
        String data = twelveDataService.getDataAction(symbol);
        String sma = twelveDataService.getSMA(symbol);
        String rsi = twelveDataService.getRSI(symbol);
        String macd = twelveDataService.getMACD(symbol);
        String atr = twelveDataService.getATR(symbol);
        String financial = finnhubService.getFinancialData(symbol);
        String statistics = finnhubService.getDefaultKeyStatistics(symbol);
        String earnings = finnhubService.getEarnings(symbol);
        String news = eodhdService.getNews(symbol);

        InfosAction info = new InfosAction(
                symbol,
                data,
                sma,
                rsi,
                macd,
                atr,
                financial,
                statistics,
                earnings,
                news,
                portfolioJson
        );
        return info;
    }

    private static Map<String, Object> getStringObjectMap(InfosAction infosAction) {
        Map<String, Object> variables = new HashMap<>();
        variables.put("symbol", infosAction.getSymbol());
        variables.put("data_value_daily", infosAction.getData());
        variables.put("data_sma", infosAction.getSma());
        variables.put("data_rsi", infosAction.getRsi());
        variables.put("data_macd", infosAction.getMacd());
        variables.put("data_atr", infosAction.getAtr());
        variables.put("data_financial", infosAction.getFinancial());
        variables.put("data_statistics", infosAction.getStatistics());
        variables.put("data_earnings", infosAction.getEarnings());
        variables.put("data_news", infosAction.getNews());
        variables.put("data_portfolio", infosAction.getPortfolio());
        return variables;
    }
}
