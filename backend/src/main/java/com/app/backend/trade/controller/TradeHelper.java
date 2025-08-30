package com.app.backend.trade.controller;

import com.app.backend.trade.model.ChatGptResponse;
import com.app.backend.trade.model.CompteEntity;
import com.app.backend.trade.model.InfosAction;
import com.app.backend.trade.model.Portfolio;
import com.app.backend.trade.service.*;
import com.app.backend.trade.util.TradeUtils;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Helper métier pour la gestion des trades automatiques et IA.
 * Fournit des méthodes pour orchestrer les prompts, valider les symboles, et exécuter des ordres via Alpaca.
 */
@Controller
public class TradeHelper {
    @Value("${trade.type}")
    private String tradeType;

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
                       EodhdService eodhdService) {
        this.alpacaService = alpacaService;
        this.chatGptService = chatGptService;
        this.twelveDataService = twelveDataService;
        this.finnhubService = finnhubService;
        this.eodhdService = eodhdService;
    }

    /**
     * Récupère le portefeuille d'un compte.
     */
    public Portfolio getPortfolio(CompteEntity compte) {
        return alpacaService.getPortfolio(compte);
    }

    /**
     * Vérifie la validité d'une liste de symboles via ChatGPT.
     * @param symbols liste des symboles séparés par des virgules
     * @return true si tous les symboles sont valides, exception sinon
     */
    public boolean isSymbolsValid(String symbols)  {
        String prompt = TradeUtils.readResourceFile("prompt/prompt_check_symbol.txt")
                .replace("{{symbols}}", symbols);
        ChatGptResponse response = chatGptService.askChatGpt(prompt);
        if (response.getError() != null) {
            throw new RuntimeException("Erreur lors de l'analyse : " + response.getError());
        }
        try {
            boolean valid = Boolean.parseBoolean(response.getMessage());
            if (!valid) {
                throw new RuntimeException("Les symboles ne sont pas valides : " + response.getMessage());
            }
            return true;
        } catch (Exception e) {
            throw new RuntimeException("check symboles : " + response.getMessage());
        }
    }

    /**
     * Exécute un trade automatique via l'IA sur une liste de symboles.
     * @param compte compte utilisateur
     * @param symbols liste des symboles
     * @return message de retour de l'IA ou erreur
     */
    public String tradeAIAuto(CompteEntity compte, List<String> symbols)  {
        if(symbols == null || symbols.isEmpty()){
            throw new RuntimeException("Aucun symbole fourni pour tradeAIAuto.");
        }
        String joinedSymbols = String.join(",", symbols);
        this.isSymbolsValid(joinedSymbols);
        String promptEntete = TradeUtils.readResourceFile("prompt/prompt_"+tradeType+"_trade_auto_entete.txt");
        String promptPied = TradeUtils.readResourceFile("prompt/prompt_"+tradeType+"_trade_auto_pied.txt");
        String promptSymbol = TradeUtils.readResourceFile("prompt/prompt_trade_auto_symbol.txt");
        String portfolioJson = new Gson().toJson(this.getPortfolio(compte));
        StringBuilder promptFinal = new StringBuilder(promptEntete.replace("{{symbols}}", joinedSymbols)
                .replace("{{data_portfolio}}", portfolioJson));
        for(String symbol : symbols){
            symbol = symbol.trim();
            if(symbol.isEmpty()) continue;
            InfosAction infosAction = this.getInfosAction(compte, symbol, true);
            Map<String, Object> variables = getStringObjectMap(infosAction);
            promptFinal.append(getPromptWithValues(promptSymbol, variables));
            sleepForRateLimit();
        }
        promptFinal.append(promptPied);
        ChatGptResponse response = chatGptService.askChatGpt(promptFinal.toString());
        if (response.getError() != null) {
            return "Erreur lors de l'analyse : " + response.getError();
        }
        return processAIAutoOrders(response.getMessage(), compte);
    }

    /**
     * Exécute un trade automatique via l'IA sur une liste de symboles, avec analyse GPT optionnelle.
     * @param compte compte utilisateur
     * @param symbols liste des symboles
     * @param analyseGpt texte d'analyse GPT (optionnel)
     * @return message de retour de l'IA ou erreur
     */
    public String tradeAIAuto(CompteEntity compte, List<String> symbols, String analyseGpt)  {
        if(symbols == null || symbols.isEmpty()){
            throw new RuntimeException("Aucun symbole fourni pour tradeAIAuto.");
        }
        String joinedSymbols = String.join(",", symbols);
        this.isSymbolsValid(joinedSymbols);
        String promptEntete = TradeUtils.readResourceFile("prompt/prompt_"+tradeType+"_trade_auto_entete.txt");
        String promptPied = TradeUtils.readResourceFile("prompt/prompt_"+tradeType+"_trade_auto_pied.txt")
                .replace("{{data_analyse_ia}}", analyseGpt != null ? analyseGpt : "not available");
        String promptSymbol = TradeUtils.readResourceFile("prompt/prompt_trade_auto_symbol.txt");
        String portfolioJson = new Gson().toJson(this.getPortfolio(compte));
        StringBuilder promptFinal = new StringBuilder(promptEntete.replace("{{symbols}}", joinedSymbols)
                .replace("{{data_portfolio}}", portfolioJson));
        for(String symbol : symbols){
            symbol = symbol.trim();
            if(symbol.isEmpty()) continue;
            InfosAction infosAction = this.getInfosAction(compte, symbol, true);
            Map<String, Object> variables = getStringObjectMap(infosAction);
            promptFinal.append(getPromptWithValues(promptSymbol, variables));
            sleepForRateLimit();
        }
        promptFinal.append(promptPied);
        ChatGptResponse response = chatGptService.askChatGpt(promptFinal.toString());
        if (response.getError() != null) {
            return "Erreur lors de l'analyse : " + response.getError();
        }
        return processAIAutoOrders(response.getMessage(), compte);
    }

    /**
     * Exécute un trade IA sur un symbole donné.
     * @param compte compte utilisateur
     * @param symbol symbole à trader
     * @return message de retour de l'IA ou erreur
     */
    public String tradeAI(CompteEntity compte, String symbol)  {
        String promptTemplate = TradeUtils.readResourceFile("prompt/prompt_trade.txt");
        InfosAction infosAction = this.getInfosAction(compte, symbol, true);
        Map<String, Object> variables = getStringObjectMap(infosAction);
        String prompt = getPromptWithValues(promptTemplate, variables);
        ChatGptResponse response = chatGptService.askChatGpt(prompt);
        if (response.getError() != null) {
            return "Erreur lors de l'analyse : " + response.getError();
        }
        return processAIOrder(response.getMessage(), compte);
    }

    // --- Méthodes privées utilitaires ---

    /**
     * Parse et exécute les ordres retournés par l'IA (mode auto).
     */
    private String processAIAutoOrders(String message, CompteEntity compte) {
        String[] parts = message != null ? message.split("===") : new String[0];
        String responseOrder = parts.length > 0 ? parts[0].trim() : "";
        try {
            Type listType = new TypeToken<List<OrderRequest>>(){}.getType();
            List<OrderRequest> orders = new Gson().fromJson(responseOrder, listType);
            if (orders != null) {
                for (OrderRequest order : orders) {
                    order.normalize();
                    if (isOrderValid(order)) {
                        boolean isSell = "sell".equals(order.side);
                        alpacaService.placeOrder(compte, order.symbol, order.qty, order.side, isSell ? null : order.priceLimit, isSell ? null : order.stopLoss, isSell ? null : order.takeProfit);
                    }
                }
            }
        } catch (Exception e) {
            return message + "Erreur de parsing de l'ordre : " + e.getMessage();
        }
        return message;
    }

    /**
     * Parse et exécute l'ordre retourné par l'IA (mode simple).
     */
    private String processAIOrder(String message, CompteEntity compte) {
        String[] parts = message != null ? message.split("===") : new String[0];
        String responseOrder = parts.length > 0 ? parts[0].trim() : "";
        try {
            OrderRequest order = new Gson().fromJson(responseOrder, OrderRequest.class);
            order.normalize();
            if (isOrderValid(order)) {
                boolean isSell = "sell".equals(order.side);
                alpacaService.placeOrder(compte, order.symbol, order.qty, order.side, isSell ? null : order.priceLimit, isSell ? null : order.stopLoss, isSell ? null : order.takeProfit);
            }
        } catch (Exception e) {
            return "Erreur de parsing de l'ordre : " + e.getMessage();
        }
        return message;
    }

    /**
     * Vérifie la validité d'un objet OrderRequest.
     */
    private boolean isOrderValid(OrderRequest order) {
        return order != null && order.symbol != null && order.qty != null && order.qty > 0 && order.side != null;
    }

    /**
     * Temporisation pour respecter le rate limit de l'API OpenAI.
     */
    private void sleepForRateLimit() {
        try {
            Thread.sleep(61000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Thread interrompu pendant la temporisation de 61s", e);
        }
    }

    /**
     * Remplace les variables dans un template de prompt par leurs valeurs.
     */
    private String getPromptWithValues(String promptTemplate,  Map<String, Object> variables)  {
        String prompt = promptTemplate;
        for (Map.Entry<String, Object> entry : variables.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();
            prompt = prompt.replace("{{" + key + "}}", value != null ? value.toString() : "");
        }
        if(prompt.contains("{{")){
            throw new RuntimeException("Attention, certaines variables n'ont pas été remplacées dans le prompt.");
        }
        return prompt;
    }

    /**
     * Récupère toutes les infos nécessaires pour un symbole (prix, indicateurs, news, etc.).
     */
    private InfosAction getInfosAction(CompteEntity compte, String symbol, boolean withPortfolio)  {
        String portfolioJson = null;
        if(withPortfolio){
            Portfolio portfolio = this.getPortfolio(compte);
            portfolioJson = new Gson().toJson(portfolio);
        }
        Double lastPrice = alpacaService.getLastPrice(compte, symbol);
        String data = twelveDataService.getDataAction(symbol);
        String sma = twelveDataService.getSMA(symbol);
        String rsi = twelveDataService.getRSI(symbol);
        String macd = twelveDataService.getMACD(symbol);
        String atr = twelveDataService.getATR(symbol);
        String financial = finnhubService.getFinancialData(symbol);
        String statistics = finnhubService.getDefaultKeyStatistics(symbol);
        String earnings = finnhubService.getEarnings(symbol);
        String news = "No information found";
        try{
            Map<String, Object> newsMap = alpacaService.getDetailedNewsForSymbol(compte, symbol, null);
            if (newsMap != null && newsMap.containsKey("news")) {
                news = new Gson().toJson(newsMap.get("news"));
            }
        } catch (Exception e) {
            TradeUtils.log("Error getNews("+symbol+"): " + e.getMessage());
        }
        return new InfosAction(
                lastPrice,
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
    }

    /**
     * Construit la map de variables pour le prompt à partir d'un InfosAction.
     */
    private static Map<String, Object> getStringObjectMap(InfosAction infosAction) {
        Map<String, Object> variables = new HashMap<>();
        variables.put("symbol", infosAction.getSymbol());
        variables.put("data_price", infosAction.getLastPrice());
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

    // Classe interne pour parser la réponse de l'IA
    private static class OrderRequest {
        String symbol;
        Double qty;
        Double quantity;
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
}
