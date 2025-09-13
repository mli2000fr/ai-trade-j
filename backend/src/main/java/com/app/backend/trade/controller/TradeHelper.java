package com.app.backend.trade.controller;

import com.app.backend.trade.exception.DayTradingException;
import com.app.backend.trade.model.*;
import com.app.backend.trade.model.alpaca.Order;
import com.app.backend.trade.service.*;
import com.app.backend.trade.util.TradeUtils;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Controller;
import org.ta4j.core.BarSeries;
import org.ta4j.core.indicators.ATRIndicator;
import org.ta4j.core.indicators.EMAIndicator;
import org.ta4j.core.indicators.MACDIndicator;
import org.ta4j.core.indicators.RSIIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;

import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;


@Controller
public class TradeHelper {
    @Value("${trade.type}")
    private String tradeType;

    private final AlpacaService alpacaService;
    private final ChatGptService chatGptService;
    private final TwelveDataService twelveDataService;
    private final FinnhubService finnhubService;
    private final EodhdService eodhdService;
    private final StrategyService strategyService;
    private final CompteService compteService;
    private final JdbcTemplate jdbcTemplate;

    @Autowired
    public TradeHelper(AlpacaService alpacaService,
                       ChatGptService chatGptService,
                       FinnhubService finnhubService,
                       TwelveDataService twelveDataService,
                       EodhdService eodhdService,
                       StrategyService strategyService,
                       CompteService compteService,
                       JdbcTemplate jdbcTemplate) {
        this.alpacaService = alpacaService;
        this.chatGptService = chatGptService;
        this.twelveDataService = twelveDataService;
        this.finnhubService = finnhubService;
        this.eodhdService = eodhdService;
        this.strategyService = strategyService;
        this.compteService = compteService;
        this.jdbcTemplate = jdbcTemplate;
    }

    /**
     * Récupère le portefeuille du compte donné.
     * @param compte CompteEntity
     * @return Portfolio
     */
    public Portfolio getPortfolio(CompteEntity compte) {
        return alpacaService.getPortfolio(compte);
    }

    /**
     * Vérifie la validité d'une liste de symboles (séparés par virgule).
     * @param symbols Liste de symboles sous forme de chaîne
     */
    public void isSymbolsValid(String symbols) {
        for (String symbol : symbols.split(",")) {
            symbol = symbol.trim();
            if (symbol.isEmpty()) continue;
            this.isAssetSymbolEligible(symbol);
        }
    }

    /**
     * Vérifie si un symbole est éligible (présent et actif en base).
     * @param symbol Symbole à vérifier
     */
    public void isAssetSymbolEligible(String symbol) {
        String sql = "SELECT COUNT(*) FROM alpaca_asset WHERE symbol = ? AND eligible = true";
        Integer count = jdbcTemplate.queryForObject(sql, new Object[]{symbol}, Integer.class);
        if (count == null || count == 0) {
            throw new RuntimeException("Le symbole n'est pas valide ou inactif : " + symbol);
        }
    }

    public String getPromptAnalyseSymbol(String compteId, String symbols) {
        CompteEntity compte = compteService.getCompteCredentialsById(compteId);
        String prompt = TradeUtils.readResourceFile("trade/prompt/prompt_analyse_stocks.txt");
        return prompt.replace("{{symbols}}", symbols);
    }


    /**
     * Exécute le trading automatique IA pour une liste de symboles.
     * @param compte CompteEntity
     * @param symbols Liste de symboles
     * @param analyseGpt Analyse IA optionnelle
     * @return ReponseAuto
     */
    public ReponseAuto tradeAIAuto(CompteEntity compte, List<String> symbols, String analyseGpt) {

        if (symbols == null || symbols.isEmpty()) {
            throw new RuntimeException("Aucun symbole fourni pour tradeAIAuto.");
        }
        String joinedSymbols = String.join(",", symbols);
        this.isSymbolsValid(joinedSymbols);
        String promptEntete = TradeUtils.readResourceFile("trade/prompt/prompt_" + tradeType + "_trade_auto_entete.txt");
        String promptPied = TradeUtils.readResourceFile("trade/prompt/prompt_" + tradeType + "_trade_auto_pied.txt")
                .replace("{{data_analyse_ia}}", (analyseGpt != null && !analyseGpt.isBlank()) ? analyseGpt : "not available");
        String promptSymbol = TradeUtils.readResourceFile("trade/prompt/prompt_trade_auto_symbol.txt");
        String portfolioJson = new Gson().toJson(this.getPortfolio(compte));
        String newsGenerals = "No information found";
        try {
            Map<String, Object> newsMap = alpacaService.getRecentNews(compte, 50);
            if (newsMap != null && newsMap.containsKey("news")) {
                newsGenerals = new Gson().toJson(newsMap.get("news"));
            }
        } catch (Exception e) {
            TradeUtils.log("Error getRecentNews: " + e.getMessage());
        }
        StringBuilder promptFinal = new StringBuilder(promptEntete.replace("{{symbols}}", joinedSymbols)
                .replace("{{data_portfolio}}", portfolioJson)
                .replace("{{data_general_news}}", newsGenerals));

        for (String symbol : symbols) {
            symbol = symbol.trim();
            if (symbol.isEmpty()) continue;
            InfosAction infosAction = this.getInfosAction(compte, symbol, true);
            Map<String, Object> variables = TradeUtils.getStringObjectMap(infosAction); // Utilitaire déplacé
            promptFinal.append(getPromptWithValues(promptSymbol, variables));
            sleepForRateLimit();
        }
        promptFinal.append(promptPied);

        ChatGptResponse response = chatGptService.askChatGpt(String.valueOf(compte.getId()), promptFinal.toString());
        if (response.getError() != null) {
            throw new RuntimeException("Erreur lors de l'analyse : " + response.getError());
        }
        return processAIAuto(compte, response);

    }

    /**
     * Traite la réponse IA et prépare la structure ReponseAuto.
     * @param compte CompteEntity
     * @param response Réponse IA
     * @return ReponseAuto
     */
    private ReponseAuto processAIAuto(CompteEntity compte, ChatGptResponse response) {
        String[] parts = response.getMessage() != null ? response.getMessage().split("===") : new String[0];
        String orders = parts.length > 0 ? parts[0].trim() : "";
        String analyseGpt = parts.length > 1 ? parts[1].trim() : "";
        try {
            Type listType = new TypeToken<List<OrderRequest>>() {
            }.getType();
            List<OrderRequest> listOrders = new Gson().fromJson(orders, listType);
            for (OrderRequest order : listOrders) {
                if (order.getQuantity() != null && order.getQuantity() != 0) {
                    // on ne fait pas de trade journalier si on a déjà une position ouverte, si on veut le forcer, passer par trade manuel
                    OppositionOrder oppositionOrder = alpacaService.hasOppositeOrOpenOrder(compte, order.symbol, order.side);
                    order.setOppositionOrder(oppositionOrder);
                    if ((order.getPrice_limit() == null || order.getPrice_limit() == 0)
                            && order.getQuantity() != null && order.getQuantity() > 0) {
                        order.setPrice_limit(alpacaService.getLastPrice(compte, order.symbol));
                    }
                    if (oppositionOrder.isDayTrading()) {
                        order.setStatut("SKIPPED_DAYTRADE");
                        order.setExecuteNow(false);
                    }
                }
            }
            ReponseAuto ra = ReponseAuto.builder().idGpt(response.getIdGpt()).analyseGpt(analyseGpt).orders(listOrders).build();
            return ra;
        } catch (Exception e) {
            throw new RuntimeException(response.getMessage() + "Erreur de parsing de l'ordre : " + e.getMessage());
        }
    }

    /**
     * Exécute une liste d'ordres pour le compte sélectionné.
     * @param compte CompteEntity
     * @param idGpt identifiant GPT
     * @param orders liste d'ordres
     * @return liste d'ordres mise à jour
     */
    public List<OrderRequest> processOrders(CompteEntity compte, String idGpt, List<OrderRequest> orders) {
        if (orders == null) return null;
        for (OrderRequest order : orders) {
            order.normalize();
            if (isOrderValid(order) && order.isExecuteNow()) {
                boolean isSell = "sell".equals(order.side);
                try {
                    Order orderR = alpacaService.placeOrder(compte, order.symbol, order.qty, order.side, isSell ? null : order.priceLimit, isSell ? null : order.stopLoss, isSell ? null : order.takeProfit, idGpt, true, false);
                    order.setStatut(orderR.getStatus());
                } catch (DayTradingException e) {
                    order.setStatut("FAILED_DAYTRADE");
                } catch (Exception e) {
                    order.setStatut("FAILED");
                }
            }
        }
        return orders;
    }

    /**
     * Vérifie la validité d'un ordre.
     * @param order OrderRequest
     * @return true si valide
     */
    private boolean isOrderValid(OrderRequest order) {
        return order != null && order.symbol != null && order.qty != null && order.qty > 0 && order.side != null;
    }

    /**
     * Temporisation pour respecter le rate limit API.
     */
    private void sleepForRateLimit() {
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Thread interrompu pendant la temporisation de 61s", e);
        }
    }

    /**
     * Remplace les variables dans le template de prompt.
     * @param promptTemplate template
     * @param variables map clé/valeur
     * @return prompt final
     */
    private String getPromptWithValues(String promptTemplate, Map<String, Object> variables) {
        String prompt = promptTemplate;
        for (Map.Entry<String, Object> entry : variables.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();
            prompt = prompt.replace("{{" + key + "}}", value != null ? value.toString() : "");
        }
        if (prompt.contains("{{")) {
            throw new RuntimeException("Attention, certaines variables n'ont pas été remplacées dans le prompt.");
        }
        return prompt;
    }

    /**
     * Récupère les informations d'une action (symbol).
     * @param compte CompteEntity
     * @param symbol symbole
     * @param withPortfolio inclure le portefeuille
     * @return InfosAction
     */
    private InfosAction getInfosAction(CompteEntity compte, String symbol, boolean withPortfolio) {
        String portfolioJson = null;
        if (withPortfolio) {
            Portfolio portfolio = this.getPortfolio(compte);
            portfolioJson = new Gson().toJson(portfolio);
        }
        Double lastPrice = alpacaService.getLastPrice(compte, symbol);
        String historical = alpacaService.getHistoricalBarsJson(symbol, 200);
        String ema20 = twelveDataService.getEMA20(symbol);
        String ema50 = twelveDataService.getEMA50(symbol);
        String sma200 = twelveDataService.getSMA200(symbol);
        String rsi = twelveDataService.getRSI(symbol);
        String macd = twelveDataService.getMACD(symbol);
        String atr = twelveDataService.getATR(symbol);
        String financial = finnhubService.getFinancialData(symbol);
        String statistics = finnhubService.getDefaultKeyStatistics(symbol);
        String earnings = finnhubService.getEarnings(symbol);
        String news = "No information found";
        try {
            Map<String, Object> newsMap = alpacaService.getDetailedNewsForSymbol(compte, symbol, null);
            if (newsMap != null && newsMap.containsKey("news")) {
                news = new Gson().toJson(newsMap.get("news"));
            }
        } catch (Exception e) {
            TradeUtils.log("Error getNews(" + symbol + "): " + e.getMessage());
        }
        return new InfosAction(
                lastPrice,
                symbol,
                historical,
                ema20,
                ema50,
                sma200,
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

    public List<Double> getLastEMA20(List<DailyValue> values, int historique) {
        String indicateurName = "ema20";
        BarSeries series = TradeUtils.mapping(values);
        EMAIndicator ema = new EMAIndicator(new ClosePriceIndicator(series), 20);
        int count = series.getBarCount();
        List<Double> result = new java.util.ArrayList<>();
        for (int i = Math.max(0, count - historique); i < count; i++) {
            result.add(ema.getValue(i).doubleValue());
        }
        return result;
    }

    public List<Double> getLastEMA50(List<DailyValue> values, int historique) {
        String indicateurName = "ema50";
        BarSeries series = TradeUtils.mapping(values);
        EMAIndicator ema = new EMAIndicator(new ClosePriceIndicator(series), 50);
        int count = series.getBarCount();
        List<Double> result = new java.util.ArrayList<>();
        for (int i = Math.max(0, count - historique); i < count; i++) {
            result.add(ema.getValue(i).doubleValue());
        }
        return result;
    }

    public List<Double> getLastEMA200(List<DailyValue> values, int historique) {
        String indicateurName = "ema200";
        BarSeries series = TradeUtils.mapping(values);
        EMAIndicator ema = new EMAIndicator(new ClosePriceIndicator(series), 200);
        int count = series.getBarCount();
        List<Double> result = new java.util.ArrayList<>();
        for (int i = Math.max(0, count - historique); i < count; i++) {
            result.add(ema.getValue(i).doubleValue());
        }
        return result;
    }

    public List<Double> getLastRSI(List<DailyValue> values, int historique) {
        String indicateurName = "rsi";
        BarSeries series = TradeUtils.mapping(values);
        RSIIndicator rsi = new RSIIndicator(new ClosePriceIndicator(series), 14);
        int count = series.getBarCount();
        List<Double> result = new java.util.ArrayList<>();
        for (int i = Math.max(0, count - historique); i < count; i++) {
            result.add(rsi.getValue(i).doubleValue());
        }
        return result;
    }

    public List<Double> getLastMACD(List<DailyValue> values, int historique) {
        String indicateurName = "macd";
        BarSeries series = TradeUtils.mapping(values);
        MACDIndicator macd = new MACDIndicator(new ClosePriceIndicator(series), 12, 26);
        int count = series.getBarCount();
        List<Double> result = new java.util.ArrayList<>();
        for (int i = Math.max(0, count - historique); i < count; i++) {
            result.add(macd.getValue(i).doubleValue());
        }
        return result;
    }

    public List<Double> getLastATR(List<DailyValue> values, int historique) {
        String indicateurName = "atr";
        BarSeries series = TradeUtils.mapping(values);
        ATRIndicator atr = new ATRIndicator(series, 14);
        int count = series.getBarCount();
        List<Double> result = new java.util.ArrayList<>();
        for (int i = Math.max(0, count - historique); i < count; i++) {
            result.add(atr.getValue(i).doubleValue());
        }
        return result;
    }
}
