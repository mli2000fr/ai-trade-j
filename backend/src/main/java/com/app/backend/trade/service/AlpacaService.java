package com.app.backend.trade.service;

import com.app.backend.trade.exception.DayTradingException;
import com.app.backend.trade.model.*;
import com.app.backend.trade.model.alpaca.AlpacaAsset;
import com.app.backend.trade.model.alpaca.AlpacaTransferActivity;
import com.app.backend.trade.model.alpaca.OffsetDateTimeAdapter;
import com.app.backend.trade.model.alpaca.Order;
import com.app.backend.trade.repository.OrderRepository;
import com.app.backend.trade.util.TradeUtils;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.java_websocket.client.WebSocketClient;
import org.java_websocket.handshake.ServerHandshake;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.net.URI;
import java.time.OffsetDateTime;
import java.util.*;
import java.util.function.Consumer;

@Service
public class AlpacaService {
    private static final Logger logger = LoggerFactory.getLogger(AlpacaService.class);

    @Value("${alpaca.markets.api.paper.url}")
    private String apiBaseUrl;

    @Value("${alpaca.markets.api.market.url}")
    private String apiMarketBaseUrl;

    @Value("${alpaca.markets.api.order.limit}")
    private String limitOrders;

    @Value("${alpaca.markets.api.news.defaut.limit}")
    private String defaultLimitNews;

    private final RestTemplate restTemplate = new RestTemplate();
    private final OrderRepository orderRepository;
    private final CompteService compteService;

    private final Gson gson = new GsonBuilder()
            .registerTypeAdapter(OffsetDateTime.class, new OffsetDateTimeAdapter())
            .create();


    @Autowired
    public AlpacaService(OrderRepository orderRepository, CompteService compteService) {
        this.orderRepository = orderRepository;
        this.compteService = compteService;
    }

    /**
     * Place un ordre sur Alpaca (achat, vente, limit, bracket, etc.).
     * @param compte CompteEntity contenant les credentials
     * @param symbol Symbole de l'action
     * @param qty Quantité
     * @param side "buy" ou "sell"
     * @param priceLimit Limite de prix (optionnel)
     * @param stopLoss Stop loss (optionnel)
     * @param takeProfit Take profit (optionnel)
     * @return Order créé
     */
    public Order placeOrder(CompteEntity compte, String symbol, double qty, String side, Double priceLimit, Double stopLoss, Double takeProfit, String idGpt, boolean forceCancelOpposite, boolean forceDayTrade) {
        // Vérification anti-day trade centralisée
        OppositionOrder oppositionOrder = hasOppositeOrOpenOrder(compte, symbol, side);
        if (oppositionOrder.isOppositionFilled() && !forceDayTrade) {
            throw new DayTradingException("Erreur : Un ordre de vente existe déjà pour ce symbole. Veuillez attendre lendemain.");
        }else if (oppositionOrder.isOppositionActived()) {
            if(forceCancelOpposite){
                List<Order> annulables = this.getOrders(compte, symbol, true);
                for(Order ord : annulables) {
                    this.cancelOrder(compte, ord.getId());
                }
            } else {
                if("sell".equals(side)){
                    throw new DayTradingException("Erreur : Potential wash trade detected. Opposite side market/stop order exists. Veuillez annuler l'ordre existant ou activer l'option forceCancelOpposite.");
                }else if(!forceDayTrade){
                    throw new DayTradingException("Erreur : Un ordre opposé (achat/vente) est déjà ouvert pour ce symbole. Veuillez annuler l'ordre existant ou activer l'option forceCancelOpposite.");
                }

            }
        }else if(oppositionOrder.isSideActived()){
            List<Order> annulables = this.getOrders(compte, symbol, true);
            for(Order ord : annulables) {
                if(side.equals(ord.getSide())){
                    this.cancelOrder(compte, ord.getId());
                }
            }
        }

        String url = apiBaseUrl + "/orders";
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.set("APCA-API-KEY-ID", compte.getCle());
        headers.set("APCA-API-SECRET-KEY", compte.getSecret());

        Map<String, Object> body = new HashMap<>();
        body.put("symbol", symbol);
        body.put("side", side);
        // si qty est un entier, càd partie décimale = 0, on le passe en Integer pour Alpaca
        if (qty == Math.floor(qty)) {
            body.put("qty", (int) qty);
            body.put("time_in_force", "gtc");
        } else {
            body.put("qty", qty);
            body.put("time_in_force", "day");
        }

        if("sell".equals(side)){
            if(takeProfit != null){
                body.put("type", "limit");
                body.put("limit_price", takeProfit);
            }else if(stopLoss != null) {
                body.put("type", "stop");
                body.put("stop_price", stopLoss);
            }else{
                body.put("type", "market");
            }
        }else{
            // Gestion des différents types d'ordre
            if (priceLimit != null && (stopLoss == null && takeProfit == null)) {
                // Ordre limit simple
                body.put("type", "limit");
                body.put("limit_price", priceLimit);
            } else if (((stopLoss != null && stopLoss != 0) || (takeProfit != null && takeProfit != 0)) && qty == Math.floor(qty)) {
                // Bracket order (seulement si qty entier)
                body.put("type", "market"); // Alpaca exige type=market pour bracket
                body.put("order_class", "bracket");
                if (takeProfit != null && takeProfit != 0) {
                    Map<String, Object> tp = new HashMap<>();
                    tp.put("limit_price", takeProfit);
                    body.put("take_profit", tp);
                }
                if (stopLoss != null && stopLoss != 0) {
                    Map<String, Object> sl = new HashMap<>();
                    sl.put("stop_price", stopLoss);
                    body.put("stop_loss", sl);
                }
            } else {
                // Ordre market simple
                body.put("type", "market");
            }
        }

        HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);
        String requestJson = gson.toJson(body);
        String responseJson = null;
        String errorJson = null;
        Order order = null;
        try {
            order = this.callOrder(url, request);
            responseJson = gson.toJson(order);
        } catch (org.springframework.web.client.HttpClientErrorException ex) {
            errorJson = ex.getResponseBodyAsString();
            logger.error("Erreur HTTP lors de la création de l'ordre Alpaca: {}", errorJson);
        } catch (Exception ex) {
            errorJson = ex.getMessage();
            logger.error("Exception lors de la création de l'ordre Alpaca: {}", errorJson);
        }
        // Sauvegarde via repository JPA
        OrderEntity orderEntity = new OrderEntity(String.valueOf(compte.getId()), order == null ? null : order.getId(), side, requestJson, responseJson, errorJson, order == null ? "FAILED" : order.getStatus(), idGpt);
        orderRepository.save(orderEntity);
        if (order == null && errorJson != null) {
            // Tentative de parsing d'une erreur JSON Alpaca (ex: insufficient qty)
            if (errorJson != null && errorJson.trim().startsWith("{")) {
                // Vérification du code d'erreur Alpaca pour Pattern Day Trading
                if (errorJson.contains("40310100") && errorJson.contains("pattern day trading protection")) {
                    throw new DayTradingException("Trade refusé : protection Pattern Day Trading activée (trop de day trades, solde < 25 000 $)." );
                }
                OrderEntity orderEntity3 = new OrderEntity(String.valueOf(compte.getId()), null, side, requestJson, responseJson, errorJson, "FAILED", idGpt);
                orderRepository.save(orderEntity3);
            }
            throw new RuntimeException("Erreur lors de la création de l'ordre Alpaca: " +request.getBody()+ " / " + errorJson);
        }
        return order;
    }

    private Order callOrder(String url, HttpEntity<Map<String, Object>> request){
        String responseJson;
        ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);
        responseJson = response.getBody();
        return gson.fromJson(responseJson, Order.class);
    }

    /**
     * Récupère le dernier prix pour un symbole donné.
     */
    public Double getLastPrice(CompteEntity compte, String symbol) {
        // 1. Essayer quotes/latest
        String urlQuote = apiMarketBaseUrl + "/v2/stocks/" + symbol + "/quotes/latest";
        HttpHeaders headers = new HttpHeaders();
        headers.set("APCA-API-KEY-ID", compte.getCle());
        headers.set("APCA-API-SECRET-KEY", compte.getSecret());
        HttpEntity<Void> request = new HttpEntity<>(headers);
        try {
            ResponseEntity<Map> response = restTemplate.exchange(urlQuote, HttpMethod.GET, request, Map.class);
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                Map<String, Object> quote = (Map<String, Object>) response.getBody().get("quote");
                if (quote != null && quote.get("ap") != null) {
                    Double askPrice = Double.valueOf(quote.get("ap").toString());
                    if (askPrice != null && askPrice > 0) {
                        return askPrice;
                    }
                }
            }
        } catch (Exception e) {
            logger.warn("Erreur lors de l'appel à quotes/latest : {}", e.getMessage());
        }
        // 2. Essayer trades/latest
        String urlTrade = apiMarketBaseUrl + "/v2/stocks/" + symbol + "/trades/latest";
        try {
            ResponseEntity<Map> response = restTemplate.exchange(urlTrade, HttpMethod.GET, request, Map.class);
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                Map<String, Object> trade = (Map<String, Object>) response.getBody().get("trade");
                if (trade != null && trade.get("p") != null) {
                    Double lastTradePrice = Double.valueOf(trade.get("p").toString());
                    if (lastTradePrice != null && lastTradePrice > 0) {
                        return lastTradePrice;
                    }
                }
            }
        } catch (Exception e) {
            logger.warn("Erreur lors de l'appel à trades/latest : {}", e.getMessage());
        }
        // 3. Essayer bars (close de la dernière bougie)
        String urlBar = apiMarketBaseUrl + "/v2/stocks/" + symbol + "/bars?timeframe=1Min&limit=1";
        try {
            ResponseEntity<Map> response = restTemplate.exchange(urlBar, HttpMethod.GET, request, Map.class);
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                Object barsObj = response.getBody().get("bars");
                if (barsObj instanceof List) {
                    List bars = (List) barsObj;
                    if (!bars.isEmpty() && bars.get(0) instanceof Map) {
                        Map bar = (Map) bars.get(0);
                        if (bar.get("c") != null) {
                            Double closePrice = Double.valueOf(bar.get("c").toString());
                            if (closePrice != null && closePrice > 0) {
                                return closePrice;
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            logger.warn("Erreur lors de l'appel à bars : {}", e.getMessage());
        }
        // Si aucune source ne donne un prix valide
        return null;
    }

    /**
     * Stream les prix en temps réel via WebSocket.
     */
    public void streamLivePrice(CompteEntity compte, String symbol, Consumer<String> onMessage) {
        try {
            String wsUrl = "wss://stream.data.alpaca.markets/v2/sip";
            WebSocketClient client = new WebSocketClient(new URI(wsUrl)) {
                @Override
                public void onOpen(ServerHandshake handshakedata) {
                    // Authentification
                    String authMsg = String.format("{\"action\":\"auth\",\"key\":\"%s\",\"secret\":\"%s\"}", compte.getId(), compte.getSecret());
                    send(authMsg);
                    // Abonnement au flux de quotes
                    String subscribeMsg = String.format("{\"action\":\"subscribe\",\"quotes\":[\"%s\"]}", symbol);
                    send(subscribeMsg);
                }
                @Override
                public void onMessage(String message) {
                    if (onMessage != null) {
                        onMessage.accept(message);
                    }
                }
                @Override
                public void onClose(int code, String reason, boolean remote) {
                    // Optionnel : gestion de la fermeture
                }
                @Override
                public void onError(Exception ex) {
                    // Optionnel : gestion des erreurs
                }
            };
            client.connect();
        } catch (Exception e) {
            logger.error("Erreur lors de la connexion WebSocket Alpaca", e);
            throw new RuntimeException("Erreur lors de la connexion WebSocket Alpaca", e);
        }
    }

    private static final List<String> ACTIVE_ORDER_STATUSES = Arrays.asList(
        "new", "partially_filled", "accepted", "pending_new", "pending_replace", "pending_cancel"
    );

    // Vérifie s'il existe un ordre exécuté OU ouvert du côté opposé (buy/sell) pour ce symbole aujourd'hui (anti-day trade)
    public OppositionOrder hasOppositeOrOpenOrder(CompteEntity compte, String symbol, String side) {
        List<Order> orders = this.getOrders(compte, symbol, false);
        String oppositeSide = side.equalsIgnoreCase("buy") ? "sell" : "buy";
        java.time.LocalDate today = java.time.LocalDate.now();
        boolean oppositionFilled = false;
        boolean oppositionActived = false;
        boolean sideActived = false;
        for (Order o : orders) {
            if (o.getSide() != null && o.getSide().equalsIgnoreCase(oppositeSide)) {
                // Cas 1 : déjà exécuté aujourd'hui
                if (o.getStatus() != null && o.getStatus().equalsIgnoreCase("filled")
                        && o.getFilledAt() != null && o.getFilledAt().toLocalDate().isEqual(today)) {
                    oppositionFilled = true;
                }
                // Cas 2 : ordre ouvert aujourd'hui (potentiellement exécutable)
                if (o.getStatus() != null && ACTIVE_ORDER_STATUSES.contains(o.getStatus())) {
                    oppositionActived = true;
                }
            }else if(o.getSide().equalsIgnoreCase(side) && ACTIVE_ORDER_STATUSES.contains(o.getStatus())){
                sideActived = true;
            }
        }
        return OppositionOrder.builder().sideActived(sideActived).oppositionActived(oppositionActived).oppositionFilled(oppositionFilled).build();
    }

    /**
     * Vend une action pour un compte donné.
     */
    public String sellStock(CompteEntity compte, TradeRequest request) {

        if(request.getTakeProfit() != null){
            return placeOrder(compte, request.getSymbol(), request.getQuantity(), "sell", null, null, request.getTakeProfit(), null, request.isCancelOpposite(), request.isForceDayTrade()).toString();
        }
        if(request.getStopLoss() != null){
            return placeOrder(compte, request.getSymbol(), request.getQuantity(), "sell", null, request.getStopLoss(), null, null, request.isCancelOpposite(), request.isForceDayTrade()).toString();
        }
        return placeOrder(compte, request.getSymbol(), request.getQuantity(), "sell", null, null, null, null, request.isCancelOpposite(), request.isForceDayTrade()).toString();

    }

    /**
     * Achète une action pour un compte donné.
     */
    public String buyStock(CompteEntity compte, TradeRequest request) {
        return placeOrder(compte, request.getSymbol(), request.getQuantity(), "buy", null, request.getStopLoss(), request.getTakeProfit(), null, request.isCancelOpposite(), request.isForceDayTrade()).toString();
    }

    /**
     * Récupère le portefeuille et les positions pour un compte donné.
     */
    public PortfolioDto getPortfolioWithPositions(CompteEntity compte) {
        PortfolioDto dto = new PortfolioDto();

        // Récupérer les positions
        String positionsUrl = apiBaseUrl + "/positions";
        HttpHeaders headers = new HttpHeaders();
        headers.set("APCA-API-KEY-ID", compte.getCle());
        headers.set("APCA-API-SECRET-KEY", compte.getSecret());
        HttpEntity<Void> request = new HttpEntity<>(headers);
        ResponseEntity<Map[]> positionsResponse = restTemplate.exchange(positionsUrl, HttpMethod.GET, request, Map[].class);
        dto.setPositions(positionsResponse.getBody() != null ? Arrays.asList(positionsResponse.getBody()) : List.of());

        // Récupérer les infos du compte
        String accountUrl = apiBaseUrl + "/account";
        ResponseEntity<Map> accountResponse = restTemplate.exchange(accountUrl, HttpMethod.GET, request, Map.class);
        Map<String, Object> account = accountResponse.getBody();
        dto.setAccount(account);

        // Récupérer le montant initial (somme des dépôts/retraits)
        double initialDeposit = getInitialDeposit(compte);
        dto.setInitialDeposit(initialDeposit);

        // Trier les positions par unrealized_plpc décroissant
        if (dto.getPositions() != null) {
            dto.getPositions().sort((a, b) -> {
                Double plpcA = a.get("unrealized_plpc") != null ? Double.valueOf(a.get("unrealized_plpc").toString()) : Double.NEGATIVE_INFINITY;
                Double plpcB = b.get("unrealized_plpc") != null ? Double.valueOf(b.get("unrealized_plpc").toString()) : Double.NEGATIVE_INFINITY;
                return plpcB.compareTo(plpcA);
            });
        }

        return dto;
    }

    /**
     * Annule un ordre pour un compte donné.
     */
    public String cancelOrder(CompteEntity compte, String orderId) {
        String url = apiBaseUrl + "/orders/" + orderId;
        HttpHeaders headers = new HttpHeaders();
        headers.set("APCA-API-KEY-ID", compte.getCle());
        headers.set("APCA-API-SECRET-KEY", compte.getSecret());
        HttpEntity<Void> request = new HttpEntity<>(headers);
        ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.DELETE, request, String.class);
        String responseStr = response.getBody() != null ? response.getBody() : "Annulation demandée.";

        OrderEntity orderEntity = new OrderEntity(String.valueOf(compte.getId()), orderId, "cancel", "cancel " + orderId, responseStr, null, response.getStatusCode().toString(), null);
        orderRepository.save(orderEntity);
        return responseStr;
    }

    /**
     * Récupère le portefeuille (cash + positions) pour un compte donné.
     */
    public Portfolio getPortfolio(CompteEntity compte) {
        PortfolioDto dto = this.getPortfolioWithPositions(compte);
        double cash = 0.0;
        if (dto.getAccount() != null && dto.getAccount().get("cash") != null) {
            try {
                cash = Double.parseDouble(dto.getAccount().get("cash").toString());
            } catch (NumberFormatException e) {
                cash = 0.0;
            }
        }
        List<Position> positions = new ArrayList<>();
        if (dto.getPositions() != null) {
            for (Map<String, Object> posMap : dto.getPositions()) {
                String symbol = posMap.get("symbol") != null ? posMap.get("symbol").toString() : null;
                int quantity = 0;
                if (posMap.get("qty") != null) {
                    try {
                        quantity = Integer.parseInt(posMap.get("qty").toString());
                    } catch (NumberFormatException e) {
                        quantity = 0;
                    }
                }
                double costBasis = 0;
                if (posMap.get("cost_basis") != null) {
                    try {
                        costBasis = Double.parseDouble(posMap.get("cost_basis").toString());
                    } catch (NumberFormatException e) {
                        costBasis = 0.0;
                    }
                }
                double marketValue = 0;
                if (posMap.get("market_value") != null) {
                    try {
                        marketValue = Double.parseDouble(posMap.get("market_value").toString());
                    } catch (NumberFormatException e) {
                        marketValue = 0.0;
                    }
                }
                if (symbol != null) {
                    positions.add(new Position(symbol, quantity, costBasis, marketValue));
                }
            }
        }
        return new Portfolio(cash, positions);
    }

    /**
     * Récupère les ordres Alpaca, avec filtre optionnel sur le symbole, l'annulabilité et la limite de résultats.
     * @param symbol symbole à filtrer (null ou vide = tous)
     * @param cancelable true = seulement les statuts annulables, false = tous
     * @return liste des ordres filtrés
     */
    public List<Order> getOrders(CompteEntity compte, String symbol, Boolean cancelable) {
        StringBuilder url = new StringBuilder(apiBaseUrl + "/orders?status=all");
        if (symbol != null && !symbol.trim().isEmpty()) {
            url.append("&symbols=").append(symbol.trim());
        }
        url.append("&limit=").append(limitOrders);
        HttpHeaders headers = new HttpHeaders();
        headers.set("APCA-API-KEY-ID", compte.getCle());
        headers.set("APCA-API-SECRET-KEY", compte.getSecret());
        HttpEntity<Void> request = new HttpEntity<>(headers);
        ResponseEntity<String> response = restTemplate.exchange(url.toString(), HttpMethod.GET, request, String.class);
        Order[] orders = gson.fromJson(response.getBody(), Order[].class);
        List<Order> result = Arrays.asList(orders);
        if (Boolean.TRUE.equals(cancelable)) {
            result = result.stream()
                    .filter(order -> ACTIVE_ORDER_STATUSES.contains(order.getStatus()))
                    .toList();
        }
        return result;
    }

    /**
     * Récupère les actualités depuis l'API Alpaca News.
     * @param symbols liste des symboles à filtrer (optionnel)
     * @param startDate date de début au format ISO (optionnel)
     * @param endDate date de fin au format ISO (optionnel)
     * @param sort ordre de tri ("asc" ou "desc", par défaut "desc")
     * @param includeContent inclure le contenu complet des articles (par défaut false)
     * @param excludeContentless exclure les articles sans contenu (par défaut true)
     * @param pageSize nombre d'articles par page (max 50, par défaut 10)
     * @return liste des actualités filtrées
     */
    public Map<String, Object> getNews(CompteEntity compte, List<String> symbols, String startDate, String endDate,
                                      String sort, Boolean includeContent, Boolean excludeContentless,
                                      Integer pageSize) {
        StringBuilder url = new StringBuilder(apiMarketBaseUrl + "/v1beta1/news");

        List<String> params = new ArrayList<>();

        if (symbols != null && !symbols.isEmpty()) {
            params.add("symbols=" + String.join(",", symbols));
        }

        if (startDate != null && !startDate.trim().isEmpty()) {
            params.add("start=" + startDate.trim());
        }

        if (endDate != null && !endDate.trim().isEmpty()) {
            params.add("end=" + endDate.trim());
        }

        if (sort != null && !sort.trim().isEmpty()) {
            params.add("sort=" + sort.trim());
        } else {
            params.add("sort=desc"); // valeur par défaut
        }

        if (includeContent != null) {
            params.add("include_content=" + includeContent);
        }

        if (excludeContentless != null) {
            params.add("exclude_contentless=" + excludeContentless);
        } else {
            params.add("exclude_contentless=true"); // valeur par défaut
        }

        if (pageSize != null && pageSize > 0 && pageSize <= 50) {
            params.add("limit=" + pageSize);
        } else {
            params.add("limit="+defaultLimitNews); // valeur par défaut
        }

        if (!params.isEmpty()) {
            url.append("?").append(String.join("&", params));
        }

        HttpHeaders headers = new HttpHeaders();
        headers.set("APCA-API-KEY-ID", compte.getCle());
        headers.set("APCA-API-SECRET-KEY", compte.getSecret());

        HttpEntity<Void> request = new HttpEntity<>(headers);

        try {
            ResponseEntity<Map> response = restTemplate.exchange(url.toString(), HttpMethod.GET, request, Map.class);
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                Map<String, Object> originalResponse = response.getBody();

                // Filtrer la réponse pour ne garder que les champs requis
                return filterNewsResponse(originalResponse);
            }
        } catch (Exception e) {
            throw new RuntimeException("Erreur lors de la récupération des actualités Alpaca: " + e.getMessage(), e);
        }

        return new HashMap<>(); // retour vide en cas de problème
    }

    /**
     * Filtre la réponse des actualités pour ne garder que les champs requis.
     * @param originalResponse réponse originale de l'API Alpaca
     * @return réponse filtrée avec seulement les champs demandés
     */
    private Map<String, Object> filterNewsResponse(Map<String, Object> originalResponse) {
        Map<String, Object> filteredResponse = new HashMap<>();

        // Copier les champs de niveau supérieur sauf "news"
        originalResponse.forEach((key, value) -> {
            if (!"news".equals(key)) {
                filteredResponse.put(key, value);
            }
        });

        // Traiter la liste des actualités
        Object newsObj = originalResponse.get("news");
        if (newsObj instanceof List) {
            List<?> newsList = (List<?>) newsObj;
            List<Map<String, Object>> filteredNewsList = new ArrayList<>();

            for (Object newsItem : newsList) {
                if (newsItem instanceof Map) {
                    Map<?, ?> newsMap = (Map<?, ?>) newsItem;
                    Map<String, Object> filteredNews = new HashMap<>();

                    // Ne garder que les champs requis
                    String[] fieldsToKeep = {"content", "summary", "created_at", "headline", "source", "symbols"};
                    for (String field : fieldsToKeep) {
                        if (newsMap.containsKey(field)) {
                            Object value = newsMap.get(field);
                            // Nettoyer le HTML du champ content
                            if ("content".equals(field) && value instanceof String) {
                                value = TradeUtils.stripHtmlTags((String) value);
                            }
                            filteredNews.put(field, value);
                        }
                    }

                    filteredNewsList.add(filteredNews);
                }
            }

            filteredResponse.put("news", filteredNewsList);
        }

        return filteredResponse;
    }


    /**
     * Récupère les actualités pour un symbole spécifique avec des paramètres simplifiés.
     * @param symbol symbole de l'action
     * @param pageSize nombre d'articles à récupérer (max 50)
     * @return liste des actualités pour ce symbole
     */
    public Map<String, Object> getNewsForSymbol(CompteEntity compte, String symbol, Integer pageSize) {
        return getNews(compte, Arrays.asList(symbol), null, null, "desc", false, true, pageSize);
    }

    /**
     * Récupère les actualités récentes (sans filtre de symbole).
     * @param pageSize nombre d'articles à récupérer (max 50)
     * @return liste des actualités récentes
     */
    public Map<String, Object> getRecentNews(CompteEntity compte, Integer pageSize) {
        return getNews(compte, null, null, null, "desc", false, true, pageSize);
    }

    /**
     * Récupère les actualités avec contenu complet pour un symbole.
     * @param symbol symbole de l'action
     * @param pageSize nombre d'articles à récupérer (max 50)
     * @return liste des actualités avec contenu complet
     */
    public Map<String, Object> getDetailedNewsForSymbol(CompteEntity compte, String symbol, Integer pageSize) {
        return getNews(compte, Arrays.asList(symbol), null, null, "desc", true, true, pageSize);
    }

    /**
     * Récupère la somme nette des dépôts et retraits (montant initial) via l'API Alpaca.
     * @param compte CompteEntity contenant les credentials
     * @return somme nette des dépôts (positif) et retraits (négatif)
     */
    public double getInitialDeposit(CompteEntity compte) {
        String url = apiBaseUrl + "/account/activities";
        HttpHeaders headers = new HttpHeaders();
        headers.set("APCA-API-KEY-ID", compte.getCle());
        headers.set("APCA-API-SECRET-KEY", compte.getSecret());
        HttpEntity<Void> request = new HttpEntity<>(headers);
        ResponseEntity<String> rawResponse = restTemplate.exchange(
                url, HttpMethod.GET, request, String.class);
        logger.info("Réponse brute Alpaca activities: {}", rawResponse.getBody());
        AlpacaTransferActivity[] activities = gson.fromJson(rawResponse.getBody(), AlpacaTransferActivity[].class);
        double sum = 0.0;
        if (activities != null) {
            for (AlpacaTransferActivity act : activities) {
                // Types de transferts de fonds selon la doc Alpaca et observation du log (inclure JNLC)
                if (act.getActivityType() != null && (
                        act.getActivityType().equals("ACHV") ||
                        act.getActivityType().equals("ACATC") ||
                        act.getActivityType().equals("ACATS") ||
                        act.getActivityType().equals("CSD") ||
                        act.getActivityType().equals("CSR") ||
                        act.getActivityType().equals("JNLC")
                ) && act.getNetAmount() != null) {
                    try {
                        sum += Double.parseDouble(act.getNetAmount());
                    } catch (Exception e) {
                        logger.warn("Impossible de parser net_amount: {}", act.getNetAmount());
                    }
                }
            }
        }
        return sum;
    }

    /**
     * Récupère l'historique des barres (candles) pour un symbole donné.
     * @param symbol Symbole de l'action (ex: "AAPL")
     * @param startDate Date de début (format ISO 8601, ex: "2023-01-01T00:00:00Z")
     * @param endDate Date de fin (format ISO 8601, ex: "2023-01-31T23:59:59Z")
     * @param timeframe Unité de temps (ex: "1Min", "5Min", "1Hour", "1Day")
     * @return bars en String JSON
     * c : close — prix de clôture de la bougie (ici 182.775)
     * h : high — prix le plus haut atteint pendant la période (185.86)
     * l : low — prix le plus bas atteint pendant la période (181.845)
     * n : number of trades — nombre de transactions durant la période (865)
     * o : open — prix d’ouverture de la bougie (185.86)
     * t : timestamp — date et heure de la bougie (2025-03-18T04:00:00Z)
     * v : volume — volume total échangé durant la période (22150)
     * vw : volume weighted average price — prix moyen pondéré par le volume (182.894341)
     */
    public String getHistoricalBars(String symbol, String startDate, String endDate, String timeframe) {
        String url = apiMarketBaseUrl + "/v2/stocks/" + symbol + "/bars?start=" + startDate + "&end=" + endDate + "&timeframe=" + timeframe + "&feed=iex";
        List<CompteEntity> listeCompte = compteService.getAllComptes();
        int maxRetries = 5;
        int delay = 1000; // 1 seconde
        for (int attempt = 0; attempt < maxRetries; attempt++) {
            try {
                int indexCompte = attempt % listeCompte.size();
                HttpHeaders headers = new HttpHeaders();
                headers.set("APCA-API-KEY-ID", listeCompte.get(indexCompte).getCle());
                headers.set("APCA-API-SECRET-KEY", listeCompte.get(indexCompte).getSecret());
                HttpEntity<Void> request = new HttpEntity<>(headers);
                ResponseEntity<Map> response = restTemplate.exchange(url, HttpMethod.GET, request, Map.class);
                logger.info("Réponse brute Alpaca getHistoricalBars: {}", response.getBody());
                if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                    Object barsObj = response.getBody().get("bars");
                    if (barsObj != null) {
                        return gson.toJson(barsObj);
                    }
                    return "[]";
                } else if (response.getStatusCode() == HttpStatus.TOO_MANY_REQUESTS) {
                    logger.warn("Rate limit atteint (429), tentative {} sur {}", attempt + 1, maxRetries);
                } else {
                    logger.error("Erreur HTTP {} lors de la récupération des barres historiques Alpaca: {}", response.getStatusCode(), response.getBody());
                    break;
                }
            } catch (org.springframework.web.client.HttpClientErrorException.TooManyRequests e) {
                logger.warn("Rate limit atteint (429), tentative {} sur {}", attempt + 1, maxRetries);
            } catch (Exception e) {
                logger.error("Erreur lors de la récupération des barres historiques Alpaca: {}", e.getMessage());
                if (attempt == maxRetries - 1) {
                    throw new RuntimeException("Erreur lors de la récupération des barres historiques Alpaca: " + e.getMessage(), e);
                }
            }
            try {
                Thread.sleep(delay);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                break;
            }
            delay *= 2; // backoff exponentiel
        }
        return "";
    }
    public String getHistoricalBarsJson(String symbol, int limit) {
        return getHistoricalBars(symbol, TradeUtils.getStartDate(limit), TradeUtils.getDateToDay(), "1Day");
    }
    public List<DailyValue> getHistoricalBarsJsonDaysMin(String symbol, String dateStart) {
        String json = getHistoricalBars(symbol, dateStart, TradeUtils.getDateToDayEnd(), "1Min");
        return mapBougies(json);
    }
    public List<DailyValue> getHistoricalBars(String symbol, String startDate, String endDate) {
        String json = getHistoricalBars(symbol, startDate, endDate == null ? TradeUtils.getDateToDay() : endDate, "1Day");
        return mapBougies(json);
    }
    public List<DailyValue> mapBougies(String json){
        List<DailyValue> result = new ArrayList<>();
        if (json == null || json.isEmpty()) return result;
        try {
            // Nettoyage éventuel du format (si json commence/termine par [ ou {)
            // On suppose que le format est une liste de maps (ex: [{...}, {...}])
            List<Map<String, Object>> bars = gson.fromJson(json, List.class);
            for (Map<String, Object> bar : bars) {
                DailyValue dv = DailyValue.builder()
                        .date(bar.getOrDefault("t", "").toString())
                        .open(bar.getOrDefault("o", "").toString())
                        .high(bar.getOrDefault("h", "").toString())
                        .low(bar.getOrDefault("l", "").toString())
                        .close(bar.getOrDefault("c", "").toString())
                        .volume(bar.getOrDefault("v", "").toString())
                        .numberOfTrades(bar.getOrDefault("n", "").toString())
                        .volumeWeightedAveragePrice(bar.getOrDefault("vw", "").toString())
                        .build();
                result.add(dv);
            }
        } catch (Exception e) {
            logger.error("Erreur lors du parsing des barres historiques Alpaca: {}", e.getMessage());
        }
        return result;
    }




    /**
     * Récupère la liste des symboles disponibles sur IEX via Alpaca.
     */
    public List<AlpacaAsset> getIexSymbolsFromAlpaca() {
        CompteEntity compte = compteService.getAllComptes().get(0);
        String url = apiBaseUrl + "/assets?feed=iex";
        HttpHeaders headers = new HttpHeaders();
        headers.set("APCA-API-KEY-ID", compte.getCle());
        headers.set("APCA-API-SECRET-KEY", compte.getSecret());
        HttpEntity<Void> request = new HttpEntity<>(headers);
        try {
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.GET, request, String.class);
            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                com.google.gson.reflect.TypeToken<List<AlpacaAsset>> typeToken =
                        new com.google.gson.reflect.TypeToken<>() {};
                List<AlpacaAsset> assets = gson.fromJson(response.getBody(), typeToken.getType());
                List<AlpacaAsset> symbolsActive = new ArrayList<>();
                for (AlpacaAsset asset : assets) {
                    if ("active".equalsIgnoreCase(asset.getStatus())
                            && asset.getExchange() != null
                            && !"CRYPTO".equalsIgnoreCase(asset.getExchange())
                            && !"OTC".equalsIgnoreCase(asset.getExchange())) {
                        symbolsActive.add(asset);
                    }
                }
                return symbolsActive;
            }
        } catch (Exception e) {
            logger.error("Erreur lors de la récupération des symboles IEX Alpaca: {}", e.getMessage());
        }
        return Collections.emptyList();
    }



}
