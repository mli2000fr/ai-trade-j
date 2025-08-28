package com.app.backend.trade.service;

import com.app.backend.trade.model.Portfolio;
import com.app.backend.trade.model.PortfolioDto;
import com.app.backend.trade.model.Position;
import com.app.backend.trade.model.alpaca.OffsetDateTimeAdapter;
import com.app.backend.trade.model.alpaca.Order;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.java_websocket.client.WebSocketClient;
import org.java_websocket.handshake.ServerHandshake;
import java.net.URI;
import java.time.OffsetDateTime;
import java.util.*;
import java.util.function.Consumer;

@Service
public class AlpacaService {
    @Value("${alpaca.markets.api.key.id}")
    private String apiKeyId;

    @Value("${alpaca.markets.api.key.secret}")
    private String apiKeySecret;

    @Value("${alpaca.markets.api.paper.url}")
    private String apiBaseUrl;

    @Value("${alpaca.markets.api.market.url}")
    private String apiMarketBaseUrl;

    @Value("${alpaca.markets.api.order.limit}")
    private String limitOrders;

    private final RestTemplate restTemplate = new RestTemplate();
    public Order placeOrder(String symbol, double qty, String side, Double priceLimit, Double stopLoss, Double takeProfit) {
        String url = apiBaseUrl + "/orders";
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.set("APCA-API-KEY-ID", apiKeyId);
        headers.set("APCA-API-SECRET-KEY", apiKeySecret);

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


        // Gestion des différents types d'ordre
        if (priceLimit != null && (stopLoss == null && takeProfit == null)) {
            // Ordre limit simple
            body.put("type", "limit");
            body.put("limit_price", priceLimit);
        } else if ((stopLoss != null || takeProfit != null) && qty == Math.floor(qty)) {
            // Bracket order (seulement si qty entier)
            body.put("type", "market"); // Alpaca exige type=market pour bracket
            body.put("order_class", "bracket");
            if (takeProfit != null) {
                Map<String, Object> tp = new HashMap<>();
                tp.put("limit_price", takeProfit);
                body.put("take_profit", tp);
            }
            if (stopLoss != null) {
                Map<String, Object> sl = new HashMap<>();
                sl.put("stop_price", stopLoss);
                body.put("stop_loss", sl);
            }
        } else {
            // Ordre market simple
            body.put("type", "market");
        }

        HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);
        ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);
        Gson gson = new GsonBuilder()
            .registerTypeAdapter(OffsetDateTime.class, new OffsetDateTimeAdapter())
            .create();
        return gson.fromJson(response.getBody(), Order.class);
    }

    public Double getLastPrice(String symbol) {
        String url = apiMarketBaseUrl + "/v2/stocks/" + symbol + "/quotes/latest";
        HttpHeaders headers = new HttpHeaders();
        headers.set("APCA-API-KEY-ID", apiKeyId);
        headers.set("APCA-API-SECRET-KEY", apiKeySecret);
        HttpEntity<Void> request = new HttpEntity<>(headers);
        ResponseEntity<Map> response = restTemplate.exchange(url, HttpMethod.GET, request, Map.class);
        if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
            Map<String, Object> quote = (Map<String, Object>) response.getBody().get("quote");
            if (quote != null && quote.get("ap") != null) {
                return Double.valueOf(quote.get("ap").toString()); // "ap" = ask price
            }
        }
        return null;
    }

    public void streamLivePrice(String symbol, Consumer<String> onMessage) {
        try {
            String wsUrl = "wss://stream.data.alpaca.markets/v2/sip";
            WebSocketClient client = new WebSocketClient(new URI(wsUrl)) {
                @Override
                public void onOpen(ServerHandshake handshakedata) {
                    // Authentification
                    String authMsg = String.format("{\"action\":\"auth\",\"key\":\"%s\",\"secret\":\"%s\"}", apiKeyId, apiKeySecret);
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
            throw new RuntimeException("Erreur lors de la connexion WebSocket Alpaca", e);
        }
    }

    private static final List<String> ACTIVE_ORDER_STATUSES = Arrays.asList(
        "new", "partially_filled", "accepted", "pending_new", "pending_replace", "pending_cancel"
    );

    // Vérifie s'il existe un ordre ouvert du côté opposé (buy/sell) pour ce symbole
    private boolean hasOppositeOpenOrder(String symbol, String side) {
        String url = apiBaseUrl + "/orders?status=all&symbols=" + symbol;
        HttpHeaders headers = new HttpHeaders();
        headers.set("APCA-API-KEY-ID", apiKeyId);
        headers.set("APCA-API-SECRET-KEY", apiKeySecret);
        HttpEntity<Void> request = new HttpEntity<>(headers);
        ResponseEntity<Map[]> response = restTemplate.exchange(url, HttpMethod.GET, request, Map[].class);
        if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
            for (Map order : response.getBody()) {
                String orderSide = (String) order.get("side");
                String orderStatus = (String) order.get("status");
                if (orderSide != null && !orderSide.equalsIgnoreCase(side)
                        && orderStatus != null && ACTIVE_ORDER_STATUSES.contains(orderStatus)) {
                    return true;
                }
            }
        }
        return false;
    }

    public String sellStock(String symbol, double qty) {
        if (hasOppositeOpenOrder(symbol, "sell")) {
            return "Erreur : Un ordre d'achat ouvert existe déjà pour ce symbole. Veuillez attendre son exécution ou l'annuler avant de vendre.";
        }
        return placeOrder(symbol, qty, "sell", null, null, null).toString();
    }

    public String buyStock(String symbol, double qty) {
        return buyStock(symbol, qty, null, null, null);
    }

    public String buyStock(String symbol, double qty, Double priceLimit, Double stopLoss, Double takeProfit) {
        if (hasOppositeOpenOrder(symbol, "buy")) {
            return "Erreur : Un ordre de vente ouvert existe déjà pour ce symbole. Veuillez attendre son exécution ou l'annuler avant d'acheter.";
        }
        return placeOrder(symbol, qty, "buy", priceLimit, stopLoss, takeProfit).toString();
    }

    public PortfolioDto getPortfolioWithPositions() {
        PortfolioDto dto = new PortfolioDto();

        // Récupérer les positions
        String positionsUrl = apiBaseUrl + "/positions";
        HttpHeaders headers = new HttpHeaders();
        headers.set("APCA-API-KEY-ID", apiKeyId);
        headers.set("APCA-API-SECRET-KEY", apiKeySecret);
        HttpEntity<Void> request = new HttpEntity<>(headers);
        ResponseEntity<Map[]> positionsResponse = restTemplate.exchange(positionsUrl, HttpMethod.GET, request, Map[].class);
        dto.setPositions(positionsResponse.getBody() != null ? Arrays.asList(positionsResponse.getBody()) : List.of());

        // Récupérer les infos du compte
        String accountUrl = apiBaseUrl + "/account";
        ResponseEntity<Map> accountResponse = restTemplate.exchange(accountUrl, HttpMethod.GET, request, Map.class);
        Map<String, Object> account = accountResponse.getBody();
        dto.setAccount(account);

        return dto;
    }

    public String cancelOrder(String orderId) {
        String url = apiBaseUrl + "/orders/" + orderId;
        HttpHeaders headers = new HttpHeaders();
        headers.set("APCA-API-KEY-ID", apiKeyId);
        headers.set("APCA-API-SECRET-KEY", apiKeySecret);
        HttpEntity<Void> request = new HttpEntity<>(headers);
        ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.DELETE, request, String.class);
        return response.getBody() != null ? response.getBody() : "Annulation demandée.";
    }

    public Portfolio getPortfolio() {
        PortfolioDto dto = this.getPortfolioWithPositions();
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
    public List<Order> getOrders(String symbol, Boolean cancelable) {
        StringBuilder url = new StringBuilder(apiBaseUrl + "/orders?status=all");
        if (symbol != null && !symbol.trim().isEmpty()) {
            url.append("&symbols=").append(symbol.trim());
        }
        url.append("&limit=").append(limitOrders);
        HttpHeaders headers = new HttpHeaders();
        headers.set("APCA-API-KEY-ID", apiKeyId);
        headers.set("APCA-API-SECRET-KEY", apiKeySecret);
        HttpEntity<Void> request = new HttpEntity<>(headers);
        ResponseEntity<String> response = restTemplate.exchange(url.toString(), HttpMethod.GET, request, String.class);
        Gson gson = new GsonBuilder()
                .registerTypeAdapter(OffsetDateTime.class, new OffsetDateTimeAdapter())
                .create();
        Order[] orders = gson.fromJson(response.getBody(), Order[].class);
        List<Order> result = Arrays.asList(orders);
        if (Boolean.TRUE.equals(cancelable)) {
            result = result.stream()
                    .filter(order -> ACTIVE_ORDER_STATUSES.contains(order.getStatus()))
                    .toList();
        }
        return result;
    }
}
