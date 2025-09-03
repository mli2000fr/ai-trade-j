package com.app.backend.trade.controller;

import com.app.backend.trade.model.*;
import com.app.backend.trade.service.*;
import com.app.backend.trade.strategy.StrategyManager;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/trade")
public class TradeController {
    private final TradeHelper tradeHelper;
    private final AlpacaService alpacaService;
    private final CompteService compteService;
    private final StrategyService strategyService;

    @Autowired
    public TradeController(AlpacaService alpacaService,
                           TradeHelper tradeHelper,
                           CompteService compteService,
                           StrategyService strategyService) {
        this.alpacaService = alpacaService;
        this.tradeHelper = tradeHelper;
        this.compteService = compteService;
        this.strategyService = strategyService;
    }

    /**
     * Effectue un trade (achat ou vente) sur une action.
     */
    @PostMapping("/trade")
    public ResponseEntity<String> trade(@RequestBody TradeRequest request) {
        CompteEntity compte = compteService.getCompteCredentialsById(request.getId());
        String result;
        if ("buy".equalsIgnoreCase(request.getAction())) {
            result = alpacaService.buyStock(compte, request);
        } else if ("sell".equalsIgnoreCase(request.getAction())) {
            result = alpacaService.sellStock(compte, request);
        } else {
            return ResponseEntity.badRequest().body("Action invalide : doit être 'buy' ou 'sell'");
        }
        return ResponseEntity.ok(result);
    }

    /**
     * Récupère le portefeuille et les positions pour un compte donné.
     */
    @GetMapping("/portfolio")
    public ResponseEntity<PortfolioDto> getPortfolioWithPositions(@RequestParam String id) {
        CompteEntity compte = compteService.getCompteCredentialsById(id);
        PortfolioDto dto = alpacaService.getPortfolioWithPositions(compte);
        return ResponseEntity.ok(dto);
    }

    /**
     * Annule un ordre pour un compte donné.
     */
    @PostMapping("/order/cancel/{id}/{orderId}")
    public ResponseEntity<String> cancelOrder(@PathVariable String orderId, @PathVariable String id) {
        CompteEntity compte = compteService.getCompteCredentialsById(id);
        String result = alpacaService.cancelOrder(compte, orderId);
        return ResponseEntity.ok(result);
    }

    /**
     * Effectue un trade automatique via l'IA pour un compte donné, avec analyse GPT optionnelle.
     */
    @PostMapping("/trade-ai-auto")
    public ResponseEntity<ReponseAuto> tradeAIAuto(@RequestBody TradeAutoRequestGpt request)  {
        CompteEntity compte = compteService.getCompteCredentialsById(request.getId());
        String analyseGpt = request.getAnalyseGpt() != null ? request.getAnalyseGpt() : "";
        ReponseAuto result = tradeHelper.tradeAIAuto(compte, request.getSymboles(), analyseGpt);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/execute-orders")
    public ResponseEntity<List<OrderRequest>> executeOrders(@RequestBody ExecuteOrdersRequest request) {
        CompteEntity compte = compteService.getCompteCredentialsById(request.getId());
        List<OrderRequest> result = tradeHelper.processOrders(compte, request.getIdGpt(), request.getOrders());
        return ResponseEntity.ok(result);
    }

    /**
     * Récupère les ordres pour un compte donné, avec filtres optionnels.
     */
    @GetMapping("/orders")
    public ResponseEntity<?> getOrders(
            @RequestParam(value = "symbol", required = false) String symbol,
            @RequestParam(value = "cancelable", required = false) Boolean cancelable,
            @RequestParam(value = "id", required = false) String id,
            @RequestParam(value = "sizeOrders", required = false) Integer sizeOrders) {
        CompteEntity compte = compteService.getCompteCredentialsById(id);
        List<com.app.backend.trade.model.alpaca.Order> orders = alpacaService.getOrders(compte, symbol, cancelable);
        if (orders == null) orders = Collections.emptyList();
        int limit = sizeOrders == null ? 10 : sizeOrders;
        if (orders.size() > limit) orders = orders.subList(0, limit);
        return ResponseEntity.ok(orders);
    }

    /**
     * Récupère les news pour les symboles donnés.
     */
    @GetMapping("/news")
    public ResponseEntity<Map<String, Object>> getNews(
            @RequestParam(value = "symbols", required = false) List<String> symbols,
            @RequestParam(value = "start", required = false) String startDate,
            @RequestParam(value = "end", required = false) String endDate,
            @RequestParam(value = "sort", required = false, defaultValue = "desc") String sort,
            @RequestParam(value = "include_content", required = false, defaultValue = "false") Boolean includeContent,
            @RequestParam(value = "exclude_contentless", required = false, defaultValue = "true") Boolean excludeContentless,
            @RequestParam(value = "page_size", required = false, defaultValue = "10") Integer pageSize) {
        CompteEntity compte = getDefaultCompte();
        Map<String, Object> news = alpacaService.getNews(compte, symbols, startDate, endDate, sort, includeContent, excludeContentless, pageSize);
        return ResponseEntity.ok(news);
    }

    /**
     * Récupère les news pour un symbole donné.
     */
    @GetMapping("/news/{symbol}")
    public ResponseEntity<Map<String, Object>> getNewsForSymbol(
            @PathVariable String symbol,
            @RequestParam(value = "pageSize", required = false, defaultValue = "10") Integer pageSize) {
        CompteEntity compte = getDefaultCompte();
        Map<String, Object> news = alpacaService.getNewsForSymbol(compte, symbol, pageSize);
        return ResponseEntity.ok(news);
    }

    /**
     * Récupère les news récentes.
     */
    @GetMapping("/news/recent")
    public ResponseEntity<Map<String, Object>> getRecentNews(
            @RequestParam(value = "pageSize", required = false, defaultValue = "10") Integer pageSize) {
        CompteEntity compte = getDefaultCompte();
        Map<String, Object> news = alpacaService.getRecentNews(compte, pageSize);
        return ResponseEntity.ok(news);
    }

    /**
     * Récupère les news détaillées pour un symbole donné.
     */
    @GetMapping("/news/{symbol}/detailed")
    public ResponseEntity<Map<String, Object>> getDetailedNewsForSymbol(
            @PathVariable String symbol,
            @RequestParam(value = "pageSize", required = false, defaultValue = "10") Integer pageSize) {
        CompteEntity compte = getDefaultCompte();
        Map<String, Object> news = alpacaService.getDetailedNewsForSymbol(compte, symbol, pageSize);
        return ResponseEntity.ok(news);
    }

    /**
     * Récupère la liste de tous les comptes.
     */
    @GetMapping("/comptes")
    public ResponseEntity<List<CompteDto>> getAllComptes() {
        List<CompteDto> comptes = compteService.getAllComptesDto();
        return ResponseEntity.ok(comptes);
    }

    /**
     * Récupère le premier compte par défaut (utilisé pour les endpoints news).
     */
    private CompteEntity getDefaultCompte() {
        List<CompteEntity> comptes = compteService.getAllComptes();
        if (comptes.isEmpty()) throw new IllegalStateException("Aucun compte disponible");
        return compteService.getCompteCredentialsById(comptes.get(0).getId().toString());
    }

    /**
     * Liste les stratégies disponibles, actives et le mode de combinaison.
     */
    @GetMapping("/strategies")
    public ResponseEntity<StrategyListDto> getStrategies() {
        return ResponseEntity.ok(new StrategyListDto(
                strategyService.getAllStrategyNames(),
                strategyService.getActiveStrategyNames(),
                strategyService.getStrategyManager().getCombinationMode().name(),
                strategyService.getLogs()
        ));
    }

    /**
     * Modifie dynamiquement les stratégies actives.
     */
    @PostMapping("/strategies/active")
    public ResponseEntity<Void> setActiveStrategies(@RequestBody SetActiveStrategiesRequest req) {
        strategyService.setActiveStrategiesByNames(req.getStrategyNames());
        return ResponseEntity.ok().build();
    }

    /**
     * Modifie dynamiquement le mode de combinaison.
     */
    @PostMapping("/strategies/mode")
    public ResponseEntity<Void> setCombinationMode(@RequestBody SetCombinationModeRequest req) {
        strategyService.setCombinationMode(StrategyManager.CombinationMode.valueOf(req.getCombinationMode()));
        return ResponseEntity.ok().build();
    }

    /**
     * Teste le signal combiné sur une série de prix de clôture fournie.
     * Body: { "closePrices": [123.4, 124.1, ...], "isEntry": true }
     * Retourne true si le signal est validé sur la dernière barre.
     */
    @PostMapping("/strategies/test-signal")
    public ResponseEntity<Map<String, Object>> testCombinedSignal(@RequestBody Map<String, Object> body) {
        List<Double> closePrices = (List<Double>) body.get("closePrices");
        boolean isEntry = body.get("isEntry") != null && Boolean.TRUE.equals(body.get("isEntry"));
        boolean result = tradeHelper.testCombinedSignalOnClosePrices(closePrices, isEntry);
        return ResponseEntity.ok(Collections.singletonMap("signal", result));
    }
}
