package com.app.backend.trade.controller;

import com.app.backend.trade.model.*;
import com.app.backend.trade.service.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.Collections;
import java.util.List;

@RestController
@RequestMapping("/api/trade")
public class TradeController {
    private final TradeHelper tradeHelper;
    private final AlpacaService alpacaService;
    private final CompteService compteService;

    @Autowired
    public TradeController(AlpacaService alpacaService,
                           TradeHelper tradeHelper,
                           CompteService compteService) {
        this.alpacaService = alpacaService;
        this.tradeHelper = tradeHelper;
        this.compteService = compteService;
    }

    /**
     * Effectue une opération d'achat ou de vente selon l'action spécifiée dans la requête.
     * @param request informations de l'ordre (action, id, etc.)
     * @return résultat de l'opération
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
     * Récupère le portefeuille et les positions du compte sélectionné.
     * @param id identifiant du compte
     * @return portefeuille et positions
     */
    @GetMapping("/portfolio")
    public ResponseEntity<PortfolioDto> getPortfolioWithPositions(@RequestParam String id) {
        CompteEntity compte = compteService.getCompteCredentialsById(id);
        PortfolioDto dto = alpacaService.getPortfolioWithPositions(compte);
        return ResponseEntity.ok(dto);
    }

    /**
     * Annule un ordre spécifique pour le compte donné.
     * @param orderId identifiant de l'ordre
     * @param id identifiant du compte
     * @return résultat de l'annulation
     */
    @PostMapping("/order/cancel/{id}/{orderId}")
    public ResponseEntity<String> cancelOrder(@PathVariable String orderId, @PathVariable String id) {
        CompteEntity compte = compteService.getCompteCredentialsById(id);
        String result = alpacaService.cancelOrder(compte, orderId);
        return ResponseEntity.ok(result);
    }

    /**
     * Exécute le trading automatique basé sur l'IA pour les symboles donnés.
     * @param request informations pour le trade IA (id, symboles, analyseGpt)
     * @return réponse contenant les ordres et l'analyse IA
     */
    @PostMapping("/trade-ai-auto")
    public ResponseEntity<ReponseAuto> tradeAIAuto(@RequestBody TradeAutoRequestGpt request)  {
        CompteEntity compte = compteService.getCompteCredentialsById(request.getId());
        String analyseGpt = request.getAnalyseGpt() != null ? request.getAnalyseGpt() : "";
        ReponseAuto result = tradeHelper.tradeAIAuto(compte, request.getSymboles(), analyseGpt);
        return ResponseEntity.ok(result);
    }

    /**
     * Exécute une liste d'ordres pour le compte sélectionné.
     * @param request informations d'exécution (id, idGpt, liste d'ordres)
     * @return liste d'ordres mise à jour
     */
    @PostMapping("/execute-orders")
    public ResponseEntity<List<OrderRequest>> executeOrders(@RequestBody ExecuteOrdersRequest request) {
        CompteEntity compte = compteService.getCompteCredentialsById(request.getId());
        List<OrderRequest> result = tradeHelper.processOrders(compte, request.getIdGpt(), request.getOrders());
        return ResponseEntity.ok(result);
    }

    /**
     * Récupère la liste des ordres du compte, avec options de filtrage.
     * @param symbol symbole à filtrer (optionnel)
     * @param cancelable filtrer les ordres annulables (optionnel)
     * @param id identifiant du compte
     * @param sizeOrders nombre maximum d'ordres à retourner (optionnel)
     * @return liste des ordres
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
     * Récupère la liste de tous les comptes disponibles.
     * @return liste des comptes
     */
    @GetMapping("/comptes")
    public ResponseEntity<List<CompteDto>> getAllComptes() {
        List<CompteDto> comptes = compteService.getAllComptesDto();
        return ResponseEntity.ok(comptes);
    }

    @GetMapping("/getPromptAnalyseSymbol")
    public ResponseEntity<String> getPromptAnalyseSymbol(@RequestParam String idCompte, @RequestParam String symbols) {
        return ResponseEntity.ok(tradeHelper.getPromptAnalyseSymbol(idCompte, symbols));
    }
}
