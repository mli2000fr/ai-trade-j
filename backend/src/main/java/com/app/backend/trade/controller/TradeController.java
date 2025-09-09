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


    @GetMapping("/portfolio")
    public ResponseEntity<PortfolioDto> getPortfolioWithPositions(@RequestParam String id) {
        CompteEntity compte = compteService.getCompteCredentialsById(id);
        PortfolioDto dto = alpacaService.getPortfolioWithPositions(compte);
        return ResponseEntity.ok(dto);
    }


    @PostMapping("/order/cancel/{id}/{orderId}")
    public ResponseEntity<String> cancelOrder(@PathVariable String orderId, @PathVariable String id) {
        CompteEntity compte = compteService.getCompteCredentialsById(id);
        String result = alpacaService.cancelOrder(compte, orderId);
        return ResponseEntity.ok(result);
    }


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


    @GetMapping("/comptes")
    public ResponseEntity<List<CompteDto>> getAllComptes() {
        List<CompteDto> comptes = compteService.getAllComptesDto();
        return ResponseEntity.ok(comptes);
    }


}
