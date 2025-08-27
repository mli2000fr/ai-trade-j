package com.app.backend.trade.controller;

import com.app.backend.trade.model.PortfolioAndOrdersDto;
import com.app.backend.trade.model.TradeRequest;
import com.app.backend.trade.service.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/trade")
public class TradeController {
    private final TradeHelper tradeHelper;
    private final AlpacaService alpacaService;

    @Autowired
    public TradeController(AlpacaService alpacaService,
                           TradeHelper tradeHelper) {
        this.alpacaService = alpacaService;
        this.tradeHelper = tradeHelper;
    }


    @GetMapping("/analyse-action")
    public ResponseEntity<String> analyseAction(@RequestParam("symbol") String symbol) throws Exception {

        String result = tradeHelper.getAnalyseAction(symbol);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/trade")
    public ResponseEntity<String> trade(@RequestBody TradeRequest request) {
        String result;
        if ("buy".equalsIgnoreCase(request.getAction())) {
            result = alpacaService.buyStock(request.getSymbol(), request.getQuantity());
        } else if ("sell".equalsIgnoreCase(request.getAction())) {
            result = alpacaService.sellStock(request.getSymbol(), request.getQuantity());
        } else {
            return ResponseEntity.badRequest().body("Action invalide : doit être 'buy' ou 'sell'");
        }
        return ResponseEntity.ok(result);
    }

    @GetMapping("/portfolio")
    public ResponseEntity<PortfolioAndOrdersDto> getPortfolioAndOrders() {
        PortfolioAndOrdersDto dto = alpacaService.getPortfolioAndOrders(true);
        return ResponseEntity.ok(dto);
    }

    @PostMapping("/order/cancel/{orderId}")
    public ResponseEntity<String> cancelOrder(@PathVariable String orderId) {
        String result = alpacaService.cancelOrder(orderId);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/trade-ai")
    public ResponseEntity<String> tradeAI(String symbol) throws Exception {

        String result = tradeHelper.tradeAI(symbol);
        return ResponseEntity.ok(result);
    }

}
