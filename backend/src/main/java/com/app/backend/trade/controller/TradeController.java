package com.app.backend.trade.controller;

import com.app.backend.trade.model.PortfolioDto;
import com.app.backend.trade.model.TradeRequest;
import com.app.backend.trade.model.TradeAutoRequest;
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
    public ResponseEntity<String> analyseAction(@RequestParam("symbol") String symbol)  {

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
    public ResponseEntity<PortfolioDto> getPortfolioWithPositions() {
        PortfolioDto dto = alpacaService.getPortfolioWithPositions();
        return ResponseEntity.ok(dto);
    }

    @PostMapping("/order/cancel/{orderId}")
    public ResponseEntity<String> cancelOrder(@PathVariable String orderId) {
        String result = alpacaService.cancelOrder(orderId);
        return ResponseEntity.ok(result);
    }

    @PostMapping("/trade-ai")
    public ResponseEntity<String> tradeAI(@RequestBody TradeRequest request)  {

        String result = tradeHelper.tradeAI(request.getSymbol());
        return ResponseEntity.ok(result);
    }

    @PostMapping("/trade-ai-auto")
    public ResponseEntity<String> tradeAIAuto(@RequestBody TradeAutoRequest request)  {
        String result = tradeHelper.tradeAIAuto(request.getSymboles());
        return ResponseEntity.ok(result);
    }

    @GetMapping("/orders")
    public ResponseEntity<?> getOrders(
            @RequestParam(value = "symbol", required = false) String symbol,
            @RequestParam(value = "cancelable", required = false) Boolean cancelable) {
        java.util.List<com.app.backend.trade.model.alpaca.Order> orders = alpacaService.getOrders(symbol, cancelable);
        // Ne retourner que les 10 premiers ordres
        if (orders == null) orders = java.util.Collections.emptyList();
        if (orders.size() > 10) orders = orders.subList(0, 10);
        return ResponseEntity.ok(orders);
    }

    @GetMapping("/news")
    public ResponseEntity<java.util.Map<String, Object>> getNews(
            @RequestParam(value = "symbols", required = false) java.util.List<String> symbols,
            @RequestParam(value = "start", required = false) String startDate,
            @RequestParam(value = "end", required = false) String endDate,
            @RequestParam(value = "sort", required = false, defaultValue = "desc") String sort,
            @RequestParam(value = "include_content", required = false, defaultValue = "false") Boolean includeContent,
            @RequestParam(value = "exclude_contentless", required = false, defaultValue = "true") Boolean excludeContentless,
            @RequestParam(value = "page_size", required = false, defaultValue = "10") Integer pageSize) {

        java.util.Map<String, Object> news = alpacaService.getNews(symbols, startDate, endDate, sort, includeContent, excludeContentless, pageSize);
        return ResponseEntity.ok(news);
    }

    @GetMapping("/news/{symbol}")
    public ResponseEntity<java.util.Map<String, Object>> getNewsForSymbol(
            @PathVariable String symbol,
            @RequestParam(value = "pageSize", required = false, defaultValue = "10") Integer pageSize) {

        java.util.Map<String, Object> news = alpacaService.getNewsForSymbol(symbol, pageSize);
        return ResponseEntity.ok(news);
    }

    @GetMapping("/news/recent")
    public ResponseEntity<java.util.Map<String, Object>> getRecentNews(
            @RequestParam(value = "pageSize", required = false, defaultValue = "10") Integer pageSize) {

        java.util.Map<String, Object> news = alpacaService.getRecentNews(pageSize);
        return ResponseEntity.ok(news);
    }

    @GetMapping("/news/{symbol}/detailed")
    public ResponseEntity<java.util.Map<String, Object>> getDetailedNewsForSymbol(
            @PathVariable String symbol,
            @RequestParam(value = "pageSize", required = false, defaultValue = "10") Integer pageSize) {

        java.util.Map<String, Object> news = alpacaService.getDetailedNewsForSymbol(symbol, pageSize);
        return ResponseEntity.ok(news);
    }

}
