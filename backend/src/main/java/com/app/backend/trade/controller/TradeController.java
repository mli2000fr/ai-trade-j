package com.app.backend.trade.controller;

import com.app.backend.trade.model.DataAction;
import com.app.backend.trade.model.PortfolioAndOrdersDto;
import com.app.backend.trade.model.TradeRequest;
import com.app.backend.trade.service.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/trade")
public class TradeController {
    private final ChatGptService chatGptService;
    private final AlphaVantageService alphaVantageService;
    private final TwelveDataService twelveDataService;
    private final FinnhubService finnhubService;
    private final EodhdService eodhdService;
    private final MarketauxService marketauxService;
    private final AlpacaService alpacaService;

    @Autowired
    public TradeController(ChatGptService chatGptService,
                           FinnhubService finnhubService,
                           AlphaVantageService alphaVantageService,
                           TwelveDataService twelveDataService,
                           EodhdService eodhdService,
                           MarketauxService marketauxService,
                           AlpacaService alpacaService) {
        this.chatGptService = chatGptService;
        this.alphaVantageService = alphaVantageService;
        this.twelveDataService = twelveDataService;
        this.finnhubService = finnhubService;
        this.eodhdService = eodhdService;
        this.marketauxService = marketauxService;
        this.alpacaService = alpacaService;
    }


    @GetMapping("/analyse-action")
    public ResponseEntity<String> analyseAction(@RequestParam("symbol") String symbol,
                                                @RequestParam("delai") int delai,
                                                @RequestParam("montant") int montant) {
       /*
        String data = alphaVantageService.getDataAction(symbol);
        String sma = alphaVantageService.getSMA(symbol);
        String rsi = alphaVantageService.getRSI(symbol);
        String macd = alphaVantageService.getMACD(symbol);
        String atr = alphaVantageService.getATR(symbol);
        */

        String data = twelveDataService.getDataAction(symbol);
        String sma = twelveDataService.getSMA(symbol);
        String rsi = twelveDataService.getRSI(symbol);
        String macd = twelveDataService.getMACD(symbol);
        String atr = twelveDataService.getATR(symbol);
        String financial = finnhubService.getFinancialData(symbol);
        String statistics = finnhubService.getDefaultKeyStatistics(symbol);
        String earnings = finnhubService.getEarnings(symbol);
        String news = eodhdService.getNews(symbol);

        DataAction dataAction = new DataAction(
                symbol,
                montant,
                delai,
                data,
                sma,
                rsi,
                macd,
                atr,
                financial,
                statistics,
                earnings,
                news
        );

        String result = chatGptService.getAnalyseAction(dataAction);
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
        PortfolioAndOrdersDto dto = alpacaService.getPortfolioAndOrders();
        return ResponseEntity.ok(dto);
    }

    @PostMapping("/order/cancel/{orderId}")
    public ResponseEntity<String> cancelOrder(@PathVariable String orderId) {
        String result = alpacaService.cancelOrder(orderId);
        return ResponseEntity.ok(result);
    }

}
