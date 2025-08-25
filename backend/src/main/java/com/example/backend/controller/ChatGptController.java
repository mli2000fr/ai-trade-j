package com.example.backend.controller;

import com.example.backend.model.ChatGptResponse;
import com.example.backend.model.DataAction;
import com.example.backend.service.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/chatgpt")
public class ChatGptController {
    private final ChatGptService chatGptService;
    private final AlphaVantageService alphaVantageService;
    private final TwelveDataService twelveDataService;
    private final FinnhubService finnhubService;
    private final EodhdService eodhdService;
    private final MarketauxService marketauxService;

    @Autowired
    public ChatGptController(ChatGptService chatGptService,
                             FinnhubService finnhubService,
                             AlphaVantageService alphaVantageService,
                             TwelveDataService twelveDataService,
                             EodhdService eodhdService,
                             MarketauxService marketauxService) {
        this.chatGptService = chatGptService;
        this.alphaVantageService = alphaVantageService;
        this.twelveDataService = twelveDataService;
        this.finnhubService = finnhubService;
        this.eodhdService = eodhdService;
        this.marketauxService = marketauxService;
    }

    @PostMapping("/ask")
    public ResponseEntity<String> askChatGpt(@RequestBody String prompt) {
        String response = chatGptService.test(prompt);
        return ResponseEntity.ok(response);
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

}
