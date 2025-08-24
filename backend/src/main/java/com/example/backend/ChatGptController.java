package com.example.backend;

import com.example.backend.model.DataAction;
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

    @Autowired
    public ChatGptController(ChatGptService chatGptService, FinnhubService finnhubService, AlphaVantageService alphaVantageService, TwelveDataService twelveDataService) {
        this.chatGptService = chatGptService;
        this.alphaVantageService = alphaVantageService;
        this.twelveDataService = twelveDataService;
        this.finnhubService = finnhubService;
    }

    @PostMapping("/ask")
    public ResponseEntity<ChatGptResponse> askChatGpt(@RequestBody String prompt) {
        ChatGptResponse response = chatGptService.askChatGpt(prompt);
        if (response.getError() != null) {
            return ResponseEntity.status(500).body(response);
        }
        return ResponseEntity.ok(response);
    }

    @PostMapping("/function-call")
    public ResponseEntity<String> functionCall(@RequestBody FunctionCallRequest request) {
        String result = chatGptService.callFunction(request, alphaVantageService);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/analyse-action-0")
    public ResponseEntity<String> analyseAction0(@RequestParam("symbol") String symbol, @RequestParam("delai") int delai) {
        String result = chatGptService.getAnalyseActionWithTwelveData(symbol, delai, twelveDataService);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/analyse-action")
    public ResponseEntity<String> analyseAction(@RequestParam("symbol") String symbol,
                                                @RequestParam("delai") int delai,
                                                @RequestParam("montant") int montant) {
        String data = twelveDataService.getDataAction(symbol);
        String sma = twelveDataService.getSMA(symbol);
        String rsi = twelveDataService.getRSI(symbol);
        String macd = twelveDataService.getMACD(symbol);
        String atr = twelveDataService.getATR(symbol);
        String financial = finnhubService.getFinancialData(symbol);
        String statistics = finnhubService.getDefaultKeyStatistics(symbol);
        String earnings = finnhubService.getEarnings(symbol);

        DataAction dataAction = new DataAction(
                symbol,
                delai,
                data,
                sma,
                rsi,
                macd,
                atr,
                financial,
                statistics,
                earnings,
                montant
        );

        String result = chatGptService.getAnalyseAction(dataAction);
        return ResponseEntity.ok(result);
    }

}
