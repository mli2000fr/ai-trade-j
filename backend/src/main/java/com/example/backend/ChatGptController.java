package com.example.backend;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/chatgpt")
public class ChatGptController {
    private final ChatGptService chatGptService;
    private final AlphaVantageService alphaVantageService;
    private final TwelveDataService twelveDataService;

    @Autowired
    public ChatGptController(ChatGptService chatGptService, AlphaVantageService alphaVantageService, TwelveDataService twelveDataService) {
        this.chatGptService = chatGptService;
        this.alphaVantageService = alphaVantageService;
        this.twelveDataService = twelveDataService;
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

    @GetMapping("/analyse-action")
    public ResponseEntity<String> analyseAction(@RequestParam String symbol, @RequestParam int delai) {
        String result = chatGptService.getAnalyseActionWithTwelveData(symbol, delai, twelveDataService);
        return ResponseEntity.ok(result);
    }
}
