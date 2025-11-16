package com.app.backend.trade.controller;

import com.app.backend.trade.portfolio.MultiSymbolPortfolioManager;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/portfolio")
public class PortfolioTrainingController {

    private final MultiSymbolPortfolioManager portfolioManager;

    public PortfolioTrainingController(MultiSymbolPortfolioManager portfolioManager) {
        this.portfolioManager = portfolioManager;
    }

    // DTO requête
    public static class TrainRequest {
        public String tri;            // ex: "daily"
        public List<String> symbols;  // univers
    }

    @PostMapping("/train")
    public ResponseEntity<Map<String,Object>> train(@RequestBody TrainRequest req) {
        String tri = (req.tri == null || req.tri.isBlank()) ? "daily" : req.tri;
        Map<String,Object> result = portfolioManager.trainModels(tri, req.symbols);
        result.put("tri", tri);
        return ResponseEntity.ok(result);
    }
}

