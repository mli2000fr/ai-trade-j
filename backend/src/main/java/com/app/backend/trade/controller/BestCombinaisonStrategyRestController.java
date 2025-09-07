package com.app.backend.trade.controller;

import com.app.backend.trade.strategy.*;
import org.springframework.web.bind.annotation.*;
import org.ta4j.core.BarSeries;
import org.springframework.beans.factory.annotation.Autowired;
import java.util.*;

@RestController
@RequestMapping("/api/best-combination")
public class BestCombinaisonStrategyRestController {

    @Autowired
    private BestCombinaisonStrategyHelper bestCombinaisonStrategyHelper;

    /**
     * Endpoint pour tester la recherche de la meilleure combinaison de stratégies.
     * Exemple d'appel :
     * GET /api/best-combination/test?symbol=BTCUSDT&in=2&out=2
     */
    @GetMapping("/test")
    public BestCombinaisonStrategyHelper.BestCombinationResult testBestCombination(
            @RequestParam String symbol,
            @RequestParam int in,
            @RequestParam int out
    ) {
        return bestCombinaisonStrategyHelper.findBestCombination(symbol, in, out);
    }
}

