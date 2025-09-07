package com.app.backend.trade.controller;

import com.app.backend.trade.model.BestCombinationResult;
import com.app.backend.trade.model.SignalType;
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
    public BestCombinationResult testBestCombination(
            @RequestParam String symbol,
            @RequestParam int in,
            @RequestParam int out
    ) {
        return bestCombinaisonStrategyHelper.findBestCombination(symbol, in, out);
    }


    /**
     * Endpoint REST pour obtenir la meilleure combinaison globale de stratégies.
     * Exemple d'appel : GET /api/best-combination-global?symbol=BTCUSDT
     */
    @GetMapping("/global")
    @ResponseBody
    public BestCombinationResult getBestCombinationGlobal(@RequestParam("symbol") String symbol) {
        return bestCombinaisonStrategyHelper.findBestCombinationGlobal(symbol);
        //http://localhost:8080/api/best-combination/global?symbol=AAPL
    }

    /**
     * Endpoint REST pour obtenir le signal (BUY, SELL, NONE) pour un symbole donné.
     * Exemple d'appel : GET /api/best-combination/signal?symbol=BTCUSDT
     */
    @GetMapping("/signal")
    public SignalType getSignal(@RequestParam String symbol) {
        return bestCombinaisonStrategyHelper.getSignal(symbol);
    }

    @GetMapping("/best-signal")
    public SignalType getBestSignal(@RequestParam String symbol) {
        return bestCombinaisonStrategyHelper.getBestSignal(symbol);
    }

    @GetMapping("/calcul")
    public Boolean calculMixStrategies() {
        bestCombinaisonStrategyHelper.calculMixStrategies();
        return true;
    }

}
