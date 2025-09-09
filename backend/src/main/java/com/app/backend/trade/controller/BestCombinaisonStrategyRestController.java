package com.app.backend.trade.controller;

import com.app.backend.trade.model.SignalType;
import com.app.backend.trade.strategy.*;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Autowired;
import java.util.*;

@RestController
@RequestMapping("/api/best-combination")
public class BestCombinaisonStrategyRestController {

    @Autowired
    private BestCombinationStrategyHelper bestCombinationStrategyHelper;


    @GetMapping("/best-signal")
    public SignalType getBestSignal(@RequestParam String symbol) {
        return bestCombinationStrategyHelper.getBestSignal(symbol);
    }

    @GetMapping("/calcul")
    public Boolean calculMixStrategies() {
        bestCombinationStrategyHelper.calculMixStrategies();
        return true;
    }

    @GetMapping("/calculScoreST")
    public ResponseEntity<Boolean> calculScoreSwingTrade() {
        bestCombinationStrategyHelper.calculScoreST();
        return ResponseEntity.ok(true);
    }

}
