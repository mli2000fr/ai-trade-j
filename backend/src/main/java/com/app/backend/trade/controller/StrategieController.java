package com.app.backend.trade.controller;

import com.app.backend.trade.model.SetActiveStrategiesRequest;
import com.app.backend.trade.model.SetCombinationModeRequest;
import com.app.backend.trade.model.SignalType;
import com.app.backend.trade.model.StrategyListDto;
import com.app.backend.trade.service.StrategyService;
import com.app.backend.trade.strategy.BestInOutStrategy;
import com.app.backend.trade.strategy.StrategyManager;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/stra")
public class StrategieController {
    private final StrategyService strategyService;
    private final StrategieHelper strategieHelper;

    @Autowired
    public StrategieController(StrategieHelper strategieHelper,
                           StrategyService strategyService) {
        this.strategieHelper = strategieHelper;
        this.strategyService = strategyService;
    }
    /**
     * Liste les stratégies disponibles, actives et le mode de combinaison.
     */
    @GetMapping("/strategies")
    public ResponseEntity<StrategyListDto> getStrategies() {
        return ResponseEntity.ok(new StrategyListDto(
                strategyService.getAllStrategyNames(),
                strategyService.getActiveStrategyNames(),
                strategyService.getStrategyManager().getCombinationMode().name(),
                strategyService.getLogs()
        ));
    }

    /**
     * Modifie dynamiquement les stratégies actives.
     */
    @PostMapping("/strategies/active")
    public ResponseEntity<Void> setActiveStrategies(@RequestBody SetActiveStrategiesRequest req) {
        strategyService.setActiveStrategiesByNames(req.getStrategyNames());
        return ResponseEntity.ok().build();
    }

    /**
     * Modifie dynamiquement le mode de combinaison.
     */
    @PostMapping("/strategies/mode")
    public ResponseEntity<Void> setCombinationMode(@RequestBody SetCombinationModeRequest req) {
        strategyService.setCombinationMode(StrategyManager.CombinationMode.valueOf(req.getCombinationMode()));
        return ResponseEntity.ok().build();
    }

    /**
     * Teste le signal combiné sur une série de prix de clôture fournie.
     * Body: { "closePrices": [123.4, 124.1, ...], "isEntry": true }
     * Retourne true si le signal est validé sur la dernière barre.
     */
    @GetMapping("/strategies/test-signal")
    public ResponseEntity<Map<String, Object>> testCombinedSignal(@RequestParam(value = "symbol", required = false) String symbol,
                                                                  @RequestParam(value = "isEntry", required = false) Boolean isEntry) {
        boolean result = strategieHelper.testCombinedSignalOnClosePrices(symbol, isEntry);
        return ResponseEntity.ok(Collections.singletonMap("signal", result));
    }

    @GetMapping("/strategies/db/update-assets")
    public ResponseEntity<Boolean> updateDBAssets() {
        strategieHelper.updateDBAssets();
        return ResponseEntity.ok(true);
    }

    @GetMapping("/strategies/db/update-daily-valu")
    public ResponseEntity<Boolean> updateDBDailyValue() {
        strategieHelper.updateDBDailyValuAllSymbols();
        return ResponseEntity.ok(true);
    }


    @GetMapping("/strategies/db/update-daily-valu-complement")
    public ResponseEntity<Boolean> updateDBDailyValueComplement() {
        strategieHelper.updateDBDailyValuAllSymbolsComplement();
        return ResponseEntity.ok(true);
    }



    @GetMapping("/strategies/test_croised_strategies")
    public ResponseEntity<Boolean> testAllCrossedStrategies(@RequestParam(value = "symbol", required = false) String symbol) {
        strategieHelper.testAllCrossedStrategies(symbol);
        return ResponseEntity.ok(true);
    }


    @GetMapping("/strategies/calcul_croised_strategies")
    public ResponseEntity<Boolean> calculCroisedStrategies() {
        strategieHelper.calculCroisedStrategies();
        return ResponseEntity.ok(true);
    }


    @GetMapping("/strategies/best_performance_actions")
    public ResponseEntity<List<BestInOutStrategy>> getBestPerfActions(@RequestParam(value = "limit", required = false) Integer limit) {
        List<BestInOutStrategy> lsiteStr = strategieHelper.getBestPerfActions(limit);
        return ResponseEntity.ok(lsiteStr);
    }

    @GetMapping("/strategies/get_indice")
    public ResponseEntity<SignalType> getIndice(@RequestParam(value = "symbol", required = false) String symbol) {
        SignalType st = strategieHelper.getBestInOutSignal(symbol);
        return ResponseEntity.ok(st);
    }
}
