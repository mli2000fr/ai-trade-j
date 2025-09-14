package com.app.backend.trade.controller;

import com.app.backend.trade.model.DailyValue;
import com.app.backend.trade.model.SignalInfo;
import com.app.backend.trade.strategy.BestInOutStrategy;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/stra")
public class StrategieController {
    private final StrategieHelper strategieHelper;

    @Autowired
    public StrategieController(StrategieHelper strategieHelper) {
        this.strategieHelper = strategieHelper;
    }

    /**
     * Met à jour la base d'actifs en base de données.
     * @return true si la mise à jour s'est déroulée correctement
     */
    @GetMapping("/strategies/db/update-assets")
    public ResponseEntity<Boolean> updateDBAssets() {
        strategieHelper.updateDBAssets();
        return ResponseEntity.ok(true);
    }

    /**
     * Met à jour les valeurs journalières pour tous les symboles en base de données.
     * @return true si la mise à jour s'est déroulée correctement
     */
    @GetMapping("/strategies/db/update-daily-valu")
    public ResponseEntity<Boolean> updateDBDailyValue() {
        strategieHelper.updateDBDailyValuAllSymbols();
        return ResponseEntity.ok(true);
    }

    /**
     * Calcule les stratégies croisées sur l'ensemble des symboles.
     * @return true si le calcul s'est déroulé correctement
     */
    @GetMapping("/strategies/calcul_croised_strategies")
    public ResponseEntity<Boolean> calculCroisedStrategies() {
        strategieHelper.calculCroisedStrategies();
        return ResponseEntity.ok(true);
    }

    /**
     * Récupère la liste des meilleures performances d'actions selon le tri et la limite.
     * @param limit nombre maximum d'actions à retourner (optionnel)
     * @param sort critère de tri (par défaut rendement)
     * @return liste des meilleures stratégies BestInOutStrategy
     */
    @GetMapping("/strategies/best_performance_actions")
    public ResponseEntity<List<BestInOutStrategy>> getBestPerfActions(
            @RequestParam(value = "limit", required = false) Integer limit,
            @RequestParam(value = "sort", required = false, defaultValue = "rendement") String sort,
            @RequestParam(value = "filtered", required = false) Boolean filtered
    ) {
        List<BestInOutStrategy> lsiteStr = strategieHelper.getBestPerfActions(limit, sort, filtered);
        return ResponseEntity.ok(lsiteStr);
    }

    /**
     * Récupère le signal d'indice pour un symbole donné.
     * @param symbol symbole à analyser (optionnel)
     * @return type de signal (SignalType)
     */
    @GetMapping("/strategies/get_indice")
    public ResponseEntity<SignalInfo> getIndice(@RequestParam(value = "symbol", required = false) String symbol) {
        SignalInfo st = strategieHelper.getBestInOutSignal(symbol);
        return ResponseEntity.ok(st);
    }

    @GetMapping("/strategies/test")
    public ResponseEntity<BestInOutStrategy> test(@RequestParam(value = "symbol", required = false) String symbol) {
        BestInOutStrategy st = strategieHelper.optimseStrategy(symbol);
        return ResponseEntity.ok(st);
    }



    @GetMapping("/getBougiesBySymbol")
    public ResponseEntity<List<DailyValue>> getBougiesBySymbol(@RequestParam String symbol, @RequestParam int historique) {
        return ResponseEntity.ok(strategieHelper.getDailyValuesFromDb(symbol, historique));
    }

}
