package com.app.backend.trade.controller;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/filtre/")
public class FiltrerSymbolController {

    @Autowired
    private FiltrerSymbolHelper filtrerSymbolHelper;

    /**
     * Lance le calcul et la sauvegarde des métriques swing trade pour tous les symboles éligibles.
     * @param topN nombre de meilleurs symboles à retourner (optionnel, défaut 20)
     * @return liste des meilleurs symboles swing trade
     */
    @GetMapping("/swing-metrics") //http://localhost:8080/api/filtre/swing-metrics
    public String calculerEtSauvegarderSwingMetrics() {
        filtrerSymbolHelper.triBestSwingTradeSymbols();
        return "job done";
    }
}
