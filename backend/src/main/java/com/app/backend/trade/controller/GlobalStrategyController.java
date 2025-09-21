package com.app.backend.trade.controller;


import com.app.backend.trade.model.MixResultat;
import com.app.backend.trade.model.SymbolPerso;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/result/")
public class GlobalStrategyController {


    @Autowired
    private GlobalStrategyHelper globalStrategyHelper;

    @GetMapping("/global")
    public List<MixResultat> getBestScoreAction(@RequestParam(value = "limit", required = false) Integer limit,
                                                @RequestParam(value = "type", required = false, defaultValue = "single") String type,
                                                @RequestParam(value = "sort", required = false, defaultValue = "rendement_score") String sort,
                                                @RequestParam(value = "topProfil", required = false) Boolean topProfil,
                                                @RequestParam(value = "buyOnly", required = false) Boolean buyOnly,
                                                @RequestParam(value = "search", required = false) String search) {
        return globalStrategyHelper.getBestScoreAction(limit, type, sort, search, topProfil, buyOnly);
    }

    @GetMapping("/infosSymbol")
    public MixResultat getInfosAction(@RequestParam(value = "symbol", required = true) String symbol) {
        return globalStrategyHelper.getInfos(symbol);
    }


    @GetMapping("/symbol_pero")
    public ResponseEntity<List<SymbolPerso>> getSymbolsPerso() {
        return ResponseEntity.ok(globalStrategyHelper.getSymbolsPerso());
    }



}