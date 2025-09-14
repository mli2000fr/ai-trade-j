package com.app.backend.trade.controller;


import com.app.backend.trade.model.MixResultat;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/result")
public class GlobalStrategyController {


    @Autowired
    private GlobalStrategyHelper globalStrategyHelper;

    @GetMapping("/global")
    public List<MixResultat> getBestScoreAction(@RequestParam(value = "limit", required = false) Integer limit,
                                                @RequestParam(value = "sort", required = false, defaultValue = "rendement_score") String sort,
                                                @RequestParam(value = "filtered", required = false) Boolean filtered) {
        return globalStrategyHelper.getBestScoreAction(limit, sort, filtered);
    }

}