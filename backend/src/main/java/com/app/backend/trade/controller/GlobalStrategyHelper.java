package com.app.backend.trade.controller;


import com.app.backend.trade.model.BestCombinationResult;
import com.app.backend.trade.model.MixResultat;
import com.app.backend.trade.strategy.BestInOutStrategy;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;

import java.util.ArrayList;
import java.util.List;

@Controller
public class GlobalStrategyHelper {


    @Autowired
    private StrategieHelper strategieHelper;

    @Autowired
    private BestCombinationStrategyHelper bestCombinationStrategyHelper;

    public List<MixResultat> getBestScoreAction(Integer limit, String sort, Boolean filtered) {

        List<BestInOutStrategy> listeSingle = strategieHelper.getBestPerfActions(limit, sort, filtered);
        List<MixResultat> results = new ArrayList<>();
        for(BestInOutStrategy single : listeSingle) {
            BestCombinationResult mix = bestCombinationStrategyHelper.getBestCombinationResult(single.getSymbol());
            results.add(MixResultat.builder()
                    .single(single)
                    .mix(mix)
                    .build());
        }
        return results;
    }


}
