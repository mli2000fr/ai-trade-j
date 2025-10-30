package com.app.backend.trade.controller;


import com.app.backend.trade.model.*;
import com.app.backend.trade.strategy.BestInOutStrategy;
import com.app.backend.trade.util.TradeUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Controller;

import java.util.ArrayList;
import java.util.List;

@Controller
public class GlobalStrategyHelper {


    @Autowired
    private StrategieHelper strategieHelper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Autowired
    private LstmHelper lstmHelper;

    @Autowired
    private BestCombinationStrategyHelper bestCombinationStrategyHelper;

    private static final String FORMAT_DATE = "dd_MM_yy";
    private static final String NOM_SYM_BUY = "TopBuy_";

    private int lastSymbolBuyCount = 0;

    public List<MixResultat> getBestScoreAction(Integer limit, String type, String sort, String search, Boolean topProfil, Boolean topClassement) {

        if(search != null && !search.isEmpty()) {
            return getBestScoreActionSingle(limit, sort, search, topProfil);
        }else if(type != null && type.equals("single")) {
            return getBestScoreActionSingle(limit, sort, null, topProfil);
        }else if(type != null && type.equals("mix")) {
            return getBestScoreActionMix(limit, sort, topProfil);
        } else {
            return getBestScoreActionLstm(limit, sort, topClassement);
        }
    }

    public List<MixResultat> getBestScoreActionLstm(Integer limit, String sort, Boolean topClassement) {
        List<String> listeSym = lstmHelper.getBestModel(limit, sort, topClassement);
        List<MixResultat> results = new ArrayList<>();
        for(String symbol : listeSym) {
            BestInOutStrategy single = strategieHelper.getBestInOutStrategy(symbol);
            BestCombinationResult mix = bestCombinationStrategyHelper.getBestCombinationResult(symbol);
            if(single == null && mix == null) {
                continue;
            }
            results.add(MixResultat.builder()
                    .name(single == null ? mix.getName() : single.getName())
                    .single(single == null ? BestInOutStrategy.empty() : single)
                    .mix(mix == null ? BestCombinationResult.empty() : mix)
                    .build());
        }
        return results;
    }

    public List<MixResultat> getBestScoreActionSingle(Integer limit, String sort, String search, Boolean topProfil) {

        List<BestInOutStrategy> listeSingle = strategieHelper.getBestPerfActions(limit, sort, search, topProfil);
        List<MixResultat> results = new ArrayList<>();
        for(BestInOutStrategy single : listeSingle) {
            BestCombinationResult mix = bestCombinationStrategyHelper.getBestCombinationResult(single.getSymbol());
            results.add(MixResultat.builder()
                     .name(single.getName())
                    .single(single)
                    .mix(mix == null ? BestCombinationResult.empty() : mix)
                    .build());
        }
        return results;
    }

    public List<MixResultat> getBestScoreActionMix(Integer limit, String sort, Boolean filtered) {

        List<BestCombinationResult> listeMix = bestCombinationStrategyHelper.getBestPerfActions(limit, sort, filtered);

        List<MixResultat> results = new ArrayList<>();
        for(BestCombinationResult mix : listeMix) {
            BestInOutStrategy single = strategieHelper.getBestInOutStrategy(mix.getSymbol());
            results.add(MixResultat.builder()
                    .name(mix.getName())
                    .single(single == null ? BestInOutStrategy.empty() : single)
                    .mix(mix)
                    .build());
        }
        return results;
    }

    public MixResultat getInfos(String symbol) {
        BestInOutStrategy single = strategieHelper.getBestInOutStrategy(symbol);
        BestCombinationResult mix = bestCombinationStrategyHelper.getBestCombinationResult(symbol);
        return MixResultat.builder()
                .single(single == null ? BestInOutStrategy.empty() : single)
                .mix(mix == null ? BestCombinationResult.empty() : mix)
                .build();
    }

    public GlobalIndice getIndice(String symbol){
        SignalInfo singleS = strategieHelper.getBestInOutSignal(symbol);
        SignalInfo mixS = bestCombinationStrategyHelper.getSignal(symbol);
        PreditLsdm preditLsdm = lstmHelper.getPredit(symbol, "rendement");
        return GlobalIndice.builder()
                .typeSingle(singleS.getType())
                .typeMix(mixS.getType())
                .typeLstm(preditLsdm.getSignal())
                .symbol(symbol)
                .positionLstm(preditLsdm.getPosition())
                .isSell(SignalType.SELL.name().equals(preditLsdm.getSignal().name()))
                .build();

    }


}
