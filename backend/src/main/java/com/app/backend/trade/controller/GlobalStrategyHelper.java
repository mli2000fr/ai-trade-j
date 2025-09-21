package com.app.backend.trade.controller;


import com.app.backend.trade.model.BestCombinationResult;
import com.app.backend.trade.model.MixResultat;
import com.app.backend.trade.model.SymbolPerso;
import com.app.backend.trade.strategy.BestInOutStrategy;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Controller;

import java.util.ArrayList;
import java.util.List;

@Controller
public class GlobalStrategyHelper {

    @Value("${signal.mix.active}")
    private boolean mixActive;

    @Value("${signal.lstm.active}")
    private boolean lstmActive;

    @Value("${signal.single.active}")
    private boolean singleActive;

    @Autowired
    private StrategieHelper strategieHelper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Autowired
    private BestCombinationStrategyHelper bestCombinationStrategyHelper;

    public List<MixResultat> getBestScoreAction(Integer limit, String type, String sort, String search, Boolean topProfil, Boolean buyOnly) {

        if(search != null && !search.isEmpty()) {
            return getBestScoreActionSingle(limit, sort, search, topProfil, null);
        }else if(type != null && type.equals("mix")) {
            return getBestScoreActionMix(limit, sort, topProfil, buyOnly);
        } else {
            return getBestScoreActionSingle(limit, sort, null, topProfil, buyOnly);
        }
    }

    public List<MixResultat> getBestScoreActionSingle(Integer limit, String sort, String search, Boolean topProfil, Boolean buyOnly) {

        List<BestInOutStrategy> listeSingle = strategieHelper.getBestPerfActions(limit, sort, search, topProfil);
        List<MixResultat> results = new ArrayList<>();
        for(BestInOutStrategy single : listeSingle) {

            if(buyOnly != null && buyOnly) {
                //strategieHelper.getBestInOutSignal()
            }

            BestCombinationResult mix = bestCombinationStrategyHelper.getBestCombinationResult(single.getSymbol());
            results.add(MixResultat.builder()
                     .name(single.getName())
                    .single(single)
                    .mix(mix == null ? BestCombinationResult.empty() : mix)
                    .build());
        }
        return results;
    }

    public List<MixResultat> getBestScoreActionMix(Integer limit, String sort, Boolean filtered, Boolean buyOnly) {

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

    public List<SymbolPerso> getSymbolsPerso() {
        String sql = "SELECT * FROM trade_ai.symbol_perso;";
        return jdbcTemplate.query(sql, (rs, rowNum) -> SymbolPerso.builder()
                        .symbols(rs.getString("symbols").replaceAll(" ", ""))
                .name(rs.getString("name"))
                .id(rs.getString("id"))
                .build()
        );
    }

}
