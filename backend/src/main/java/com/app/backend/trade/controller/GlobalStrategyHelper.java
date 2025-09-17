package com.app.backend.trade.controller;


import com.app.backend.trade.model.BestCombinationResult;
import com.app.backend.trade.model.MixResultat;
import com.app.backend.trade.model.SymbolPerso;
import com.app.backend.trade.strategy.BestInOutStrategy;
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
    private BestCombinationStrategyHelper bestCombinationStrategyHelper;

    public List<MixResultat> getBestScoreAction(Integer limit, String type, String sort, String search, Boolean filtered) {

        if(search != null && !search.isEmpty()) {
            return getBestScoreActionSingle(limit, sort, search, filtered);
        }else if(type != null && type.equals("mix")) {
            return getBestScoreActionMix(limit, sort, filtered);
        } else {
            return getBestScoreActionSingle(limit, sort, null, filtered);
        }
    }

    public List<MixResultat> getBestScoreActionSingle(Integer limit, String sort, String search, Boolean filtered) {

        List<BestInOutStrategy> listeSingle = strategieHelper.getBestPerfActions(limit, sort, search, filtered);
        List<MixResultat> results = new ArrayList<>();
        for(BestInOutStrategy single : listeSingle) {
            BestCombinationResult mix = bestCombinationStrategyHelper.getBestCombinationResult(single.getSymbol());
            results.add(MixResultat.builder()
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
                        .symbols(rs.getString("symbols"))
                .name(rs.getString("name"))
                .id(rs.getString("id"))
                .build()
        );
    }


    public void rattrapage(){
        String sql = "SELECT symbol FROM trade_ai.best_in_out_single_strategy;";
        List<String> listSymbol =  jdbcTemplate.queryForList(sql, String.class);

        for(String symbol : listSymbol) {
            BestInOutStrategy best = strategieHelper.getBestInOutStrategy(symbol);
            String updateSql = """
                        UPDATE best_in_out_single_strategy SET
                            rendement_check = ?,
                            score_swing_trade_check = ?
                        WHERE symbol = ?
                    """;
            jdbcTemplate.update(updateSql,
                    best.check.rendement,
                    best.check.scoreSwingTrade,
                    symbol
            );
        }

        String sqlM = "SELECT symbol FROM trade_ai.best_in_out_mix_strategy;";
        List<String> listSymbolM =  jdbcTemplate.queryForList(sqlM, String.class);
        for(String symbol : listSymbolM) {
            BestCombinationResult mix = bestCombinationStrategyHelper.getBestCombinationResult(symbol);
            String updateSqlM = """
                        UPDATE best_in_out_mix_strategy SET
                            rendement_check = ?,
                            score_swing_trade_check = ?
                        WHERE symbol = ?
                    """;
            jdbcTemplate.update(updateSqlM,
                    mix.check.rendement,
                    mix.check.scoreSwingTrade,
                    symbol
            );
        }
    }
}
