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

    public List<SymbolPerso> getSymbolsPerso() {
        String sql = "SELECT * FROM trade_ai.symbol_perso order by created_at DESC;";
        return jdbcTemplate.query(sql, (rs, rowNum) -> SymbolPerso.builder()
                        .symbols(rs.getString("symbols").replaceAll(" ", ""))
                .name(rs.getString("name"))
                .date(rs.getDate("created_at") == null ? null :
                    rs.getDate("created_at").toLocalDate().format(java.time.format.DateTimeFormatter.ofPattern(FORMAT_DATE)))
                .id(rs.getString("id"))
                .build()
        );
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



    public String getSymbolBuy() {
        java.time.LocalDateTime todayDateTime = java.time.LocalDateTime.now();
        java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(todayDateTime.toLocalDate());
        java.time.format.DateTimeFormatter formatter = java.time.format.DateTimeFormatter.ofPattern(FORMAT_DATE);
        String lastTradingDayStr = lastTradingDay.format(formatter);

        List<SymbolPerso> listSymPerso = this.getSymbolsPerso();
        boolean isExistant = listSymPerso.stream().anyMatch(sp -> lastTradingDayStr.equals(sp.getDate())
            && sp.getName().startsWith(NOM_SYM_BUY));
        if(isExistant){
            return "existant";
        }

        String sql = "select symbol from trade_ai.best_in_out_single_strategy;";
        List<String> listeSym = jdbcTemplate.query(sql, (rs, rowNum) -> {
            return rs.getString("symbol");
        });
        List<String> listeSymBut = new ArrayList<>();
        int executedCount = 0;
        for(String symbol : listeSym) {
            SignalInfo single = strategieHelper.getBestInOutSignal(symbol);
            SignalInfo mix = bestCombinationStrategyHelper.getSignal(symbol);
            PreditLsdm predit = lstmHelper.getPredit(symbol, "rendement");
            if(single != null && mix != null
            && single.getType() == SignalType.BUY
            && mix.getType() == SignalType.BUY
                    && predit != null && predit.getSignal() == SignalType.BUY) {
                listeSymBut.add(symbol);
            }
            executedCount++;
            lastSymbolBuyCount = executedCount;
        }
        String symbolBuy = String.join(",", listeSymBut);
        String insertSql = "INSERT INTO symbol_perso (name, created_at, symbols) VALUES (?, ?, ?)";
        jdbcTemplate.update(insertSql,
                NOM_SYM_BUY + lastTradingDayStr,
                lastTradingDay,
                symbolBuy);
        return symbolBuy;
    }

    public int getLastSymbolBuyCount() {
        return lastSymbolBuyCount;
    }
}
