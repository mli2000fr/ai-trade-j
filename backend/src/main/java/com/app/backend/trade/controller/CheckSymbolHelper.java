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
public class CheckSymbolHelper {


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

    private String getLastTradingDayStr() {
        java.time.LocalDateTime todayDateTime = java.time.LocalDateTime.now();
        java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(todayDateTime.toLocalDate());
        java.time.format.DateTimeFormatter formatter = java.time.format.DateTimeFormatter.ofPattern(FORMAT_DATE);
        return lastTradingDay.format(formatter);
    }
    public List<String> getSymbols() {
        return getSymbols(null);
    }
    public List<String> getSymbols(String prefix) {
        if(prefix != null && !prefix.isEmpty()) {
            String lastTradingDayStr = getLastTradingDayStr();
            List<SymbolPerso> listSymPerso = this.getSymbolsPerso();
            boolean isExistant = listSymPerso.stream().anyMatch(sp -> lastTradingDayStr.equals(sp.getDate())
                    && sp.getName().startsWith(prefix));
            if(isExistant){
                throw new RuntimeException("Symbol buy for date " + lastTradingDayStr + " already exists.");
            }
        }
        String sql = "select symbol from trade_ai.best_in_out_single_strategy;";
        return jdbcTemplate.query(sql, (rs, rowNum) -> {
            return rs.getString("symbol");
        });
    }
    public void saveToPerso(String symbols) {
        String lastTradingDayStr = getLastTradingDayStr();
        java.time.LocalDateTime todayDateTime = java.time.LocalDateTime.now();
        java.time.LocalDate lastTradingDay = TradeUtils.getLastTradingDayBefore(todayDateTime.toLocalDate());
        String insertSql = "INSERT INTO symbol_perso (name, created_at, symbols) VALUES (?, ?, ?)";
        jdbcTemplate.update(insertSql,
                NOM_SYM_BUY + lastTradingDayStr,
                lastTradingDay,
                symbols);
    }

    public String getSymbolBuy() {
        List<String> listeSym = this.getSymbols(NOM_SYM_BUY);
        List<String> listeSymBuy = new ArrayList<>();
        int executedCount = 0;
        for(String symbol : listeSym) {
            SignalInfo single = strategieHelper.getBestInOutSignal(symbol);
            SignalInfo mix = bestCombinationStrategyHelper.getSignal(symbol);
            PreditLsdm predit = lstmHelper.getPredit(symbol, "rendement");
            if(single != null && mix != null
            && single.getType() == SignalType.BUY
            && mix.getType() == SignalType.BUY
                    && predit != null && predit.getSignal() == SignalType.BUY) {
                listeSymBuy.add(symbol);
            }
            executedCount++;
            lastSymbolBuyCount = executedCount;
        }
        String symbolBuy = String.join(",", listeSymBuy);
        this.saveToPerso(symbolBuy);
        return symbolBuy;
    }

    public int getLastSymbolBuyCount() {
        return lastSymbolBuyCount;
    }
    
    
    /*
        check by ai
     */
    public String checkSymbolByAgent() {
        
        
        
        return "NOT IMPLEMENTED";
    }
    
    
}
