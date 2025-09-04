package com.app.backend.trade.controller;

import com.app.backend.trade.model.*;
import com.app.backend.trade.model.alpaca.AlpacaAsset;
import com.app.backend.trade.service.*;
import com.app.backend.trade.util.TradeUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.ta4j.core.BarSeries;
import org.ta4j.core.BaseBarSeries;
import org.ta4j.core.Rule;
import java.time.ZonedDateTime;
import java.util.List;


@Controller
public class StrategieHelper {

    private final AlpacaService alpacaService;
    private final StrategyService strategyService;

    @Autowired
    public StrategieHelper(AlpacaService alpacaService,
                           StrategyService strategyService) {
        this.alpacaService = alpacaService;
        this.strategyService = strategyService;
    }

    /**
     * Retourne un signal d'achat/vente combiné selon les stratégies actives et le mode choisi.
     * @param series série de prix (BarSeries)
     * @param isEntry true pour entrée (achat), false pour sortie (vente)
     * @return true si le signal est validé
     */
    public boolean getCombinedSignal(BarSeries series, int index, boolean isEntry) {
        Rule rule = isEntry ? strategyService.getEntryRule(series) : strategyService.getExitRule(series);
        boolean result = rule.isSatisfied(index);
        String log = "Test signal " + (isEntry ? "ENTREE" : "SORTIE") +
                " | index=" + index +
                " | prix=" + (series.getBar(index) != null ? series.getBar(index).getClosePrice() : "?") +
                " | stratégies actives=" + strategyService.getActiveStrategyNames() +
                " | mode=" + strategyService.getStrategyManager().getCombinationMode().name() +
                " | résultat=" + result;
        strategyService.addLog(log);
        return result;
    }


    public boolean testCombinedSignalOnClosePrices(String symbol, boolean isEntry) {

        List<DailyValue> listeValues = alpacaService.getHistoricalBars(symbol, TradeUtils.getStartDate(700));
        BarSeries series = toBarSeries(listeValues);
        int lastIndex = series.getEndIndex();
        return getCombinedSignal(series, lastIndex, isEntry);
    }

/*
    public void updateDailyValuAllSymbolsToSup(){
        List<String> listegetIexSymbols = this.alpacaService.get();
        int error = 0;
        for(int i = 4349; i < listegetIexSymbols.size(); i++){
            try{
                alpacaService.updateDailyValue(listegetIexSymbols.get(i));
                Thread.sleep(200);
            }catch(Exception e){
                error++;
            }
        }
    }
*/

    public void updateDBDailyValuAllSymbols(){
        List<String> listeDbSymbols = this.alpacaService.getAllAssetSymbolsFromDb();
        int error = 0;
        for(String symbol : listeDbSymbols){
            try{
                List<DailyValue> listeValues = alpacaService.updateDailyValue(symbol);
                for(DailyValue dv : listeValues){
                    alpacaService.insertDailyValue(symbol, dv);
                }
                Thread.sleep(200);
            }catch(Exception e){
                error++;
                TradeUtils.log("Erreur updateDailyValue("+symbol+") : " + e.getMessage());
            }
        }
        TradeUtils.log("updateDBDailyValuAllSymbols: total "+listeDbSymbols.size()+", error" + error);
    }

    public void updateDBAssets(){
        List<AlpacaAsset> listeSymbols = this.alpacaService.getIexSymbolsFromAlpaca();
        this.alpacaService.saveSymbolsToDatabase(listeSymbols);
    }

    /**
     * Convertit une liste de DailyValue en BarSeries (ta4j).
     */
    private BarSeries toBarSeries(List<DailyValue> values) {
        BarSeries series = new BaseBarSeries();
        for (DailyValue v : values) {
            try {
                series.addBar(
                        ZonedDateTime.parse(v.getDate()),
                        Double.parseDouble(v.getOpen()),
                        Double.parseDouble(v.getHigh()),
                        Double.parseDouble(v.getLow()),
                        Double.parseDouble(v.getClose()),
                        Double.parseDouble(v.getVolume())
                );
            } catch (Exception e) {
                TradeUtils.log("Erreur conversion DailyValue en BarSeries: " + e.getMessage());
            }
        }
        return series;
    }
}
