package com.app.backend.trade.model;

import com.app.backend.trade.strategy.ParamsOptim;
import lombok.Builder;
import lombok.Data;
import lombok.ToString;

@Builder
@Data
@ToString
public class ComboResult {
    private String symbol;
    private String entryName;
    private Object entryParams;
    private String exitName;
    private Object exitParams;
    private RiskResult result;
    private RiskResult checkResult;
    private ParamsOptim paramsOptim;
    // Ajout du rendement train pour le calcul du ratio d'overfit
    private double trainRendement;
}
