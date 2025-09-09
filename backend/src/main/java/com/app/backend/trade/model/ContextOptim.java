package com.app.backend.trade.model;

import lombok.*;

@Builder
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class ContextOptim {
    public double initialCapital;
    public double riskPerTrade;
    public double stopLossPct;
    public double takeProfitPct;
    public int nbSimples;
}
