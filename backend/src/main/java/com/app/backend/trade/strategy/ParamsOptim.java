package com.app.backend.trade.strategy;

import lombok.*;

@Builder
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class ParamsOptim {
    public double initialCapital;
    public double riskPerTrade;
    public double stopLossPct;
    public double takeProfitPct;
    public int nbSimples;
}
