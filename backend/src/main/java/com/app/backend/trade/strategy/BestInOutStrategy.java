package com.app.backend.trade.strategy;

import com.app.backend.model.RiskResult;
import lombok.*;

@Builder
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class BestInOutStrategy {
    public String symbol;
    public String entryName;
    public Object entryParams;
    public String exitName;
    public Object exitParams;
    public RiskResult result;
    public ParamsOptim paramsOptim;
}

