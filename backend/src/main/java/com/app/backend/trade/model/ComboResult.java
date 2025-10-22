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
    private RiskResult finalResult;
    private RiskResult testResult;
    private ParamsOptim paramsOptim;
}
