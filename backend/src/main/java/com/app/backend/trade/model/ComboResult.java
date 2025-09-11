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
    private OptimResult result;
    private ParamsOptim paramsOptim;
}