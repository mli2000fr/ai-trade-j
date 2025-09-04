package com.app.backend.trade.model;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.ToString;

@Data
@ToString
@NoArgsConstructor
@AllArgsConstructor
public class InfosAction {
    private Double lastPrice;
    private String symbol;
    private String historical;
    private String sma200;
    private String ema20;
    private String ema50;
    private String rsi;
    private String macd;
    private String atr;
    private String financial;
    private String statistics;
    private String earnings;
    private String news;
    private String portfolio;
}
