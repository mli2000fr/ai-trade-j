package com.app.backend.trade.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TradeRequest {
    private String symbol;
    private String action; // "buy" ou "sell"
    private double quantity;
    private String id;
    private boolean cancelOpposite;
    private boolean forceDayTrade;
}
