package com.app.backend.trade.model;


import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Builder
@Setter
@Getter
@ToString
public class OrderRequest {
    public String symbol;
    public Double qty;
    public Double quantity;
    public String side;
    public String action;
    public Double priceLimit;
    public Double price_limit;
    public Double stopLoss;
    public Double stop_loss;
    public Double takeProfit;
    public Double take_profit;
    public boolean executeNow = true;
    public String statut = "";

    // Méthode utilitaire pour normaliser les champs
    public void normalize() {
        if (side == null && action != null) side = action;
        if (qty == null && quantity != null) qty = quantity;
        if (priceLimit == null && price_limit != null) priceLimit = price_limit;
        if (stopLoss == null && stop_loss != null) stopLoss = stop_loss;
        if (takeProfit == null && take_profit != null) takeProfit = take_profit;
    }
}

