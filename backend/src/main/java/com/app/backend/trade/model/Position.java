package com.app.backend.trade.model;

import com.google.gson.annotations.SerializedName;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Builder
@Setter
@Getter
@ToString
public class Position {

    private String symbol;
    private int quantity;
    @SerializedName("cost_basis")
    private double costBasis;
    @SerializedName("market_Value")
    private double marketValue;

    public Position(String symbol, int quantity, double costBasis, double marketValue) {
        this.symbol = symbol;
        this.quantity = quantity;
        this.costBasis = costBasis;
        this.marketValue = marketValue;
    }
}
