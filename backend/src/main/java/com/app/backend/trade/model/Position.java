package com.app.backend.trade.model;

import com.google.gson.annotations.SerializedName;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@ToString
public class Position {

    private String symbol;
    private int quantity;
    @SerializedName("cost_basis")
    private double costBasis;
    @SerializedName("market_Value")
    private double marketValue;
}
