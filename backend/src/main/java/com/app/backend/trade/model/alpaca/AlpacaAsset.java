package com.app.backend.trade.model.alpaca;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AlpacaAsset {
    private String id;
    private String symbol;
    private String exchange;
    private String status;
    private String name;

}

