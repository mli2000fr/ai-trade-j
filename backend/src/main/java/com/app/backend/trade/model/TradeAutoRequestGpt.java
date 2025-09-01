package com.app.backend.trade.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TradeAutoRequestGpt {
    private String id;
    private List<String> symboles;
    private String analyseGpt;
}
