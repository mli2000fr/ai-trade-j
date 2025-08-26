package com.app.backend.trade.model;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.util.List;

@Builder
@Setter
@Getter
@ToString
public class Portfolio {

    private double cash;
    private List<Position> listPositions;

    public Portfolio(double cash, List<Position> listPositions) {
        this.cash = cash;
        this.listPositions = listPositions;
    }
}
