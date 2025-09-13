package com.app.backend.trade.model;

import java.sql.Date;

public class SignalInfo {
    private final String symbol;
    private final SignalType type;
    private final Date date;

    public SignalInfo(String symbol, SignalType type, Date date) {
        this.symbol = symbol;
        this.type = type;
        this.date = date;
    }

    public String getSymbol() {
        return symbol;
    }

    public SignalType getType() {
        return type;
    }

    public Date getDate() {
        return date;
    }
}

