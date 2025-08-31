package com.app.backend.trade.model;

public class TradeRequest {
    private String symbol;
    private String action; // "buy" ou "sell"
    private double quantity;
    private String id;
    private boolean cancelOpposite;
    private boolean forceDayTrade;

    public boolean isCancelOpposite() {
        return cancelOpposite;
    }

    public boolean isForceDayTrade() {
        return forceDayTrade;
    }

    public void setForceDayTrade(boolean forceDayTrade) {
        this.forceDayTrade = forceDayTrade;
    }

    public void setCancelOpposite(boolean cancelOpposite) {
        this.cancelOpposite = cancelOpposite;
    }

    public String getSymbol() {
        return symbol;
    }
    public void setSymbol(String symbol) {
        this.symbol = symbol;
    }
    public String getAction() {
        return action;
    }
    public void setAction(String action) {
        this.action = action;
    }
    public double getQuantity() {
        return quantity;
    }
    public void setQuantity(double quantity) {
        this.quantity = quantity;
    }
    public String getId() {
        return id;
    }
    public void setId(String id) {
        this.id = id;
    }
}
