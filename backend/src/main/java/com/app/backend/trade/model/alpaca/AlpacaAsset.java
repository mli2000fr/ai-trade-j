package com.app.backend.trade.model.alpaca;

public class AlpacaAsset {
    private String id;
    private String symbol;
    private String exchange;
    private String status;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getSymbol() { return symbol; }
    public void setSymbol(String symbol) { this.symbol = symbol; }
    public String getExchange() { return exchange; }
    public void setExchange(String exchange) { this.exchange = exchange; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
}

