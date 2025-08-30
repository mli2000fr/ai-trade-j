package com.app.backend.trade.model;

import java.util.List;

public class TradeAutoRequest {
    private List<String> symboles;
    private String id;

    public List<String> getSymboles() {
        return symboles;
    }

    public void setSymboles(List<String> symboles) {
        this.symboles = symboles;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }
}
