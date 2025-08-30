package com.app.backend.trade.model;
import java.util.List;

public class TradeAutoRequestGpt {
    private String id;
    private List<String> symboles;
    private String analyseGpt;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public List<String> getSymboles() { return symboles; }
    public void setSymboles(List<String> symboles) { this.symboles = symboles; }
    public String getAnalyseGpt() { return analyseGpt; }
    public void setAnalyseGpt(String analyseGpt) { this.analyseGpt = analyseGpt; }
}

