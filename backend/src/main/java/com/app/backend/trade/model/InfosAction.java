package com.app.backend.trade.model;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;

@Data
@ToString
@NoArgsConstructor
public class InfosAction {
    private String symbol;
    private String data;
    private String sma;
    private String rsi;
    private String macd;
    private String atr;
    private String financial;
    private String statistics;
    private String earnings;
    private String news;
    private String portfolio;

    // Constructeur explicite pour garantir la compatibilité sans Lombok
    public InfosAction(String symbol,
                       String data,
                       String sma,
                       String rsi,
                       String macd,
                       String atr,
                       String financial,
                       String statistics,
                       String earnings,
                       String news,
                       String portfolio) {
        this.symbol = symbol;
        this.news = news;
        this.data = data;
        this.sma = sma;
        this.rsi = rsi;
        this.macd = macd;
        this.atr = atr;
        this.financial = financial;
        this.statistics = statistics;
        this.earnings = earnings;
        this.portfolio = portfolio;
    }


    public String getSymbol() { return symbol; }
    public String getData() { return data; }
    public String getSma() { return sma; }
    public String getRsi() { return rsi; }
    public String getMacd() { return macd; }
    public String getAtr() { return atr; }
    public String getFinancial() { return financial; }
    public String getStatistics() { return statistics; }
    public String getEarnings() { return earnings; }
    public String getNews() { return news; }
    public String getPortfolio() { return portfolio; }
}
