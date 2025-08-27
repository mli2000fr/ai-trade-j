package com.app.backend.trade.model;

import lombok.*;

@Data
@ToString
@NoArgsConstructor
public class DataAction {
    private String symbol;
    private int delai;
    private String data;
    private String sma;
    private String rsi;
    private String macd;
    private String atr;
    private String financial;
    private String statistics;
    private String earnings;
    private int montant;
    private String news;

    // Constructeur explicite pour garantir la compatibilité sans Lombok
    public DataAction(String symbol,
                      int montant,
                      int delai,
                      String data,
                      String sma,
                      String rsi,
                      String macd,
                      String atr,
                      String financial,
                      String statistics,
                      String earnings,
                      String news) {
        this.symbol = symbol;
        this.news = news;
        this.delai = delai;
        this.data = data;
        this.sma = sma;
        this.rsi = rsi;
        this.macd = macd;
        this.atr = atr;
        this.financial = financial;
        this.statistics = statistics;
        this.earnings = earnings;
        this.montant = montant;
    }

    public DataAction(String symbol,
                      String data,
                      String sma,
                      String rsi,
                      String macd,
                      String atr,
                      String financial,
                      String statistics,
                      String earnings,
                      String news) {
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
    }

    public String getSymbol() { return symbol; }
    public int getDelai() { return delai; }
    public String getData() { return data; }
    public String getSma() { return sma; }
    public String getRsi() { return rsi; }
    public String getMacd() { return macd; }
    public String getAtr() { return atr; }
    public String getFinancial() { return financial; }
    public String getStatistics() { return statistics; }
    public String getEarnings() { return earnings; }
    public int getMontant() { return montant; }
    public String getNews() { return news; }
}
