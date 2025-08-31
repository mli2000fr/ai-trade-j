package com.app.backend.trade.model.alpaca;

public class ErrResponseOrder {
    private String available;
    private int code;
    private String existing_qty;
    private String held_for_orders;
    private String existing_order_id;
    private String reject_reason;
    private String message;
    private String symbol;

    public String getExisting_order_id() {
        return existing_order_id;
    }

    public void setExisting_order_id(String existing_order_id) {
        this.existing_order_id = existing_order_id;
    }

    public String getReject_reason() {
        return reject_reason;
    }

    public void setReject_reason(String reject_reason) {
        this.reject_reason = reject_reason;
    }

    public String getAvailable() {
        return available;
    }
    public void setAvailable(String available) {
        this.available = available;
    }
    public int getCode() {
        return code;
    }
    public void setCode(int code) {
        this.code = code;
    }
    public String getExisting_qty() {
        return existing_qty;
    }
    public void setExisting_qty(String existing_qty) {
        this.existing_qty = existing_qty;
    }
    public String getHeld_for_orders() {
        return held_for_orders;
    }
    public void setHeld_for_orders(String held_for_orders) {
        this.held_for_orders = held_for_orders;
    }
    public String getMessage() {
        return message;
    }
    public void setMessage(String message) {
        this.message = message;
    }
    public String getSymbol() {
        return symbol;
    }
    public void setSymbol(String symbol) {
        this.symbol = symbol;
    }
}

