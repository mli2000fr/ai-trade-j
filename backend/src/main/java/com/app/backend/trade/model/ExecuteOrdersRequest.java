package com.app.backend.trade.model;

import java.util.List;

public class ExecuteOrdersRequest {
    private String id;
    private List<OrderRequest> orders;

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public List<OrderRequest> getOrders() { return orders; }
    public void setOrders(List<OrderRequest> orders) { this.orders = orders; }
}

