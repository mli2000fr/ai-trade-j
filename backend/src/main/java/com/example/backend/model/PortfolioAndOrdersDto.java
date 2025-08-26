package com.example.backend.model;

import java.util.List;
import java.util.Map;

public class PortfolioAndOrdersDto {
    private List<Map<String, Object>> positions;
    private List<Map<String, Object>> orders;
    private Map<String, Object> account;

    public PortfolioAndOrdersDto() {}

    public PortfolioAndOrdersDto(List<Map<String, Object>> positions, List<Map<String, Object>> orders, Map<String, Object> account) {
        this.positions = positions;
        this.orders = orders;
        this.account = account;
    }

    public List<Map<String, Object>> getPositions() {
        return positions;
    }

    public void setPositions(List<Map<String, Object>> positions) {
        this.positions = positions;
    }

    public List<Map<String, Object>> getOrders() {
        return orders;
    }

    public void setOrders(List<Map<String, Object>> orders) {
        this.orders = orders;
    }

    public Map<String, Object> getAccount() {
        return account;
    }

    public void setAccount(Map<String, Object> account) {
        this.account = account;
    }
}
