package com.app.backend.trade.model;

import java.util.List;
import java.util.Map;

public class PortfolioDto {
    private List<Map<String, Object>> positions;
    private Map<String, Object> account;

    public PortfolioDto() {}

    public PortfolioDto(List<Map<String, Object>> positions, Map<String, Object> account) {
        this.positions = positions;
        this.account = account;
    }

    public List<Map<String, Object>> getPositions() {
        return positions;
    }

    public void setPositions(List<Map<String, Object>> positions) {
        this.positions = positions;
    }

    public Map<String, Object> getAccount() {
        return account;
    }

    public void setAccount(Map<String, Object> account) {
        this.account = account;
    }
}
