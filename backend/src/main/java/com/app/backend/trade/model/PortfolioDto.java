package com.app.backend.trade.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PortfolioDto {
    private List<Map<String, Object>> positions;
    private Map<String, Object> account;
    private double initialDeposit;
}
