package com.app.backend.trade.model;

import lombok.Builder;
import lombok.Data;

@Builder
@Data
public class IndicateurTech {
    private String date;
    private Double value;
}
