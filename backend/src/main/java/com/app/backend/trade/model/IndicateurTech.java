package com.app.backend.trade.model;

import lombok.Builder;
import lombok.Data;

@Builder
@Data
public class IndicateurTech {
    private Integer id;
    private String indicateurName;
    private String dateStr;
    private Double value;
}
