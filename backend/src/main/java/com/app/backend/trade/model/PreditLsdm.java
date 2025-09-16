package com.app.backend.trade.model;

import lombok.Builder;
import lombok.Data;

@Builder
@Data
public class PreditLsdm {
    private double lastClose;
    private double predictedClose;
    private SignalType signal;
    private String lastDate;
    private String position;

}
