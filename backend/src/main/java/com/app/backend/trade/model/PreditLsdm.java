package com.app.backend.trade.model;

import com.app.backend.trade.lstm.LstmTradePredictor;
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
    private String explication;
    private LstmTradePredictor.LoadedModel loadedModel;

}
