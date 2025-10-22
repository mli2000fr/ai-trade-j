package com.app.backend.trade.model;

import lombok.Builder;
import lombok.Data;

@Builder
@Data
public class GlobalIndice {
    private String symbol;
    private SignalType typeSingle;
    private SignalType typeMix;
    private SignalType typeLstm;
    private String positionLstm;
    private boolean isSell;
}

