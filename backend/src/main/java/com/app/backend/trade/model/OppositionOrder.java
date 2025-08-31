package com.app.backend.trade.model;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

@Builder
@Setter
@Getter
@ToString
public class OppositionOrder {

    private boolean oppositionFilled;
    private boolean oppositionActived;

    public boolean isDayTrading() {
        return oppositionFilled || oppositionActived;
    }
}
