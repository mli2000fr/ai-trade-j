package com.app.backend.trade.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@ToString
public class OppositionOrder {
    private boolean oppositionFilled;
    private boolean oppositionActived;
    private boolean sideActived;

    public boolean isDayTrading() {
        return oppositionFilled || oppositionActived;
    }
}
