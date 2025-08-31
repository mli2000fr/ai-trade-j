package com.app.backend.trade.model.alpaca;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import com.google.gson.annotations.SerializedName;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class AlpacaTransferActivity {
    @SerializedName("net_amount")
    private String netAmount;
    @SerializedName("activity_type")
    private String activityType;
}
