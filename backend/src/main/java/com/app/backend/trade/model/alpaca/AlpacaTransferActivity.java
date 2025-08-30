package com.app.backend.trade.model.alpaca;

import com.google.gson.annotations.SerializedName;

public class AlpacaTransferActivity {
    @SerializedName("net_amount")
    private String netAmount;
    @SerializedName("activity_type")
    private String activityType;

    public String getNetAmount() {
        return netAmount;
    }

    public void setNetAmount(String netAmount) {
        this.netAmount = netAmount;
    }

    public String getActivityType() {
        return activityType;
    }

    public void setActivityType(String activityType) {
        this.activityType = activityType;
    }
}
