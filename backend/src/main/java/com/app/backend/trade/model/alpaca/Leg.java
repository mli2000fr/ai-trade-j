package com.app.backend.trade.model.alpaca;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import com.google.gson.annotations.SerializedName;
import java.time.OffsetDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Leg {
    private String id;
    @SerializedName("client_order_id")
    private String clientOrderId;
    @SerializedName("created_at")
    private OffsetDateTime createdAt;
    @SerializedName("updated_at")
    private OffsetDateTime updatedAt;
    @SerializedName("submitted_at")
    private OffsetDateTime submittedAt;
    @SerializedName("filled_at")
    private OffsetDateTime filledAt;
    @SerializedName("expired_at")
    private OffsetDateTime expiredAt;
    @SerializedName("canceled_at")
    private OffsetDateTime canceledAt;
    @SerializedName("failed_at")
    private OffsetDateTime failedAt;
    @SerializedName("asset_id")
    private String assetId;
    private String symbol;
    @SerializedName("asset_class")
    private String assetClass;
    private String qty;
    @SerializedName("filled_qty")
    private String filledQty;
    @SerializedName("filled_avg_price")
    private String filledAvgPrice;
    @SerializedName("order_class")
    private String orderClass;
    @SerializedName("order_type")
    private String orderType;
    private String type;
    private String side;
    @SerializedName("position_intent")
    private String positionIntent;
    @SerializedName("time_in_force")
    private String timeInForce;
    @SerializedName("limit_price")
    private String limitPrice;
    @SerializedName("stop_price")
    private String stopPrice;
    private String status;
}
