package com.app.backend.trade.model.alpaca;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

import com.google.gson.annotations.SerializedName;

import java.time.OffsetDateTime;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Order {

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
    @SerializedName("extended_hours")
    private boolean extendedHours;
    private List<Leg> legs;
    @SerializedName("expires_at")
    private OffsetDateTime expiresAt;

    @Override
    public String toString() {
        return "Order{" +
                "id='" + id + '\'' +
                ", clientOrderId='" + clientOrderId + '\'' +
                ", createdAt=" + createdAt +
                ", updatedAt=" + updatedAt +
                ", submittedAt=" + submittedAt +
                ", filledAt=" + filledAt +
                ", expiredAt=" + expiredAt +
                ", canceledAt=" + canceledAt +
                ", failedAt=" + failedAt +
                ", assetId='" + assetId + '\'' +
                ", symbol='" + symbol + '\'' +
                ", assetClass='" + assetClass + '\'' +
                ", qty='" + qty + '\'' +
                ", filledQty='" + filledQty + '\'' +
                ", filledAvgPrice='" + filledAvgPrice + '\'' +
                ", orderClass='" + orderClass + '\'' +
                ", orderType='" + orderType + '\'' +
                ", type='" + type + '\'' +
                ", side='" + side + '\'' +
                ", positionIntent='" + positionIntent + '\'' +
                ", timeInForce='" + timeInForce + '\'' +
                ", limitPrice='" + limitPrice + '\'' +
                ", stopPrice='" + stopPrice + '\'' +
                ", status='" + status + '\'' +
                ", extendedHours=" + extendedHours +
                ", legs=" + legs +
                ", expiresAt=" + expiresAt +
                '}';
    }
}
