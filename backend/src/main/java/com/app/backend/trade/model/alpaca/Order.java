package com.app.backend.trade.model.alpaca;

import com.google.gson.annotations.SerializedName;

import java.time.OffsetDateTime;
import java.util.List;

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

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getClientOrderId() {
        return clientOrderId;
    }

    public void setClientOrderId(String clientOrderId) {
        this.clientOrderId = clientOrderId;
    }

    public OffsetDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(OffsetDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public OffsetDateTime getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(OffsetDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }

    public OffsetDateTime getSubmittedAt() {
        return submittedAt;
    }

    public void setSubmittedAt(OffsetDateTime submittedAt) {
        this.submittedAt = submittedAt;
    }

    public OffsetDateTime getFilledAt() {
        return filledAt;
    }

    public void setFilledAt(OffsetDateTime filledAt) {
        this.filledAt = filledAt;
    }

    public OffsetDateTime getExpiredAt() {
        return expiredAt;
    }

    public void setExpiredAt(OffsetDateTime expiredAt) {
        this.expiredAt = expiredAt;
    }

    public OffsetDateTime getCanceledAt() {
        return canceledAt;
    }

    public void setCanceledAt(OffsetDateTime canceledAt) {
        this.canceledAt = canceledAt;
    }

    public OffsetDateTime getFailedAt() {
        return failedAt;
    }

    public void setFailedAt(OffsetDateTime failedAt) {
        this.failedAt = failedAt;
    }

    public String getAssetId() {
        return assetId;
    }

    public void setAssetId(String assetId) {
        this.assetId = assetId;
    }

    public String getSymbol() {
        return symbol;
    }

    public void setSymbol(String symbol) {
        this.symbol = symbol;
    }

    public String getAssetClass() {
        return assetClass;
    }

    public void setAssetClass(String assetClass) {
        this.assetClass = assetClass;
    }

    public String getQty() {
        return qty;
    }

    public void setQty(String qty) {
        this.qty = qty;
    }

    public String getFilledQty() {
        return filledQty;
    }

    public void setFilledQty(String filledQty) {
        this.filledQty = filledQty;
    }

    public String getFilledAvgPrice() {
        return filledAvgPrice;
    }

    public void setFilledAvgPrice(String filledAvgPrice) {
        this.filledAvgPrice = filledAvgPrice;
    }

    public String getOrderClass() {
        return orderClass;
    }

    public void setOrderClass(String orderClass) {
        this.orderClass = orderClass;
    }

    public String getOrderType() {
        return orderType;
    }

    public void setOrderType(String orderType) {
        this.orderType = orderType;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getSide() {
        return side;
    }

    public void setSide(String side) {
        this.side = side;
    }

    public String getPositionIntent() {
        return positionIntent;
    }

    public void setPositionIntent(String positionIntent) {
        this.positionIntent = positionIntent;
    }

    public String getTimeInForce() {
        return timeInForce;
    }

    public void setTimeInForce(String timeInForce) {
        this.timeInForce = timeInForce;
    }

    public String getLimitPrice() {
        return limitPrice;
    }

    public void setLimitPrice(String limitPrice) {
        this.limitPrice = limitPrice;
    }

    public String getStopPrice() {
        return stopPrice;
    }

    public void setStopPrice(String stopPrice) {
        this.stopPrice = stopPrice;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public boolean isExtendedHours() {
        return extendedHours;
    }

    public void setExtendedHours(boolean extendedHours) {
        this.extendedHours = extendedHours;
    }

    public List<Leg> getLegs() {
        return legs;
    }

    public void setLegs(List<Leg> legs) {
        this.legs = legs;
    }

    public OffsetDateTime getExpiresAt() {
        return expiresAt;
    }

    public void setExpiresAt(OffsetDateTime expiresAt) {
        this.expiresAt = expiresAt;
    }

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
