package com.app.backend.trade.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "trade_order")
public class OrderEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    @Column(columnDefinition = "TEXT", nullable = false)
    private String request;

    @Column(columnDefinition = "TEXT")
    private String reponse;

    @Column(columnDefinition = "TEXT")
    private String error;

    @Column(name = "created_at")
    private LocalDateTime createdAt = LocalDateTime.now();

    public OrderEntity() {}

    public OrderEntity(String request, String reponse, String error) {
        this.request = request;
        this.reponse = reponse;
        this.error = error;
        this.createdAt = LocalDateTime.now();
    }

    // Getters et setters
    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    public String getRequest() { return request; }
    public void setRequest(String request) { this.request = request; }
    public String getReponse() { return reponse; }
    public void setReponse(String reponse) { this.reponse = reponse; }
    public String getError() { return error; }
    public void setError(String error) { this.error = error; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}
