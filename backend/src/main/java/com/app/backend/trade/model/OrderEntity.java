package com.app.backend.trade.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

/**
 * Entité JPA représentant un ordre de trading stocké en base.
 */
@Entity
@Table(name = "trade_order")
public class OrderEntity {
    /** Identifiant unique de l'ordre (clé primaire). */
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    /** Requête envoyée à l'API (JSON). */
    @Column(columnDefinition = "TEXT", nullable = false)
    private String request;

    /** Réponse reçue de l'API (JSON). */
    @Column(columnDefinition = "TEXT")
    private String reponse;

    /** Message d'erreur éventuel. */
    @Column(columnDefinition = "TEXT")
    private String error;

    /** Date de création de l'ordre. */
    @Column(name = "created_at")
    private LocalDateTime createdAt = LocalDateTime.now();

    @Column(columnDefinition = "TEXT")
    private String idCompte;

    @Column(columnDefinition = "TEXT")
    private String idOrder;


    @Column(columnDefinition = "TEXT")
    private String side;

    @Column(columnDefinition = "TEXT")
    private String statut;

    @Column(columnDefinition = "TEXT")
    private String idGpt;

    public OrderEntity() {}

    public OrderEntity(String idCompte, String idOrder, String side, String request, String reponse, String error, String statut, String idGpt) {
        this.idCompte = idCompte;
        this.idOrder = idOrder;
        this.side = side;
        this.request = request;
        this.reponse = reponse;
        this.error = error;
        this.createdAt = LocalDateTime.now();
        this.statut = statut;
        this.idGpt = idGpt;
    }

    public String getIdOrder() {
        return idOrder;
    }

    public void setIdOrder(String idOrder) {
        this.idOrder = idOrder;
    }

    public String getSide() {
        return side;
    }

    public void setSide(String side) {
        this.side = side;
    }

    public String getIdGpt() {
        return idGpt;
    }

    public void setIdGpt(String idGpt) {
        this.idGpt = idGpt;
    }

    /** @return l'identifiant de l'ordre */
    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    /** @return la requête envoyée */
    public String getRequest() { return request; }
    public void setRequest(String request) { this.request = request; }
    /** @return la réponse reçue */
    public String getReponse() { return reponse; }
    public void setReponse(String reponse) { this.reponse = reponse; }
    /** @return le message d'erreur */
    public String getError() { return error; }
    public void setError(String error) { this.error = error; }
    /** @return la date de création */
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public String getStatut() {
        return statut;
    }

    public void setStatut(String statut) {
        this.statut = statut;
    }

    public String getIdCompte() {
        return idCompte;
    }

    public void setIdCompte(String idCompte) {
        this.idCompte = idCompte;
    }
}
