package com.app.backend.trade.model;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import jakarta.persistence.*;
import java.time.LocalDateTime;

/**
 * Entité JPA représentant un ordre de trading stocké en base.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
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

    // Constructeur personnalisé utilisé dans AlpacaService
    public OrderEntity(String idCompte, String idOrder, String side, String request, String reponse, String error, String statut, String idGpt) {
        this.idCompte = idCompte;
        this.idOrder = idOrder;
        this.side = side;
        this.request = request;
        this.reponse = reponse;
        this.error = error;
        this.statut = statut;
        this.idGpt = idGpt;
    }
}
