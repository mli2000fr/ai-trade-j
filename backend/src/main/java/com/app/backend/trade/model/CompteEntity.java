package com.app.backend.trade.model;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import jakarta.persistence.*;

/**
 * Entité JPA représentant un compte utilisateur pour l'accès aux API de trading.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "compte")
public class CompteEntity {
    /** Identifiant unique du compte (clé primaire). */
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    /** Nom du compte (obligatoire). */
    @Column(columnDefinition = "TEXT", nullable = false)
    private String nom;

    /** Alias du compte (optionnel). */
    @Column(columnDefinition = "TEXT")
    private String alias;

    /**
     * Clé API (sensible, ne pas exposer côté client).
     */
    @Column(columnDefinition = "TEXT")
    private String cle;

    /**
     * Secret API (sensible, ne pas exposer côté client).
     */
    @Column(columnDefinition = "TEXT")
    private String secret;

    /** Indique si le compte est réel (true) ou en mode démo (false). */
    @Column(name = "real")
    private Boolean real;
}
