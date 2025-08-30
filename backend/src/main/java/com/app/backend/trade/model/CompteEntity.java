package com.app.backend.trade.model;

import jakarta.persistence.*;

/**
 * Entité JPA représentant un compte utilisateur pour l'accès aux API de trading.
 */
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

    public CompteEntity() {}

    public CompteEntity(String nom, String alias, String cle, String secret, Boolean real) {
        this.nom = nom;
        this.alias = alias;
        this.cle = cle;
        this.secret = secret;
        this.real = real;
    }

    /** @return l'identifiant du compte */
    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    /** @return le nom du compte */
    public String getNom() { return nom; }
    public void setNom(String nom) { this.nom = nom; }
    /** @return l'alias du compte */
    public String getAlias() { return alias; }
    public void setAlias(String alias) { this.alias = alias; }
    /** @return la clé API (sensible) */
    public String getCle() { return cle; }
    public void setCle(String cle) { this.cle = cle; }
    /** @return le secret API (sensible) */
    public String getSecret() { return secret; }
    public void setSecret(String secret) { this.secret = secret; }
    /** @return true si le compte est réel, false sinon */
    public Boolean getReal() { return real; }
    public void setReal(Boolean real) { this.real = real; }
}
