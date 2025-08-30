package com.app.backend.trade.model;

import jakarta.persistence.*;

@Entity
@Table(name = "compte")
public class CompteEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    @Column(columnDefinition = "TEXT", nullable = false)
    private String nom;

    @Column(columnDefinition = "TEXT")
    private String alias;

    @Column(columnDefinition = "TEXT")
    private String cle;

    @Column(columnDefinition = "TEXT")
    private String secret;

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

    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    public String getNom() { return nom; }
    public void setNom(String nom) { this.nom = nom; }
    public String getAlias() { return alias; }
    public void setAlias(String alias) { this.alias = alias; }
    public String getCle() { return cle; }
    public void setCle(String cle) { this.cle = cle; }
    public String getSecret() { return secret; }
    public void setSecret(String secret) { this.secret = secret; }
    public Boolean getReal() { return real; }
    public void setReal(Boolean real) { this.real = real; }
}
