package com.app.backend.trade.model;

public class CompteDto {
    private Integer id;
    private String nom;
    private String alias;
    private Boolean real;

    public CompteDto() {}

    public CompteDto(Integer id, String nom, String alias, Boolean real) {
        this.id = id;
        this.nom = nom;
        this.alias = alias;
        this.real = real;
    }

    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }
    public String getNom() { return nom; }
    public void setNom(String nom) { this.nom = nom; }
    public String getAlias() { return alias; }
    public void setAlias(String alias) { this.alias = alias; }
    public Boolean getReal() { return real; }
    public void setReal(Boolean real) { this.real = real; }
}
