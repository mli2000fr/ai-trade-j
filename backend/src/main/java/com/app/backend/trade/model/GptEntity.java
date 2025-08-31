package com.app.backend.trade.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "gpt")
public class GptEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private LocalDateTime date;

    @Column(nullable = false, columnDefinition = "TEXT")
    private String prompt;

    @Column(columnDefinition = "LONGTEXT")
    private String reponse;

    @Column(columnDefinition = "TEXT")
    private String idCompte;

    // Constructeurs
    public GptEntity() {}
    public GptEntity(String idCompte, LocalDateTime date, String prompt, String reponse) {
        this.idCompte = idCompte;
        this.date = date;
        this.prompt = prompt;
        this.reponse = reponse;
    }

    // Getters et setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public LocalDateTime getDate() { return date; }
    public void setDate(LocalDateTime date) { this.date = date; }
    public String getPrompt() { return prompt; }
    public void setPrompt(String prompt) { this.prompt = prompt; }
    public String getReponse() { return reponse; }
    public void setReponse(String reponse) { this.reponse = reponse; }

    public String getIdCompte() {
        return idCompte;
    }

    public void setIdCompte(String idCompte) {
        this.idCompte = idCompte;
    }
}

