package com.app.backend.trade.model;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import jakarta.persistence.*;
import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
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

    // Constructeur personnalisé utilisé dans ChatGptService
    public GptEntity(String idCompte, LocalDateTime date, String prompt, String reponse) {
        this.idCompte = idCompte;
        this.date = date;
        this.prompt = prompt;
        this.reponse = reponse;
    }
}
