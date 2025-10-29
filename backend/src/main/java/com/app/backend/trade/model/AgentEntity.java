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
@Table(name = "agent_ai")
public class AgentEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private LocalDateTime date;

    @Column(nullable = false)
    private String agentName;

    @Column(nullable = false, columnDefinition = "TEXT")
    private String prompt;

    @Column(columnDefinition = "LONGTEXT")
    private String reponse;

    // Constructeur personnalisé utilisé dans ChatGptService
    public AgentEntity(String agentName,LocalDateTime date, String prompt, String reponse) {
        this.agentName = agentName;
        this.date = date;
        this.prompt = prompt;
        this.reponse = reponse;
    }
}
