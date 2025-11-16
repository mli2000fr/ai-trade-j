package com.app.backend.trade.portfolio.learning;

import lombok.Getter;
import lombok.Setter;
import org.springframework.stereotype.Component;

/**
 * Configuration du modèle supervisé d'allocation de portefeuille.
 * Objectif: prédire un score risque-ajusté futur par symbole.
 */
@Getter
@Setter
@Component
public class PortfolioLearningConfig {
    private int lookaheadBars = 5;              // horizon futur pour le retour cible
    private int minHistoryBars = 60;            // historique minimum pour volatilité
    private int maxSymbolsTrain = 300;          // plafond symbole
    private int batchSize = 128;                // batch training (non utilisé pour l’instant)
    private int epochs = 30;                    // nombre d'époques max
    private double learningRate = 0.001;        // lr MLP
    private double l2 = 1e-4;                   // régularisation L2
    private int hidden1 = 64;                   // taille couche 1
    private int hidden2 = 32;                   // taille couche 2
    private double dropout = 0.2;               // dropout
    private double minScoreFilter = 0.0;        // filtre post-inférence
    private boolean normalizeFeatures = true;   // normaliser features (zscore simple)

    // Early stopping
    private int patienceEs = 5;                 // patience pour early stopping
    private double minDeltaEs = 1e-4;           // amélioration minimale requise sur val loss

    // Walk-forward
    private int walkForwardSplits = 3;          // nombre de splits séquentiels (train -> val)

    // Pénalités custom loss
    private double lambdaTurnover = 0.1;        // poids pénalité turnover
    private double lambdaDrawdown = 0.2;        // poids pénalité drawdown
}
