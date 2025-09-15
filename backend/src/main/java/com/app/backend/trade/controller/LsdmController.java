package com.app.backend.trade.controller;

import com.app.backend.trade.model.PreditLsdm;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * Contrôleur REST pour la gestion des opérations LSTM (entraînement et prédiction).
 * <p>
 * Permet d'entraîner un modèle LSTM et de prédire la prochaine clôture via des endpoints HTTP.
 * </p>
 */
@RestController
@RequestMapping("/api/lsdm")
public class LsdmController {

    @Autowired
    private LsdmHelper lsdmHelper;

    /**
     * Lance l'entraînement du modèle LSTM pour un symbole donné.
     * @param symbol le symbole à entraîner (ex : "BTCUSDT")
     * @param windowSize la taille de la fenêtre pour le LSTM
     * @param numEpochs le nombre d'epochs pour l'entraînement
     * @param learningRate le taux d'apprentissage
     * @param optimizer le nom de l'optimiseur (ex : "adam")
     * @return message de confirmation
     */
    @GetMapping("/train")
    public String trainLstm(
            @RequestParam String symbol,
            @RequestParam int windowSize,
            @RequestParam int numEpochs,
            @RequestParam double learningRate,
            @RequestParam String optimizer) {
        lsdmHelper.trainLstm(symbol);
        return "Entraînement LSTM terminé pour " + symbol;
    }

    /**
     * Prédit la prochaine valeur de clôture pour un symbole donné.
     * http://localhost:8080/api/lsdm/predict?symbol=CELUW&windowSize=20&learningRate=0.001&optimizer=adam
     * @param symbol le symbole à prédire (ex : "BTCUSDT")
     * @return la valeur de clôture prédite
     */
    @GetMapping("/predict")
    public PreditLsdm predictNextClose(
            @RequestParam String symbol) {
        return lsdmHelper.getPredit(symbol);
    }
}
