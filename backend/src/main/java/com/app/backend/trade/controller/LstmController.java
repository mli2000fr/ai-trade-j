package com.app.backend.trade.controller;

import com.app.backend.trade.lstm.LstmTuningService;
import com.app.backend.trade.model.PreditLsdm;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.List;

/**
 * Contrôleur REST pour la gestion des opérations LSTM (entraînement et prédiction).
 * <p>
 * Permet d'entraîner un modèle LSTM et de prédire la prochaine clôture via des endpoints HTTP.
 * </p>
 */
@RestController
@RequestMapping("/api/lstm")
public class LstmController {

    @Autowired
    private LstmHelper lsdmHelper;

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
            @RequestParam String symbol) {
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
            @RequestParam String symbol) throws IOException {
        return lsdmHelper.getPredit(symbol);
    }

    @GetMapping("/tuneAllSymbols")
    public boolean tuneAllSymbols() {
        lsdmHelper.tuneAllSymbols();
        return true;
    }


    @GetMapping("/tuneAllSymbolsBis")
    public boolean tuneAllSymbolsBis() {
        lsdmHelper.tuneAllSymbolsBis();
        return true;
    }

    /**
     * Endpoint REST pour récupérer le reporting centralisé des erreurs de tuning LSTM
     */
    @GetMapping("/tuning-exceptions")
    @ResponseBody
    public List<LstmTuningService.TuningExceptionReportEntry> getTuningExceptionReport() {
        return lsdmHelper.getTuningExceptionReport();
    }
}
