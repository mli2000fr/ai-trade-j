package com.app.backend.trade.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/lsdm")
public class LsdmController {

    @Autowired
    private LsdmHelper lsdmHelper;

    @GetMapping("/train")
    public String trainLstm(@RequestParam String symbol, @RequestParam int windowSize, @RequestParam int numEpochs) {
        lsdmHelper.trainLstm(symbol, windowSize, numEpochs);
        return "Entraînement LSTM terminé pour " + symbol;
    }

    @GetMapping("/predict")
    public double predictNextClose(@RequestParam String symbol, @RequestParam int windowSize) {
        return lsdmHelper.predictNextClose(symbol, windowSize);
    }
}
