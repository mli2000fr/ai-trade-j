package com.app.backend.trade.portfolio.advanced;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "portfolio.multihead")
@Getter
@Setter
public class PortfolioMultiHeadConfig {
    private int hidden1 = 64;
    private int hidden2 = 32;
    private double learningRate = 1e-3;
    private double l2 = 1e-4;
    private double dropout = 0.1;
    private int epochs = 50;
    private int patience = 5;
    private double minDelta = 1e-4;
    private boolean normalize = true;
    private double selectThreshold = 0.5; // probabilité minimale inclusion
    private double positiveRetThreshold = 0.0; // futureRet > seuil => sélection
}

