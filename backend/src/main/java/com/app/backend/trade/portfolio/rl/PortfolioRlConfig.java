package com.app.backend.trade.portfolio.rl;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "portfolio.rl")
@Getter
@Setter
public class PortfolioRlConfig {
    private double learningRate = 1e-3;
    private double l2 = 1e-4;
    private int hidden1 = 128;
    private int hidden2 = 64;
    private double dropout = 0.1;
    private int epochs = 30;
    private int patience = 5;
    private double minDelta = 1e-4;
    private boolean normalize = true;
    private double turnoverPenalty = 0.001; // pénalité simple
    private double drawdownPenalty = 0.002; // pénalité simple
}

