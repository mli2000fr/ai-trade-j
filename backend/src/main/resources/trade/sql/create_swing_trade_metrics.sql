CREATE TABLE trade_ai.swing_trade_metrics (
    symbol VARCHAR(20) PRIMARY KEY,
    volatilite DOUBLE,
    ratio_tendance DOUBLE,
    volume_moyen DOUBLE,
    score DOUBLE,
    top INT,
    date_calcul TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);