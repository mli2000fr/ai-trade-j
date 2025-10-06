CREATE TABLE trade_ai.lstm_models (
    symbol VARCHAR(32) PRIMARY KEY,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    hyperparams_json TEXT,
    scalers_json TEXT,
    model_blob LONGBLOB,
    mse DOUBLE, profit_factor DOUBLE, win_rate DOUBLE, max_drawdown DOUBLE, rmse DOUBLE, sum_profit DOUBLE, total_trades INT, business_score DOUBLE,
    total_series_tested INT,
    rendement DOUBLE,
    phase INT,
    ratio DOUBLE,
    eligible BOOLEAN DEFAULT TRUE,
    methode VARCHAR(100) DEFAULT 'unknown',
    normalization_scope VARCHAR(16) DEFAULT 'window'
);