CREATE TABLE trade_ai.lstm_models (
    symbol VARCHAR(32) PRIMARY KEY,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    hyperparams_json TEXT,
    model_blob LONGBLOB,
    normalization_scope VARCHAR(16) DEFAULT 'window'
);