CREATE TABLE lstm_hyperparams (
    symbol VARCHAR(32) PRIMARY KEY,
    window_size INT,
    lstm_neurons INT,
    dropout_rate DOUBLE,
    learning_rate DOUBLE,
    num_epochs INT,
    patience INT,
    min_delta DOUBLE,
    k_folds INT,
    optimizer VARCHAR(16),
    l1 DOUBLE,
    l2 DOUBLE,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);