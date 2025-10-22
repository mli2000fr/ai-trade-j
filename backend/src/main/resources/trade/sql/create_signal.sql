CREATE TABLE trade_ai.signal (
    symbol VARCHAR(20),
    created_date DATE,
    signal_single VARCHAR(20),
    signal_mix VARCHAR(20),
    signal_lstm VARCHAR(20),
    price_lstm DOUBLE,
    position_lstm VARCHAR(100),
    price_clo DOUBLE,
    PRIMARY KEY (symbol, created_date)
);

CREATE INDEX idx_symbol_created_date ON trade_ai.signal(symbol, created_date);