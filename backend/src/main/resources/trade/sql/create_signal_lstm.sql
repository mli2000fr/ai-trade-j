CREATE TABLE trade_ai.signal_lstm (
    id int PRIMARY KEY AUTO_INCREMENT,
    symbol VARCHAR(20),
    lstm_created_at DATE,
    signal_lstm VARCHAR(20),
    price_lstm DOUBLE,
    position_lstm VARCHAR(100),
    price_clo DOUBLE,
    result_tuning TEXT,
);
CREATE INDEX idx_signal_lstm_symbol ON trade_ai.signal_lstm(symbol);