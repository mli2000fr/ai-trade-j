CREATE TABLE min_value (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(16) NOT NULL,
    date DATETIME NOT NULL,
    open VARCHAR(32),
    high VARCHAR(32),
    low VARCHAR(32),
    close VARCHAR(32),
    volume VARCHAR(32),
    number_of_trades VARCHAR(32),
    volume_weighted_average_price VARCHAR(32)
);
CREATE INDEX idx_symbol_date ON trade_ai.daily_value (symbol, date);