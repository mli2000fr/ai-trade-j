CREATE TABLE trade_ai.signal_single (
    id VARCHAR(64) PRIMARY KEY AUTO_INCREMENT,
    symbol VARCHAR(16) NOT NULL,
    signal_single VARCHAR(16),
    single_created_at DATE
);