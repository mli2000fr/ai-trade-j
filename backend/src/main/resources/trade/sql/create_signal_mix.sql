CREATE TABLE trade_ai.signal_mix (
    id VARCHAR(64) PRIMARY KEY AUTO_INCREMENT,
    symbol VARCHAR(16) NOT NULL,
    signal_mix VARCHAR(16),
    mix_created_at DATE
);