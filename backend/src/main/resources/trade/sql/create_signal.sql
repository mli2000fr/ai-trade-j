CREATE TABLE trade_ai.signal (
    id int PRIMARY KEY AUTO_INCREMENT,
    symbol VARCHAR(16) NOT NULL,
    signal_single VARCHAR(16),
    signal_mix VARCHAR(16),
    single_created_at DATE,
    mix_created_at DATE
);