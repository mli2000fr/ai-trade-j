CREATE TABLE alpaca_asset (
    id VARCHAR(64) PRIMARY KEY,
    symbol VARCHAR(16) NOT NULL,
    exchange VARCHAR(16),
    status VARCHAR(16),
    name VARCHAR(128),
    created_at TIMESTAMP
);
