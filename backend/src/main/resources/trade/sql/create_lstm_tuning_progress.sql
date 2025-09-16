CREATE TABLE trade_ai.lstm_tuning_progress (
  symbol VARCHAR(32) PRIMARY KEY,
  total_configs INT,
  tested_configs INT,
  status VARCHAR(16),
  start_time DATETIME,
  end_time DATETIME,
  last_update DATETIME
);