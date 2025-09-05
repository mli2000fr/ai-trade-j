-- Table pour stocker les meilleurs paramètres de stratégies pour chaque symbole
CREATE TABLE IF NOT EXISTS best_param (
    symbol VARCHAR(10) PRIMARY KEY,
    created_date DATE NOT NULL,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- Performance ranking JSON
    performance_ranking JSON,

    -- Trend Following params
    tf_trend_period INT,
    tf_performance DOUBLE,

    -- Improved Trend Following params
    itf_trend_period INT,
    itf_short_ma_period INT,
    itf_long_ma_period INT,
    itf_breakout_threshold DOUBLE,
    itf_use_rsi_filter BOOLEAN,
    itf_rsi_period INT,
    itf_performance DOUBLE,

    -- SMA Crossover params
    sma_short_period INT,
    sma_long_period INT,
    sma_performance DOUBLE,

    -- RSI params
    rsi_period INT,
    rsi_oversold DOUBLE,
    rsi_overbought DOUBLE,
    rsi_performance DOUBLE,

    -- Breakout params
    breakout_lookback_period INT,
    breakout_performance DOUBLE,

    -- MACD params
    macd_short_period INT,
    macd_long_period INT,
    macd_signal_period INT,
    macd_performance DOUBLE,

    -- Mean Reversion params
    mr_sma_period INT,
    mr_threshold DOUBLE,
    mr_performance DOUBLE,

    -- Best strategy info
    best_strategy_name VARCHAR(50),
    best_strategy_performance DOUBLE,

    -- Detailed results JSON (pour métriques complètes)
    detailed_results JSON,

    INDEX idx_symbol (symbol),
    INDEX idx_created_date (created_date),
    INDEX idx_best_strategy (best_strategy_name)
);

