CREATE TABLE best_in_out_strategy (
    symbol VARCHAR(20) PRIMARY KEY,
    entry_strategy_name VARCHAR(50),
    entry_strategy_params TEXT,
    exit_strategy_name VARCHAR(50),
    exit_strategy_params TEXT,
    rendement DOUBLE,
    trade_count INT,
    win_rate DOUBLE,
    max_drawdown DOUBLE,
    avg_pnl DOUBLE,
    profit_factor DOUBLE,
    avg_trade_bars DOUBLE,
    max_trade_gain DOUBLE,
    max_trade_loss DOUBLE,
    created_date DATE,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

        -- param de test
        initial_capital DOUBLE,
        risk_per_trade DOUBLE,
        stop_loss_pct DOUBLE,
        take_profit_pct DOUBLE
);