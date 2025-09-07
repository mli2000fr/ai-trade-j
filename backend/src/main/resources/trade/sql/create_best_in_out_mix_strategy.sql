CREATE TABLE best_in_out_mix_strategy (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    symbol VARCHAR(32) NOT NULL,
    in_strategy_names TEXT NOT NULL,
    out_strategy_names TEXT NOT NULL,
    score DOUBLE NOT NULL,
    in_params TEXT,
    out_params TEXT,
    backtest_result TEXT,
    create_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

            -- param de test
            initial_capital DOUBLE,
            risk_per_trade DOUBLE,
            stop_loss_pct DOUBLE,
            take_profit_pct DOUBLE
);