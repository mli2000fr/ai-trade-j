CREATE TABLE param_stra (
    symbol VARCHAR(20) PRIMARY KEY,
    breakout_lookbackp INT,
    macd_shortp INT,
    macd_longp INT,
    macd_signalp INT,
    meanr_smap INT,
    meanr_thresholdp DOUBLE,
    rsi_p INT,
    rsi_oversoldt DOUBLE,
    rsi_overbought DOUBLE,
    sma_shortp INT,
    sma_longp INT,
    trend_p INT
);