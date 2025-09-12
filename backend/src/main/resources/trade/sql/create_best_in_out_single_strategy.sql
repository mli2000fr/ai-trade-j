CREATE TABLE trade_ai.best_in_out_single_strategy (
    symbol VARCHAR(20) PRIMARY KEY,
    entry_strategy_name VARCHAR(50),
    entry_strategy_params TEXT,
    exit_strategy_name VARCHAR(50),
    exit_strategy_params TEXT,
    rendement DOUBLE,
    rendement_check DOUBLE,
    trade_count INT,
    win_rate DOUBLE,
    max_drawdown DOUBLE,
    avg_pnl DOUBLE,
    profit_factor DOUBLE,
    avg_trade_bars DOUBLE,
    max_trade_gain DOUBLE,
    max_trade_loss DOUBLE,
    score_swing_trade DOUBLE,
    fltred_out BOOLEAN DEFAULT FALSE,
    check_result TEXT,
    created_date DATE,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        -- param de test
        initial_capital DOUBLE,
        risk_per_trade DOUBLE,
        stop_loss_pct DOUBLE,
        take_profit_pct DOUBLE,
    nb_simples INT
);

Voici l’explication de chaque colonne de la table best_in_out_strategy :
symbol : le symbole de l’actif (ex : « AAPL » pour Apple). C’est la clé primaire, chaque ligne correspond à un actif unique.
entry_strategy_name : nom de la stratégie utilisée pour l’entrée (achat), par exemple « RSI », « SMA Crossover », etc.
entry_strategy_params : paramètres de la stratégie d’entrée, stockés en format JSON (ex : période, seuil, etc.).
exit_strategy_name : nom de la stratégie utilisée pour la sortie (vente).
exit_strategy_params : paramètres de la stratégie de sortie, également en JSON.
rendement : performance globale de la stratégie (gain ou perte total, généralement en pourcentage ou ratio).
trade_count : nombre total de trades réalisés par la stratégie sur la période testée.
win_rate : taux de réussite des trades (proportion de trades gagnants).
max_drawdown : perte maximale enregistrée pendant la période (mesure du risque).
avg_pnl : profit moyen par trade (PnL = Profit and Loss).
profit_factor : ratio entre les gains et les pertes (mesure de la qualité de la stratégie).
avg_trade_bars : durée moyenne d’un trade (en nombre de bougies ou périodes).
max_trade_gain : plus grand gain réalisé sur un trade.
max_trade_loss : plus grande perte réalisée sur un trade.
created_date : date de création de l’enregistrement.
updated_date : date de dernière modification de l’enregistrement.

Paramètres de gestion du risque :
initial_capital : capital initial utilisé pour le backtest (ex : 10 000 $).
risk_per_trade : part du capital risquée sur chaque trade (ex : 0.02 pour 2 %).
stop_loss_pct : pourcentage de stop loss appliqué (ex : 0.04 pour 4 %).
take_profit_pct : pourcentage de take profit appliqué (ex : 0.08 pour 8 %).

rendement: favorise les symboles rentables
winRate: favorise la régularité des gains
profitFactor: favorise la qualité des trades
maxDrawdown: pénalise les symboles trop risqués
avgPnL: favorise les trades avec bon gain moyen