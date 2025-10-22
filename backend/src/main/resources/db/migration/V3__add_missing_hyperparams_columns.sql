-- Migration V3: ajout conditionnel des colonnes hyperparams potentiellement absentes
-- Objectif: éviter BadSqlGrammar sur REPLACE INTO lstm_hyperparams quand la table a été créée avant l'ajout
-- des colonnes kl_drift_threshold, mean_shift_sigma_threshold, use_multi_horizon_avg, entry_threshold_factor.

-- kl_drift_threshold
SET @col := (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
             WHERE TABLE_SCHEMA = DATABASE()
               AND TABLE_NAME='lstm_hyperparams'
               AND COLUMN_NAME='kl_drift_threshold');
SET @sql := IF(@col=0, 'ALTER TABLE lstm_hyperparams ADD COLUMN kl_drift_threshold DOUBLE DEFAULT 0.15 AFTER slippage_pct', 'SELECT 1');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- mean_shift_sigma_threshold
SET @col := (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
             WHERE TABLE_SCHEMA = DATABASE()
               AND TABLE_NAME='lstm_hyperparams'
               AND COLUMN_NAME='mean_shift_sigma_threshold');
SET @sql := IF(@col=0, 'ALTER TABLE lstm_hyperparams ADD COLUMN mean_shift_sigma_threshold DOUBLE DEFAULT 2.0 AFTER kl_drift_threshold', 'SELECT 1');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- use_multi_horizon_avg
SET @col := (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
             WHERE TABLE_SCHEMA = DATABASE()
               AND TABLE_NAME='lstm_hyperparams'
               AND COLUMN_NAME='use_multi_horizon_avg');
SET @sql := IF(@col=0, 'ALTER TABLE lstm_hyperparams ADD COLUMN use_multi_horizon_avg BOOLEAN DEFAULT FALSE AFTER slippage_pct', 'SELECT 1');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- entry_threshold_factor
SET @col := (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
             WHERE TABLE_SCHEMA = DATABASE()
               AND TABLE_NAME='lstm_hyperparams'
               AND COLUMN_NAME='entry_threshold_factor');
SET @sql := IF(@col=0, 'ALTER TABLE lstm_hyperparams ADD COLUMN entry_threshold_factor DOUBLE DEFAULT 1.2 AFTER use_multi_horizon_avg', 'SELECT 1');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

