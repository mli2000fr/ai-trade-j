-- Migration V2: ajout des colonnes manquantes pour aligner avec l'INSERT de LstmHyperparamsRepository
-- Ajoute use_multi_horizon_avg et entry_threshold_factor si elles n'existent pas déjà.

-- Ajout colonne use_multi_horizon_avg si absente
SET @col := (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
             WHERE TABLE_SCHEMA = DATABASE()
               AND TABLE_NAME='lstm_tuning_metrics'
               AND COLUMN_NAME='use_multi_horizon_avg');
SET @sql := IF(@col=0, 'ALTER TABLE lstm_tuning_metrics ADD COLUMN use_multi_horizon_avg TINYINT(1) NOT NULL DEFAULT 0 AFTER avg_bars_in_position', 'SELECT 1');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- Ajout colonne entry_threshold_factor si absente
SET @col := (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
             WHERE TABLE_SCHEMA = DATABASE()
               AND TABLE_NAME='lstm_tuning_metrics'
               AND COLUMN_NAME='entry_threshold_factor');
SET @sql := IF(@col=0, 'ALTER TABLE lstm_tuning_metrics ADD COLUMN entry_threshold_factor DOUBLE NULL AFTER use_multi_horizon_avg', 'SELECT 1');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

