-- Migration: Ajout de la colonne business_score pour stocker le score métier agrégé
-- Hypothèse: Base MySQL 8.x (support ADD COLUMN IF NOT EXISTS). Adapter si version plus ancienne.

-- Ajout colonne si absente
ALTER TABLE lstm_tuning_metrics
    ADD COLUMN IF NOT EXISTS business_score DOUBLE NULL AFTER num_trades;

-- Ajout index pour requêtes de filtrage/tri éventuelles
ALTER TABLE lstm_tuning_metrics
    ADD INDEX IF NOT EXISTS idx_lstm_tuning_metrics_business_score (business_score);

-- Option: pour anciennes versions MySQL (<8.0) remplacer par un bloc dynamique :
-- SET @col := (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME='lstm_tuning_metrics' AND COLUMN_NAME='business_score');
-- SET @sql := IF(@col=0, 'ALTER TABLE lstm_tuning_metrics ADD COLUMN business_score DOUBLE NULL AFTER num_trades', 'SELECT 1');
-- PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

