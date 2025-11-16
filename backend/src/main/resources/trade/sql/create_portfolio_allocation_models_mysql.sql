-- MySQL DDL pour la persistance du modèle d'allocation de portefeuille
-- Cible: table trade_ai.portfolio_allocation_models

-- 1) Créer le schéma (base) si nécessaire
CREATE DATABASE IF NOT EXISTS trade_ai
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE trade_ai;

-- 2) Créer la table
CREATE TABLE IF NOT EXISTS portfolio_allocation_models (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  tri VARCHAR(64) NOT NULL,
  algo_version VARCHAR(64) NOT NULL,
  input_size INT NOT NULL,
  feature_means JSON NULL,
  feature_stds JSON NULL,
  model_zip LONGBLOB NOT NULL,
  learning_rate DOUBLE NULL,
  l2 DOUBLE NULL,
  hidden1 INT NULL,
  hidden2 INT NULL,
  epochs INT NULL,
  notes TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3) Index pour accélérer le chargement du dernier modèle par tri
CREATE INDEX idx_pam_tri_created_at
  ON portfolio_allocation_models (tri, created_at);

-- (Optionnel) Variante trigger si votre version MySQL ne supporte pas ON UPDATE CURRENT_TIMESTAMP
-- DELIMITER //
-- CREATE TRIGGER trg_pam_touch_updated_at
-- BEFORE UPDATE ON portfolio_allocation_models
-- FOR EACH ROW
-- BEGIN
--   SET NEW.updated_at = CURRENT_TIMESTAMP;
-- END//
-- DELIMITER ;

-- Notes:
-- - JSON nécessite MySQL 5.7+ (sinon remplacer par TEXT)
-- - LONGBLOB supporte des modèles DL4J volumineux
-- - Les requêtes Java référencent trade_ai.portfolio_allocation_models (schéma.nom_table)

