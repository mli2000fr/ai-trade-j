-- Script MySQL pour la table des modèles de policy RL
-- Création base + sélection
CREATE DATABASE IF NOT EXISTS trade_ai;
USE trade_ai;

-- Table pour les modèles de policy RL (MySQL)
CREATE TABLE IF NOT EXISTS portfolio_rl_policy_models (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    tri VARCHAR(32) NOT NULL,
    algo_version VARCHAR(64) NOT NULL,
    input_size INT NOT NULL,
    feature_means TEXT NULL,
    feature_stds TEXT NULL,
    model_zip LONGBLOB NOT NULL,
    lr DOUBLE NOT NULL,
    l2 DOUBLE NOT NULL,
    hidden1 INT NOT NULL,
    hidden2 INT NOT NULL,
    dropout DOUBLE NOT NULL,
    notes TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_rl_policy_tri_created_at (tri, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Optionnel: unicité version par tri
-- CREATE UNIQUE INDEX ux_rl_policy_tri_version ON portfolio_rl_policy_models (tri, algo_version);
