Script SQL (PostgreSQL) pour créer la table
Schéma: trade_ai
Table: portfolio_allocation_models
BLOB: BYTEA
Index sur (tri, created_at)
Trigger pour updated_at
-- Schéma CREATE SCHEMA IF NOT EXISTS trade_ai;
-- Table CREATE TABLE IF NOT EXISTS trade_ai.portfolio_allocation_models ( id BIGSERIAL PRIMARY KEY, tri VARCHAR(64) NOT NULL, algo_version VARCHAR(64) NOT NULL, input_size INT NOT NULL, feature_means JSONB, feature_stds JSONB, model_zip BYTEA NOT NULL, learning_rate DOUBLE PRECISION, l2 DOUBLE PRECISION, hidden1 INT, hidden2 INT, epochs INT, notes TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW() );
-- Index pour chargement rapide du dernier modèle par tri CREATE INDEX IF NOT EXISTS idx_pam_tri_created_at ON trade_ai.portfolio_allocation_models (tri, created_at DESC);
-- Trigger updated_at CREATE OR REPLACE FUNCTION trade_ai.touch_updated_at() RETURNS TRIGGER AS <span>BEGIN NEW.updated_at = NOW(); RETURN NEW; END;</span> LANGUAGE plpgsql;
DROP TRIGGER IF EXISTS trg_pam_touch_updated_at ON trade_ai.portfolio_allocation_models;
CREATE TRIGGER trg_pam_touch_updated_at BEFORE UPDATE ON trade_ai.portfolio_allocation_models FOR EACH ROW EXECUTE FUNCTION trade_ai.touch_updated_at();


Notes d’intégration
Sauvegarde: PortfolioAllocationModelRepository.saveModel(tri, algoVersion, notes, model) est appelé par MultiSymbolPortfolioManager.trainAllocationModel(...).
Chargement: MultiSymbolPortfolioManager tente d’abord loadLatestModel(tri, …) depuis la BDD, sinon fallback sur fichier data/models/portfolio_allocation.zip, sinon entraînement auto.
Données persistées: architecture (poids réseau zip), input_size, feature_means/feature_stds (normalisation cohérente à l’inférence), hyperparamètres (lr, l2, hidden1/2, epochs), algo_version, notes, timestamps.