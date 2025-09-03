CREATE TABLE `trade_order` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,,
     side TEXT,
    `request` TEXT NOT NULL,
    `reponse` TEXT,
    `error` TEXT,
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    aliasCompte VARCHAR(255),
    idCompte TEXT,
    statut TEXT,
    id_gpt TEXT
);
