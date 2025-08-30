CREATE TABLE `trade_order` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `request` TEXT NOT NULL,
    `reponse` TEXT,
    `error` TEXT,
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    aliasCompte VARCHAR(255),
);
