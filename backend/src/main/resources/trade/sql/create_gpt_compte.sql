CREATE TABLE compte (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nom TEXT NOT NULL,
    alias TEXT,
    cle TEXT,
    secret TEXT,
    real BOOLEAN
);