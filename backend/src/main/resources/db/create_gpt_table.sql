CREATE TABLE gpt (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    date DATETIME NOT NULL,
    prompt LONGTEXT NOT NULL,
    reponse TEXT,
    aliasCompte VARCHAR(255),
    id_compte TEXT
);

