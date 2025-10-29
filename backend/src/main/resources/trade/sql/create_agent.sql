CREATE TABLE trade_ai.agent_ai (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    date DATETIME NOT NULL,
    prompt LONGTEXT NOT NULL,
    agent_name VARCHAR(30) NOT NULL,
    reponse LONGTEXT
);

