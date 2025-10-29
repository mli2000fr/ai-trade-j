package com.app.backend.trade.model;

public enum Agent {
    GPT("gpt"),
    DEEPSEEK("deepseek");

    private final String name;

    Agent(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
