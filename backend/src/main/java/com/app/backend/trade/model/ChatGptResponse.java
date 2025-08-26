package com.app.backend.trade.model;

public class ChatGptResponse {
    private String message;
    private String error;

    public ChatGptResponse() {}

    public ChatGptResponse(String message, String error) {
        this.message = message;
        this.error = error;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String getError() {
        return error;
    }

    public void setError(String error) {
        this.error = error;
    }
}

