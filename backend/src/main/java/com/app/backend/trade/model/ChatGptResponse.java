package com.app.backend.trade.model;

public class ChatGptResponse {
    private Long idGpt;
    private String message;
    private String error;

    public ChatGptResponse() {}

    public ChatGptResponse(Long idGpt, String message, String error) {
        this.idGpt = idGpt;
        this.message = message;
        this.error = error;
    }

    public Long getIdGpt() {
        return idGpt;
    }

    public void setIdGpt(Long idGpt) {
        this.idGpt = idGpt;
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

