package com.app.backend.trade.exception;

public class ModelPersistenceException extends RuntimeException {
    public ModelPersistenceException(String message, Throwable cause) {
        super(message, cause);
    }
    public ModelPersistenceException(String message) {
        super(message);
    }
}

