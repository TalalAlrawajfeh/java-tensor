package com.tensor;

public class InvalidShapeException extends RuntimeException {
    public InvalidShapeException() {
    }

    public InvalidShapeException(String message) {
        super(message);
    }

    public InvalidShapeException(String message, Throwable cause) {
        super(message, cause);
    }

    public InvalidShapeException(Throwable cause) {
        super(cause);
    }

    public InvalidShapeException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
