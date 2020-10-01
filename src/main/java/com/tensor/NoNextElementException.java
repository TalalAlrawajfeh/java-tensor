package com.tensor;

public class NoNextElementException extends RuntimeException {
    public NoNextElementException() {
    }

    public NoNextElementException(String message) {
        super(message);
    }

    public NoNextElementException(String message, Throwable cause) {
        super(message, cause);
    }

    public NoNextElementException(Throwable cause) {
        super(cause);
    }

    public NoNextElementException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
