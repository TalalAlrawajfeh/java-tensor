package com.tensor;

public class DataSizeMismatchException extends RuntimeException {
    public DataSizeMismatchException() {
    }

    public DataSizeMismatchException(String message) {
        super(message);
    }

    public DataSizeMismatchException(String message, Throwable cause) {
        super(message, cause);
    }

    public DataSizeMismatchException(Throwable cause) {
        super(cause);
    }

    public DataSizeMismatchException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
