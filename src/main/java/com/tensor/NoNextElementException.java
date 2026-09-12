package com.tensor;

import java.util.NoSuchElementException;

@SuppressWarnings("this-escape") // initCause is required because the superclass has no cause constructor.
public class NoNextElementException extends NoSuchElementException {
    private static final long serialVersionUID = 1L;

    public NoNextElementException() {
    }

    public NoNextElementException(String message) {
        super(message);
    }

    public NoNextElementException(String message, Throwable cause) {
        super(message);
        initCause(cause);
    }

    public NoNextElementException(Throwable cause) {
        super(cause == null ? null : cause.toString());
        initCause(cause);
    }

    public NoNextElementException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message);
        initCause(cause);
    }
}
