package com.tensor;

@SuppressWarnings("this-escape") // initCause is required because the superclass has no cause constructor.
public class IndexOutOfBoundsException extends java.lang.IndexOutOfBoundsException {
    private static final long serialVersionUID = 1L;

    public IndexOutOfBoundsException() {
    }

    public IndexOutOfBoundsException(String message) {
        super(message);
    }

    public IndexOutOfBoundsException(String message, Throwable cause) {
        super(message);
        initCause(cause);
    }

    public IndexOutOfBoundsException(Throwable cause) {
        super(cause == null ? null : cause.toString());
        initCause(cause);
    }

    public IndexOutOfBoundsException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message);
        initCause(cause);
    }
}
