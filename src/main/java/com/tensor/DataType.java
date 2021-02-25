package com.tensor;

public enum DataType {
    BOOLEAN((byte) 1, 1),
    BYTE((byte) 2, 8),
    SHORT((byte) 3, 16),
    INTEGER((byte) 4, 32),
    FLOAT((byte) 5, 32),
    LONG((byte) 6, 64),
    DOUBLE((byte) 7, 64);

    private final byte value;
    private final int size;

    DataType(byte value, int size) {
        this.value = value;
        this.size = size;
    }

    public static DataType fromValue(byte value) {
        if (BOOLEAN.value == value) {
            return BOOLEAN;
        } else if (BYTE.value == value) {
            return BYTE;
        } else if (SHORT.value == value) {
            return SHORT;
        } else if (INTEGER.value == value) {
            return INTEGER;
        } else if (FLOAT.value == value) {
            return FLOAT;
        } else if (LONG.value == value) {
            return LONG;
        } else if (DOUBLE.value == value) {
            return DOUBLE;
        }
        throw new IllegalArgumentException("invalid value");
    }

    public byte getValue() {
        return this.value;
    }

    public int getSize() {
        return this.size;
    }
}
