package com.tensor;

public class NumberHelper {
    private static final String TYPE_IS_NOT_SUPPORTED = "given type is not supported";

    private static void validateNumbers(Number... numbers) {
        for (Number number : numbers) {
            if (number == null) {
                throw new InvalidArgumentException("numeric values must not be null");
            }
        }
    }

    public static <T extends Number> T add(Class<T> type, T first, T second) {
        validateNumbers(first, second);
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(first.intValue() + second.intValue());
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(first.longValue() + second.longValue());
        } else if (Float.class.equals(type)) {
            return (T) Float.valueOf(first.floatValue() + second.floatValue());
        } else if (Double.class.equals(type)) {
            return (T) Double.valueOf(first.doubleValue() + second.doubleValue());
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf((byte) (first.byteValue() + second.byteValue()));
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf((short) (first.shortValue() + second.shortValue()));
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T pow(Class<T> type, T base, T exponent) {
        validateNumbers(base, exponent);
        if (Float.class.equals(type)) {
            return (T) Float.valueOf(
                    (float) Math.pow(base.floatValue(), exponent.floatValue()));
        } else if (Double.class.equals(type)) {
            return (T) Double.valueOf(
                    Math.pow(base.doubleValue(), exponent.doubleValue()));
        } else if (Integer.class.equals(type)
                || Long.class.equals(type)
                || Byte.class.equals(type)
                || Short.class.equals(type)) {
            long integralExponent = exponent.longValue();
            if (integralExponent < 0) {
                long integralBase = base.longValue();
                if (integralBase == 1) {
                    return one(type);
                } else if (integralBase == -1) {
                    return (integralExponent & 1) == 0
                            ? one(type)
                            : subtract(type, zero(type), one(type));
                }
                return cast(type, Math.pow(integralBase, integralExponent));
            }

            T result = one(type);
            T currentPower = base;
            while (integralExponent > 0) {
                if ((integralExponent & 1) != 0) {
                    result = multiply(type, result, currentPower);
                }
                integralExponent >>>= 1;
                if (integralExponent > 0) {
                    currentPower = multiply(type, currentPower, currentPower);
                }
            }
            return result;
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T subtract(Class<T> type, T first, T second) {
        validateNumbers(first, second);
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(first.intValue() - second.intValue());
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(first.longValue() - second.longValue());
        } else if (Float.class.equals(type)) {
            return (T) Float.valueOf(first.floatValue() - second.floatValue());
        } else if (Double.class.equals(type)) {
            return (T) Double.valueOf(first.doubleValue() - second.doubleValue());
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf((byte) (first.byteValue() - second.byteValue()));
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf((short) (first.shortValue() - second.shortValue()));
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T multiply(Class<T> type, T first, T second) {
        validateNumbers(first, second);
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(first.intValue() * second.intValue());
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(first.longValue() * second.longValue());
        } else if (Float.class.equals(type)) {
            return (T) Float.valueOf(first.floatValue() * second.floatValue());
        } else if (Double.class.equals(type)) {
            return (T) Double.valueOf(first.doubleValue() * second.doubleValue());
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf((byte) (first.byteValue() * second.byteValue()));
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf((short) (first.shortValue() * second.shortValue()));
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T divide(Class<T> type, T first, T second) {
        validateNumbers(first, second);
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(first.intValue() / second.intValue());
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(first.longValue() / second.longValue());
        } else if (Float.class.equals(type)) {
            return (T) Float.valueOf(first.floatValue() / second.floatValue());
        } else if (Double.class.equals(type)) {
            return (T) Double.valueOf(first.doubleValue() / second.doubleValue());
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf((byte) (first.byteValue() / second.byteValue()));
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf((short) (first.shortValue() / second.shortValue()));
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T mod(Class<T> type, T first, T second) {
        validateNumbers(first, second);
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(first.intValue() % second.intValue());
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(first.longValue() % second.longValue());
        } else if (Float.class.equals(type)) {
            return (T) Float.valueOf(first.floatValue() % second.floatValue());
        } else if (Double.class.equals(type)) {
            return (T) Double.valueOf(first.doubleValue() % second.doubleValue());
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf((byte) (first.byteValue() % second.byteValue()));
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf((short) (first.shortValue() % second.shortValue()));
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T and(Class<T> type, T first, T second) {
        validateNumbers(first, second);
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(first.intValue() & second.intValue());
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(first.longValue() & second.longValue());
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf((byte) (first.byteValue() & second.byteValue()));
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf((short) (first.shortValue() & second.shortValue()));
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T or(Class<T> type, T first, T second) {
        validateNumbers(first, second);
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(first.intValue() | second.intValue());
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(first.longValue() | second.longValue());
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf((byte) (first.byteValue() | second.byteValue()));
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf((short) (first.shortValue() | second.shortValue()));
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T xor(Class<T> type, T first, T second) {
        validateNumbers(first, second);
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(first.intValue() ^ second.intValue());
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(first.longValue() ^ second.longValue());
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf((byte) (first.byteValue() ^ second.byteValue()));
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf((short) (first.shortValue() ^ second.shortValue()));
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T not(Class<T> type, T value) {
        validateNumbers(value);
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(~value.intValue());
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(~value.longValue());
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf((byte) (~value.byteValue()));
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf((short) (~value.shortValue()));
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T leftShift(Class<T> type, T value, T shift) {
        validateNumbers(value, shift);
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(value.intValue() << shift.intValue());
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(value.longValue() << shift.longValue());
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf((byte) (value.byteValue() << shift.byteValue()));
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf((short) (value.shortValue() << shift.shortValue()));
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T rightShift(Class<T> type, T value, T shift) {
        validateNumbers(value, shift);
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(value.intValue() >> shift.intValue());
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(value.longValue() >> shift.longValue());
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf((byte) (value.byteValue() >> shift.byteValue()));
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf((short) (value.shortValue() >> shift.shortValue()));
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> boolean equals(Class<T> type, T first, T second) {
        validateNumbers(first, second);
        if (Integer.class.equals(type)) {
            return first.intValue() == second.intValue();
        } else if (Long.class.equals(type)) {
            return first.longValue() == second.longValue();
        } else if (Float.class.equals(type)) {
            return first.floatValue() == second.floatValue();
        } else if (Double.class.equals(type)) {
            return first.doubleValue() == second.doubleValue();
        } else if (Byte.class.equals(type)) {
            return first.byteValue() == second.byteValue();
        } else if (Short.class.equals(type)) {
            return first.shortValue() == second.shortValue();
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> boolean notEquals(Class<T> type, T first, T second) {
        validateNumbers(first, second);
        if (Integer.class.equals(type)) {
            return first.intValue() != second.intValue();
        } else if (Long.class.equals(type)) {
            return first.longValue() != second.longValue();
        } else if (Float.class.equals(type)) {
            return first.floatValue() != second.floatValue();
        } else if (Double.class.equals(type)) {
            return first.doubleValue() != second.doubleValue();
        } else if (Byte.class.equals(type)) {
            return first.byteValue() != second.byteValue();
        } else if (Short.class.equals(type)) {
            return first.shortValue() != second.shortValue();
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> boolean booleanValue(Class<T> type, T value) {
        validateNumbers(value);
        if (Integer.class.equals(type)) {
            return value.intValue() != 0;
        } else if (Long.class.equals(type)) {
            return value.longValue() != 0;
        } else if (Float.class.equals(type)) {
            return value.floatValue() != 0;
        } else if (Double.class.equals(type)) {
            return value.doubleValue() != 0;
        } else if (Byte.class.equals(type)) {
            return value.byteValue() != 0;
        } else if (Short.class.equals(type)) {
            return value.shortValue() != 0;
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }


    public static <T extends Number> boolean greaterThan(Class<T> type, T first, T second) {
        validateNumbers(first, second);
        if (Integer.class.equals(type)) {
            return first.intValue() > second.intValue();
        } else if (Long.class.equals(type)) {
            return first.longValue() > second.longValue();
        } else if (Float.class.equals(type)) {
            return first.floatValue() > second.floatValue();
        } else if (Double.class.equals(type)) {
            return first.doubleValue() > second.doubleValue();
        } else if (Byte.class.equals(type)) {
            return first.byteValue() > second.byteValue();
        } else if (Short.class.equals(type)) {
            return first.shortValue() > second.shortValue();
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> boolean greaterThanOrEquals(Class<T> type, T first, T second) {
        validateNumbers(first, second);
        if (Integer.class.equals(type)) {
            return first.intValue() >= second.intValue();
        } else if (Long.class.equals(type)) {
            return first.longValue() >= second.longValue();
        } else if (Float.class.equals(type)) {
            return first.floatValue() >= second.floatValue();
        } else if (Double.class.equals(type)) {
            return first.doubleValue() >= second.doubleValue();
        } else if (Byte.class.equals(type)) {
            return first.byteValue() >= second.byteValue();
        } else if (Short.class.equals(type)) {
            return first.shortValue() >= second.shortValue();
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> boolean lessThan(Class<T> type, T first, T second) {
        validateNumbers(first, second);
        if (Integer.class.equals(type)) {
            return first.intValue() < second.intValue();
        } else if (Long.class.equals(type)) {
            return first.longValue() < second.longValue();
        } else if (Float.class.equals(type)) {
            return first.floatValue() < second.floatValue();
        } else if (Double.class.equals(type)) {
            return first.doubleValue() < second.doubleValue();
        } else if (Byte.class.equals(type)) {
            return first.byteValue() < second.byteValue();
        } else if (Short.class.equals(type)) {
            return first.shortValue() < second.shortValue();
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> boolean lessThanOrEquals(Class<T> type, T first, T second) {
        validateNumbers(first, second);
        if (Integer.class.equals(type)) {
            return first.intValue() <= second.intValue();
        } else if (Long.class.equals(type)) {
            return first.longValue() <= second.longValue();
        } else if (Float.class.equals(type)) {
            return first.floatValue() <= second.floatValue();
        } else if (Double.class.equals(type)) {
            return first.doubleValue() <= second.doubleValue();
        } else if (Byte.class.equals(type)) {
            return first.byteValue() <= second.byteValue();
        } else if (Short.class.equals(type)) {
            return first.shortValue() <= second.shortValue();
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T zero(Class<T> type) {
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(0);
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(0);
        } else if (Float.class.equals(type)) {
            return (T) Float.valueOf(0F);
        } else if (Double.class.equals(type)) {
            return (T) Double.valueOf(0.0);
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf((byte) 0);
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf((short) 0);
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T one(Class<T> type) {
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(1);
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(1);
        } else if (Float.class.equals(type)) {
            return (T) Float.valueOf(1F);
        } else if (Double.class.equals(type)) {
            return (T) Double.valueOf(1.0);
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf((byte) 1);
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf((short) 1);
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T minValue(Class<T> type) {
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(Integer.MIN_VALUE);
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(Long.MIN_VALUE);
        } else if (Float.class.equals(type)) {
            return (T) Float.valueOf(Float.NEGATIVE_INFINITY);
        } else if (Double.class.equals(type)) {
            return (T) Double.valueOf(Double.NEGATIVE_INFINITY);
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf(Byte.MIN_VALUE);
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf(Short.MIN_VALUE);
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T maxValue(Class<T> type) {
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(Integer.MAX_VALUE);
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(Long.MAX_VALUE);
        } else if (Float.class.equals(type)) {
            return (T) Float.valueOf(Float.POSITIVE_INFINITY);
        } else if (Double.class.equals(type)) {
            return (T) Double.valueOf(Double.POSITIVE_INFINITY);
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf(Byte.MAX_VALUE);
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf(Short.MAX_VALUE);
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T max(Class<T> type, T number1, T number2) {
        validateNumbers(number1, number2);
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(Integer.max(number1.intValue(), number2.intValue()));
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(Long.max(number1.longValue(), number2.longValue()));
        } else if (Float.class.equals(type)) {
            return (T) Float.valueOf(Float.max(number1.floatValue(), number2.floatValue()));
        } else if (Double.class.equals(type)) {
            return (T) Double.valueOf(Double.max(number1.doubleValue(), number2.doubleValue()));
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf((byte) Integer.max(number1.intValue(), number2.intValue()));
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf((short) Integer.max(number1.intValue(), number2.intValue()));
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T min(Class<T> type, T number1, T number2) {
        validateNumbers(number1, number2);
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(Integer.min(number1.intValue(), number2.intValue()));
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(Long.min(number1.longValue(), number2.longValue()));
        } else if (Float.class.equals(type)) {
            return (T) Float.valueOf(Float.min(number1.floatValue(), number2.floatValue()));
        } else if (Double.class.equals(type)) {
            return (T) Double.valueOf(Double.min(number1.doubleValue(), number2.doubleValue()));
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf((byte) Integer.min(number1.intValue(), number2.intValue()));
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf((short) Integer.min(number1.intValue(), number2.intValue()));
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> T cast(Class<T> type, Number number) {
        validateNumbers(number);
        if (Integer.class.equals(type)) {
            return (T) Integer.valueOf(number.intValue());
        } else if (Long.class.equals(type)) {
            return (T) Long.valueOf(number.longValue());
        } else if (Float.class.equals(type)) {
            return (T) Float.valueOf(number.floatValue());
        } else if (Double.class.equals(type)) {
            return (T) Double.valueOf(number.doubleValue());
        } else if (Byte.class.equals(type)) {
            return (T) Byte.valueOf(number.byteValue());
        } else if (Short.class.equals(type)) {
            return (T) Short.valueOf(number.shortValue());
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }

    public static <T extends Number> int compare(Class<T> type, Number number1, Number number2) {
        validateNumbers(number1, number2);
        if (Integer.class.equals(type)) {
            return Integer.compare(number1.intValue(), number2.intValue());
        } else if (Long.class.equals(type)) {
            return Long.compare(number1.longValue(), number2.longValue());
        } else if (Float.class.equals(type)) {
            return Float.compare(number1.floatValue(), number2.floatValue());
        } else if (Double.class.equals(type)) {
            return Double.compare(number1.doubleValue(), number2.doubleValue());
        } else if (Byte.class.equals(type)) {
            return Byte.compare(number1.byteValue(), number2.byteValue());
        } else if (Short.class.equals(type)) {
            return Short.compare(number1.shortValue(), number2.shortValue());
        }
        throw new IllegalArgumentException(TYPE_IS_NOT_SUPPORTED);
    }
}
