package com.tensor;

import org.junit.Test;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Iterator;

import static org.junit.Assert.*;

public class JTensorConstructionSerializationEdgeCasesTest {
    private static <T> void assertLogicalValues(JTensor<T> tensor, T... expected) {
        Object[] actual = new Object[tensor.getSize()];
        Iterator<int[]> iterator = tensor.indicesIterator();
        int position = 0;
        while (iterator.hasNext()) {
            actual[position++] = tensor.getItem(iterator.next());
        }
        assertArrayEquals(expected, actual);
    }

    @Test
    public void inputShapeIsDefensivelyCopiedByEveryPublicConstructor() {
        int[] firstShape = {2, 2};
        int[] secondShape = {2, 2};
        int[] thirdShape = {2, 2};
        JTensor<Integer> defaulted = new JTensor<>(Integer.class, firstShape);
        JTensor<Integer> supplied = new JTensor<>(Integer.class, secondShape,
                new Integer[]{1, 2, 3, 4});
        JTensor<Integer> initialized = new JTensor<>(Integer.class, thirdShape,
                indices -> indices[0] + indices[1]);

        firstShape[0] = 99;
        secondShape[0] = 99;
        thirdShape[0] = 99;

        assertArrayEquals(new int[]{2, 2}, defaulted.getShape());
        assertArrayEquals(new int[]{2, 2}, supplied.getShape());
        assertArrayEquals(new int[]{2, 2}, initialized.getShape());
    }

    @Test
    public void constructorRejectsNullsAndMismatchedDataSizeClearly() {
        assertThrows(InvalidArgumentException.class,
                () -> new JTensor<Integer>(null, new int[]{1}));
        assertThrows(InvalidShapeException.class,
                () -> new JTensor<>(Integer.class, null));
        assertThrows(InvalidArgumentException.class,
                () -> new JTensor<>(Integer.class, new int[]{1}, (Integer[]) null));
        assertThrows(InvalidArgumentException.class,
                () -> new JTensor<>(Integer.class, new int[]{1}, (java.util.function.Function<int[], Integer>) null));
        assertThrows(DataSizeMismatchException.class,
                () -> new JTensor<>(Integer.class, new int[]{2, 2}, new Integer[]{1, 2, 3}));
    }

    @Test
    public void numericFactoriesFillAllElementsIncludingIdentityOffDiagonal() {
        assertLogicalValues(JTensor.zeros(Integer.class, new int[]{2, 2}), 0, 0, 0, 0);
        assertLogicalValues(JTensor.ones(Double.class, new int[]{2, 2}), 1.0, 1.0, 1.0, 1.0);
        assertLogicalValues(JTensor.identity(Integer.class, 3),
                1, 0, 0,
                0, 1, 0,
                0, 0, 1);
        assertLogicalValues(JTensor.singleValue("x"), "x");
    }

    @Test
    public void repeatRejectsNullWithLibraryValidationException() {
        assertThrows(InvalidArgumentException.class,
                () -> JTensor.repeat(new int[]{2}, null));
    }

    @Test
    public void nestedArrayFactoriesPreserveRankShapeAndRowMajorValues() {
        JTensor<Integer> two = JTensor.from2DArray(Integer.class,
                new Integer[][]{{1, 2, 3}, {4, 5, 6}});
        JTensor<Integer> three = JTensor.from3DArray(Integer.class,
                new Integer[][][]{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
        JTensor<Integer> four = JTensor.from4DArray(Integer.class,
                new Integer[][][][]{{{{1, 2}}, {{3, 4}}}, {{{5, 6}}, {{7, 8}}}});

        assertArrayEquals(new int[]{2, 3}, two.getShape());
        assertLogicalValues(two, 1, 2, 3, 4, 5, 6);
        assertArrayEquals(new int[]{2, 2, 2}, three.getShape());
        assertLogicalValues(three, 1, 2, 3, 4, 5, 6, 7, 8);
        assertArrayEquals(new int[]{2, 2, 1, 2}, four.getShape());
        assertLogicalValues(four, 1, 2, 3, 4, 5, 6, 7, 8);
    }

    @Test
    public void raggedNestedArraysAtEveryDepthAreRejected() {
        assertThrows(InvalidArgumentException.class,
                () -> JTensor.from2DArray(Integer.class,
                        new Integer[][]{{1, 2}, {3}}));
        assertThrows(InvalidArgumentException.class,
                () -> JTensor.from3DArray(Integer.class,
                        new Integer[][][]{{{1, 2}}, {{3}}}));
        assertThrows(InvalidArgumentException.class,
                () -> JTensor.from4DArray(Integer.class,
                        new Integer[][][][]{{{{1, 2}}}, {{{3}}}}));
    }

    @Test
    public void allSupportedSerializationTypesRoundTripExactly() {
        assertRoundTrip(new JTensor<>(Boolean.class, new int[]{3, 3},
                new Boolean[]{true, false, true, false, true, false, true, false, true}));
        assertRoundTrip(new JTensor<>(Byte.class, new int[]{2}, new Byte[]{Byte.MIN_VALUE, Byte.MAX_VALUE}));
        assertRoundTrip(new JTensor<>(Short.class, new int[]{2}, new Short[]{Short.MIN_VALUE, Short.MAX_VALUE}));
        assertRoundTrip(new JTensor<>(Integer.class, new int[]{2}, new Integer[]{Integer.MIN_VALUE, Integer.MAX_VALUE}));
        assertRoundTrip(new JTensor<>(Float.class, new int[]{3},
                new Float[]{Float.NEGATIVE_INFINITY, -0.0f, Float.NaN}));
        assertRoundTrip(new JTensor<>(Long.class, new int[]{2}, new Long[]{Long.MIN_VALUE, Long.MAX_VALUE}));
        assertRoundTrip(new JTensor<>(Double.class, new int[]{3},
                new Double[]{Double.POSITIVE_INFINITY, -0.0d, Double.NaN}));
    }

    private static void assertRoundTrip(JTensor<?> tensor) {
        JTensor<?> restored = JTensor.fromByteArray(tensor.toByteArray());
        assertEquals(tensor, restored);
        assertEquals(tensor.getType(), restored.getType());
        assertArrayEquals(tensor.getShape(), restored.getShape());
        assertFalse(restored.isView());
    }

    @Test
    public void typedBuffersUseLogicalViewOrderAndRejectWrongType() {
        JTensor<Float> floats = new JTensor<>(Float.class, new int[]{2, 2},
                new Float[]{1f, 2f, 3f, 4f}).transpose();
        JTensor<Long> longs = new JTensor<>(Long.class, new int[]{2, 2},
                new Long[]{1L, 2L, 3L, 4L}).transpose();
        JTensor<Double> doubles = new JTensor<>(Double.class, new int[]{2, 2},
                new Double[]{1d, 2d, 3d, 4d}).transpose();

        assertArrayEquals(new float[]{1f, 3f, 2f, 4f}, floats.toFloatBuffer().array(), 0f);
        assertArrayEquals(new long[]{1L, 3L, 2L, 4L}, longs.toLongBuffer().array());
        assertArrayEquals(new double[]{1d, 3d, 2d, 4d}, doubles.toDoubleBuffer().array(), 0d);
        assertThrows(InvalidTypeException.class, floats::toIntBuffer);
    }

    @Test
    public void deserializationRejectsUnknownTypeInvalidRankAndInvalidShape() {
        byte[] valid = new JTensor<>(Integer.class, new int[]{2},
                new Integer[]{1, 2}).toByteArray();

        byte[] unknownType = Arrays.copyOf(valid, valid.length);
        unknownType[0] = (byte) 127;
        assertThrows(InvalidArgumentException.class, () -> JTensor.fromByteArray(unknownType));

        byte[] negativeRank = Arrays.copyOf(valid, valid.length);
        ByteBuffer.wrap(negativeRank).putInt(1, -1);
        assertThrows(InvalidArgumentException.class, () -> JTensor.fromByteArray(negativeRank));

        byte[] zeroDimension = Arrays.copyOf(valid, valid.length);
        ByteBuffer.wrap(zeroDimension).putInt(5, 0);
        assertThrows(InvalidArgumentException.class, () -> JTensor.fromByteArray(zeroDimension));
    }

    @Test
    public void equalityRequiresSameTypeShapeAndLogicalValues() {
        JTensor<Integer> matrix = new JTensor<>(Integer.class, new int[]{2, 2},
                new Integer[]{1, 2, 3, 4});
        JTensor<Integer> sameValuesDifferentShape = new JTensor<>(Integer.class, new int[]{4},
                new Integer[]{1, 2, 3, 4});
        JTensor<Long> sameValuesDifferentType = new JTensor<>(Long.class, new int[]{2, 2},
                new Long[]{1L, 2L, 3L, 4L});
        JTensor<Integer> doubleTranspose = matrix.transpose().transpose();

        assertNotEquals(matrix, sameValuesDifferentShape);
        assertNotEquals(matrix, sameValuesDifferentType);
        assertEquals(matrix, doubleTranspose);
        assertEquals(matrix.hashCode(), doubleTranspose.hashCode());
        assertNotEquals(matrix, null);
        assertNotEquals(matrix, "not a tensor");
    }

    @Test
    public void backingDataAccessorRetainsDocumentedMutableArraySemantics() {
        JTensor<Integer> tensor = new JTensor<>(Integer.class, new int[]{2},
                new Integer[]{1, 2});

        tensor.getData()[1] = 99;

        assertEquals(Integer.valueOf(99), tensor.getItem(new int[]{1}));
    }
}
