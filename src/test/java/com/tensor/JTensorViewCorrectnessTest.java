package com.tensor;

import org.junit.Test;

import java.util.Iterator;

import static org.junit.Assert.*;

public class JTensorViewCorrectnessTest {
    private static JTensor<Integer> matrix2x3() {
        return new JTensor<>(
                Integer.class,
                new int[]{2, 3},
                new Integer[]{1, 2, 3, 4, 5, 6});
    }

    private static <T> void assertLogicalValues(JTensor<T> tensor, T... expected) {
        Object[] actual = new Object[tensor.getSize()];
        Iterator<int[]> iterator = tensor.indicesIterator();
        int i = 0;
        while (iterator.hasNext()) {
            actual[i++] = tensor.getItem(iterator.next());
        }
        assertArrayEquals(expected, actual);
    }

    @Test
    public void reshapeContiguousTensorPreservesLogicalOrder() {
        JTensor<Integer> reshaped = matrix2x3().reshape(new int[]{3, 2});

        assertArrayEquals(new int[]{3, 2}, reshaped.getShape());
        assertLogicalValues(reshaped, 1, 2, 3, 4, 5, 6);
    }

    @Test
    public void transposeHasExpectedLogicalValues() {
        JTensor<Integer> transposed = matrix2x3().transpose();

        assertArrayEquals(new int[]{3, 2}, transposed.getShape());
        assertLogicalValues(transposed, 1, 4, 2, 5, 3, 6);
    }

    @Test
    public void reshapeTransposeMaterializesInLogicalOrder() {
        JTensor<Integer> reshaped = matrix2x3().transpose().reshape(new int[]{6});

        assertFalse(reshaped.isView());
        assertLogicalValues(reshaped, 1, 4, 2, 5, 3, 6);
    }

    @Test
    public void reshapeSliceMaterializesInLogicalOrder() {
        JTensor<Integer> tensor = new JTensor<>(
                Integer.class,
                new int[]{3, 4},
                new Integer[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

        JTensor<Integer> reshaped = tensor
                .slice(new int[][]{{0, 3}, {1, 3}})
                .reshape(new int[]{2, 3});

        assertLogicalValues(reshaped, 2, 3, 6, 7, 10, 11);
    }

    @Test
    public void copyConstructorCopiesContiguousTensor() {
        JTensor<Integer> source = matrix2x3();
        JTensor<Integer> copy = new JTensor<>(source);

        assertEquals(source, copy);
        assertFalse(copy.isView());
        assertArrayEquals(new int[]{3, 1}, copy.getStrides());
    }

    @Test
    public void copyConstructorNormalizesTransposeLayout() {
        JTensor<Integer> source = matrix2x3().transpose();
        JTensor<Integer> copy = new JTensor<>(source);

        assertEquals(source, copy);
        assertFalse(copy.isView());
        assertArrayEquals(new int[]{2, 1}, copy.getStrides());
        assertLogicalValues(copy, 1, 4, 2, 5, 3, 6);
    }

    @Test
    public void copyConstructorNormalizesSlicedLayout() {
        JTensor<Integer> original = matrix2x3();
        JTensor<Integer> source = original.slice(new int[][]{{0, 2}, {1, 3}});
        JTensor<Integer> copy = new JTensor<>(source);

        assertEquals(source, copy);
        assertFalse(copy.isView());
        assertArrayEquals(new int[]{2, 1}, copy.getStrides());
        assertLogicalValues(copy, 2, 3, 5, 6);

        original.setItem(new int[]{0, 1}, 99);
        copy.setItem(new int[]{1, 1}, 88);

        assertEquals(Integer.valueOf(2), copy.getItem(new int[]{0, 0}));
        assertEquals(Integer.valueOf(6), original.getItem(new int[]{1, 2}));
    }

    @Test
    public void copiedViewIsIndependentFromOriginalBackingData() {
        JTensor<Integer> original = matrix2x3();
        JTensor<Integer> copy = new JTensor<>(original.transpose());

        original.setItem(new int[]{0, 0}, 99);
        copy.setItem(new int[]{0, 1}, 88);

        assertEquals(Integer.valueOf(1), copy.getItem(new int[]{0, 0}));
        assertEquals(Integer.valueOf(4), original.getItem(new int[]{1, 0}));
    }

    @Test
    public void tooFewIndicesAreRejectedClearly() {
        InvalidArgumentException exception = assertThrows(
                InvalidArgumentException.class,
                () -> matrix2x3().getItem(new int[]{1}));

        assertTrue(exception.getMessage().contains("expected 2 but got 1"));
    }

    @Test
    public void tooManyIndicesAreRejectedClearly() {
        InvalidArgumentException exception = assertThrows(
                InvalidArgumentException.class,
                () -> matrix2x3().getItem(new int[]{1, 2, 0}));

        assertTrue(exception.getMessage().contains("expected 2 but got 3"));
    }

    @Test
    public void setItemAlsoRequiresExactIndexRank() {
        assertThrows(
                InvalidArgumentException.class,
                () -> matrix2x3().setItem(new int[]{1}, 10));
        assertThrows(
                InvalidArgumentException.class,
                () -> matrix2x3().setItem(new int[]{1, 2, 0}, 10));
    }

    @Test
    public void shapeGetterReturnsDefensiveCopy() {
        JTensor<Integer> tensor = matrix2x3();
        int[] shape = tensor.getShape();

        shape[0] = 1000;

        assertArrayEquals(new int[]{2, 3}, tensor.getShape());
        assertEquals(Integer.valueOf(6), tensor.getItem(new int[]{1, 2}));
    }

    @Test
    public void stridesGetterReturnsDefensiveCopy() {
        JTensor<Integer> tensor = matrix2x3();
        int[] strides = tensor.getStrides();

        strides[0] = 1000;

        assertArrayEquals(new int[]{3, 1}, tensor.getStrides());
        assertEquals(Integer.valueOf(6), tensor.getItem(new int[]{1, 2}));
    }

    @Test
    public void equalsAndHashCodeSupportNullElements() {
        JTensor<String> first = new JTensor<>(String.class, new int[]{2});
        JTensor<String> second = new JTensor<>(String.class, new int[]{2});

        assertEquals(first, second);
        assertEquals(first.hashCode(), second.hashCode());

        second.setItem(new int[]{1}, "value");
        assertNotEquals(first, second);
        assertNotEquals(first.hashCode(), second.hashCode());
    }

    @Test
    public void transposeSliceReshapePreservesLogicalOrder() {
        JTensor<Integer> result = matrix2x3()
                .transpose()
                .slice(new int[][]{{0, 3}, {1, 2}})
                .reshape(new int[]{3});

        assertLogicalValues(result, 4, 5, 6);
    }

    @Test
    public void sliceRequiresOneIntervalPerDimension() {
        assertThrows(
                InvalidArgumentException.class,
                () -> matrix2x3().slice(new int[][]{{0, 2}}));
        assertThrows(
                InvalidArgumentException.class,
                () -> matrix2x3().slice(new int[][]{{0, 2}, {0, 2}, {0, 1}}));
    }

    @Test
    public void primitiveBufferExportsViewInLogicalOrder() {
        assertArrayEquals(
                new int[]{1, 4, 2, 5, 3, 6},
                matrix2x3().transpose().toIntBuffer().array());
        assertArrayEquals(
                new int[]{2, 3, 5, 6},
                matrix2x3().slice(new int[][]{{0, 2}, {1, 3}}).toIntBuffer().array());
    }

    @Test
    public void byteSerializationRoundTripsViewAsContiguousTensor() {
        JTensor<Integer> source = matrix2x3()
                .transpose()
                .slice(new int[][]{{0, 3}, {1, 2}});

        JTensor<?> restored = JTensor.fromByteArray(source.toByteArray());

        assertEquals(source, restored);
        assertFalse(restored.isView());
        assertArrayEquals(new int[]{1, 1}, restored.getStrides());
        assertArrayEquals(new Integer[]{4, 5, 6}, restored.getData());
    }
}
