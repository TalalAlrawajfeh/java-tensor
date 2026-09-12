package com.tensor;

import org.junit.Test;

import java.util.Iterator;

import static org.junit.Assert.*;

public class JTensorOperationsEdgeCasesTest {
    private static JTensor<Integer> matrix() {
        return new JTensor<>(Integer.class, new int[]{2, 3},
                new Integer[]{1, 2, 3, 4, 5, 6});
    }

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
    public void broadcastingRowVectorAcrossMatrixProducesExpectedValues() {
        JTensor<Integer> row = new JTensor<>(Integer.class, new int[]{3},
                new Integer[]{10, 20, 30});

        JTensor<Integer> result = JTensor.add(matrix(), row);

        assertArrayEquals(new int[]{2, 3}, result.getShape());
        assertLogicalValues(result, 11, 22, 33, 14, 25, 36);
    }

    @Test
    public void broadcastingTwoSingletonAxesProducesOuterResult() {
        JTensor<Integer> column = new JTensor<>(Integer.class, new int[]{2, 1},
                new Integer[]{10, 20});
        JTensor<Integer> row = new JTensor<>(Integer.class, new int[]{1, 3},
                new Integer[]{1, 2, 3});

        JTensor<Integer> result = JTensor.add(column, row);

        assertArrayEquals(new int[]{2, 3}, result.getShape());
        assertLogicalValues(result, 11, 12, 13, 21, 22, 23);
    }

    @Test
    public void broadcastingWorksWhenAnOperandIsANonContiguousView() {
        JTensor<Integer> transposed = matrix().transpose();
        JTensor<Integer> offsets = new JTensor<>(Integer.class, new int[]{2},
                new Integer[]{10, 20});

        JTensor<Integer> result = JTensor.add(transposed, offsets);

        assertLogicalValues(result, 11, 24, 12, 25, 13, 26);
    }

    @Test
    public void incompatibleBroadcastShapesAreRejected() {
        JTensor<Integer> first = new JTensor<>(Integer.class, new int[]{2, 3});
        JTensor<Integer> second = new JTensor<>(Integer.class, new int[]{2, 2});

        assertThrows(InvalidArgumentException.class, () -> JTensor.add(first, second));
    }

    @Test
    public void concatenateUsesLogicalValuesFromViewsOnBothAxes() {
        JTensor<Integer> first = matrix().transpose();
        JTensor<Integer> second = new JTensor<>(Integer.class, new int[]{2, 3},
                new Integer[]{7, 8, 9, 10, 11, 12}).transpose();

        JTensor<Integer> rows = JTensor.concatenate(first, second, 0);
        JTensor<Integer> columns = JTensor.concatenate(first, second, 1);

        assertArrayEquals(new int[]{6, 2}, rows.getShape());
        assertLogicalValues(rows, 1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12);
        assertArrayEquals(new int[]{3, 4}, columns.getShape());
        assertLogicalValues(columns, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12);
    }

    @Test
    public void arithmeticOperationsHandleEveryElement() {
        JTensor<Integer> left = new JTensor<>(Integer.class, new int[]{4},
                new Integer[]{8, 9, 10, 11});
        JTensor<Integer> right = new JTensor<>(Integer.class, new int[]{4},
                new Integer[]{2, 3, 4, 5});

        assertLogicalValues(JTensor.subtract(left, right), 6, 6, 6, 6);
        assertLogicalValues(JTensor.multiply(left, right), 16, 27, 40, 55);
        assertLogicalValues(JTensor.divide(left, right), 4, 3, 2, 2);
        assertLogicalValues(JTensor.mod(left, right), 0, 0, 2, 1);
        assertLogicalValues(JTensor.pow(
                new JTensor<>(Integer.class, new int[]{3}, new Integer[]{2, 3, 4}),
                new JTensor<>(Integer.class, new int[]{3}, new Integer[]{3, 2, 1})),
                8, 9, 4);
        assertLogicalValues(JTensor.sqrt(
                new JTensor<>(Integer.class, new int[]{3}, new Integer[]{1, 4, 9})),
                1, 2, 3);
    }

    @Test
    public void bitwiseOperationsPreserveNumericRuntimeType() {
        JTensor<Byte> left = new JTensor<>(Byte.class, new int[]{2},
                new Byte[]{(byte) 6, (byte) 8});
        JTensor<Byte> right = new JTensor<>(Byte.class, new int[]{2},
                new Byte[]{(byte) 3, (byte) 1});

        assertLogicalValues(JTensor.and(left, right), (byte) 2, (byte) 0);
        assertLogicalValues(JTensor.or(left, right), (byte) 7, (byte) 9);
        assertLogicalValues(JTensor.xor(left, right), (byte) 5, (byte) 9);
        assertLogicalValues(JTensor.leftShift(left, right), (byte) 48, (byte) 16);
        assertLogicalValues(JTensor.rightShift(left, right), (byte) 0, (byte) 4);
        assertLogicalValues(JTensor.not(left), (byte) -7, (byte) -9);
    }

    @Test
    public void mapFilterReplaceAndContainsAreNullSafeAndNonMutating() {
        JTensor<String> source = new JTensor<>(String.class, new int[]{4},
                new String[]{null, "a", "bb", "ccc"});

        JTensor<Integer> lengths = source.map(Integer.class, value -> value == null ? -1 : value.length());
        JTensor<String> filtered = source.filter(value -> value != null && value.length() >= 2);
        JTensor<String> replaced = source.replace(value -> value == null, value -> "missing");

        assertLogicalValues(lengths, -1, 1, 2, 3);
        assertLogicalValues(filtered, "bb", "ccc");
        assertLogicalValues(replaced, "missing", "a", "bb", "ccc");
        assertTrue(source.contains(null));
        assertFalse(source.contains("missing"));
        assertNull(source.getItem(new int[]{0}));
    }

    @Test
    public void partialRankMaskSelectsWholeTrailingBlocks() {
        JTensor<Boolean> mask = new JTensor<>(Boolean.class, new int[]{2},
                new Boolean[]{false, true});

        JTensor<Integer> selected = matrix().applyMask(mask);

        assertArrayEquals(new int[]{1, 3}, selected.getShape());
        assertLogicalValues(selected, 4, 5, 6);
    }

    @Test
    public void allFalseMaskAndFilterReturnEmptyTensor() {
        JTensor<Boolean> mask = JTensor.repeat(new int[]{2, 3}, false);

        JTensor<Integer> selected = matrix().applyMask(mask);
        JTensor<Integer> filtered = matrix().filter(value -> false);

        assertEquals(0, selected.getSize());
        assertEquals(0, filtered.getSize());
        assertFalse(selected.indicesIterator().hasNext());
        assertFalse(filtered.indicesIterator().hasNext());
    }

    @Test
    public void resizeOfViewTakesLogicalPrefix() {
        JTensor<Integer> resized = matrix().transpose().resize(new int[]{2, 2});

        assertArrayEquals(new int[]{2, 2}, resized.getShape());
        assertLogicalValues(resized, 1, 4, 2, 5);
    }

    @Test
    public void reductionsWorkAcrossBothAxesAndViews() {
        JTensor<Integer> source = matrix();

        assertLogicalValues(JTensor.sum(source, 0, false), 5, 7, 9);
        assertLogicalValues(JTensor.sum(source, 1, true), 6, 15);
        assertLogicalValues(JTensor.product(source, 0, false), 4, 10, 18);
        assertLogicalValues(JTensor.min(source.transpose(), 1, false), 1, 2, 3);
        assertLogicalValues(JTensor.max(source.transpose(), 1, false), 4, 5, 6);
    }

    @Test
    public void reduceAllCollapsesTheRequestedAxisAndEverythingAfterIt() {
        JTensor<Integer> source = new JTensor<>(Integer.class, new int[]{2, 2, 2},
                new Integer[]{1, 2, 3, 4, 5, 6, 7, 8});

        JTensor<Integer> kept = source.reduceAll(0, Integer::sum, 1, true);
        JTensor<Integer> squeezed = source.reduceAll(0, Integer::sum, 1, false);

        assertArrayEquals(new int[]{2, 1, 1}, kept.getShape());
        assertLogicalValues(kept, 10, 26);
        assertArrayEquals(new int[]{2}, squeezed.getShape());
        assertLogicalValues(squeezed, 10, 26);
    }

    @Test
    public void comparisonsBooleanLogicAndCastsHaveExpectedValues() {
        JTensor<Integer> left = new JTensor<>(Integer.class, new int[]{3},
                new Integer[]{-1, 0, 2});
        JTensor<Integer> right = new JTensor<>(Integer.class, new int[]{3},
                new Integer[]{0, 0, 1});

        assertLogicalValues(JTensor.greaterThan(left, right), false, false, true);
        assertLogicalValues(JTensor.lessThanOrEquals(left, right), true, true, false);
        assertLogicalValues(JTensor.compare(left, right), -1, 0, 1);
        assertLogicalValues(JTensor.cast(left, Double.class), -1.0, 0.0, 2.0);
        JTensor<Boolean> booleans = JTensor.castToBoolean(left);
        assertLogicalValues(booleans, true, false, true);
        assertLogicalValues(JTensor.castFromBoolean(Integer.class, booleans), 1, 0, 1);
        assertLogicalValues(JTensor.booleanNot(booleans), false, true, false);
        assertLogicalValues(JTensor.booleanAnd(
                booleans, new JTensor<>(Boolean.class, new int[]{3},
                        new Boolean[]{true, true, false})), true, false, false);
    }

    @Test
    public void argReductionsUseCorrectIndicesIncludingTiesAndNegativeFloats() {
        JTensor<Double> source = new JTensor<>(Double.class, new int[]{2, 4},
                new Double[]{-5.0, -2.0, -2.0, -7.0, 4.0, 1.0, 4.0, 3.0});

        assertLogicalValues(JTensor.argMax(source, 1, false), 1, 0);
        assertLogicalValues(JTensor.argMin(source, 1, true), 3, 1);
    }

    @Test
    public void oneDimensionalReductionWithoutKeptDimensionReturnsScalarLikeSingleton() {
        JTensor<Integer> vector = new JTensor<>(Integer.class, new int[]{3},
                new Integer[]{1, 2, 3});

        JTensor<Integer> sum = JTensor.sum(vector, 0, false);
        JTensor<Integer> reduceAll = vector.reduceAll(0, Integer::sum, 0, false);

        assertArrayEquals(new int[]{1}, sum.getShape());
        assertLogicalValues(sum, 6);
        assertArrayEquals(new int[]{1}, reduceAll.getShape());
        assertLogicalValues(reduceAll, 6);
    }
}
