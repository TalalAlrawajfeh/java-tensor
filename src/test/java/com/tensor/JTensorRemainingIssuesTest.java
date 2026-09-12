package com.tensor;

import org.junit.Test;

import java.util.Arrays;
import java.util.Iterator;

import static org.junit.Assert.*;

public class JTensorRemainingIssuesTest {
    private static <T> void assertLogicalValues(JTensor<T> tensor, T... expected) {
        Object[] actual = new Object[tensor.getSize()];
        Iterator<int[]> iterator = tensor.indicesIterator();
        int position = 0;
        while (iterator.hasNext()) {
            actual[position++] = tensor.getItem(iterator.next());
        }
        assertArrayEquals(expected, actual);
    }

    private static JTensor<Integer> cube() {
        return new JTensor<>(Integer.class, new int[]{2, 3, 4},
                index -> 100 * index[0] + 10 * index[1] + index[2]);
    }

    @Test
    public void reduceAllOnTransposedViewUsesLogicalTrailingDimensions() {
        JTensor<Integer> transposed = cube().transpose();

        JTensor<Integer> squeezed = transposed.reduceAll(0, Integer::sum, 1, false);
        JTensor<Integer> kept = transposed.reduceAll(0, Integer::sum, 1, true);

        assertArrayEquals(new int[]{4}, squeezed.getShape());
        assertLogicalValues(squeezed, 360, 366, 372, 378);
        assertArrayEquals(new int[]{4, 1, 1}, kept.getShape());
        assertLogicalValues(kept, 360, 366, 372, 378);
    }

    @Test
    public void reduceAllOnDimensionSwappedAndSlicedViewUsesLogicalOrder() {
        JTensor<Integer> view = cube()
                .swapDimensions(0, 2)
                .slice(new int[][]{{1, 4}, {0, 3}, {0, 2}});

        JTensor<Integer> result = view.reduceAll(0, Integer::sum, 1, false);

        assertLogicalValues(result, 366, 372, 378);
    }

    @Test
    public void emptyTensorCanBeRavelledFlattenedAndBuiltFromEmptyArray() {
        JTensor<Integer> empty = JTensor.empty(Integer.class);
        JTensor<Integer> ravelled = empty.ravel();
        JTensor<Integer> flattened = empty.flatten();
        JTensor<Integer> fromArray = JTensor.from1DArray(Integer.class, new Integer[0]);

        for (JTensor<Integer> tensor : new JTensor[]{ravelled, flattened, fromArray}) {
            assertEquals(0, tensor.getSize());
            assertArrayEquals(new int[]{}, tensor.getShape());
            assertFalse(tensor.indicesIterator().hasNext());
        }
        assertNotSame(empty.getData(), flattened.getData());
    }

    @Test
    public void resizeToEmptyProducesIndependentValidEmptyTensor() {
        JTensor<Integer> source = new JTensor<>(Integer.class, new int[]{3},
                new Integer[]{1, 2, 3});

        JTensor<Integer> resized = source.resize(new int[]{});

        assertEquals(0, resized.getSize());
        assertArrayEquals(new int[]{}, resized.getShape());
        assertFalse(resized.isView());
        assertNotSame(source.getData(), resized.getData());
    }

    @Test
    public void squeezingOnlyDimensionReportsUnsupportedScalarClearly() {
        InvalidArgumentException exception = assertThrows(
                InvalidArgumentException.class,
                () -> JTensor.singleValue(1).squeeze(0));

        assertTrue(exception.getMessage().contains("scalar tensors are not supported"));
    }

    @Test
    public void explicitTypeRepeatSupportsNullAndWiderDeclaredTypes() {
        JTensor<String> nulls = JTensor.repeat(String.class, new int[]{2}, null);
        JTensor<Number> numbers = JTensor.repeat(Number.class, new int[]{2}, Integer.valueOf(1));

        assertLogicalValues(nulls, null, null);
        numbers.setItem(new int[]{1}, Double.valueOf(2.5));
        assertLogicalValues(numbers, Integer.valueOf(1), Double.valueOf(2.5));
    }

    @Test
    public void inferredRepeatRejectsIncompatibleRuntimeAssignmentClearly() {
        Number value = Integer.valueOf(1);
        JTensor<Number> inferred = JTensor.repeat(new int[]{1}, value);

        assertThrows(InvalidTypeException.class,
                () -> inferred.setItem(new int[]{0}, Double.valueOf(2.5)));
    }

    @Test
    public void constructorAndInitializerRejectValuesOutsideDeclaredType() {
        assertThrows(InvalidTypeException.class,
                () -> new JTensor(Integer.class, new int[]{2},
                        new Number[]{Integer.valueOf(1), Double.valueOf(2)}));
        assertThrows(InvalidTypeException.class,
                () -> new JTensor(Integer.class, new int[]{1},
                        index -> Double.valueOf(2)));
    }

    @Test
    public void primitiveClassTokensAreRejectedBeforeReflectiveAllocation() {
        InvalidTypeException exception = assertThrows(
                InvalidTypeException.class,
                () -> new JTensor<>(int.class, new int[]{1}));

        assertTrue(exception.getMessage().contains("wrapper class"));
        assertThrows(InvalidTypeException.class,
                () -> JTensor.repeat(int.class, new int[]{1}, 1));
    }

    @Test
    public void constructorNormalizesCovariantArrayAndDoesNotAliasInput() {
        Integer[] input = {1, 2};
        JTensor<Number> tensor = new JTensor<>(Number.class, new int[]{2}, input);

        input[0] = 99;
        tensor.setItem(new int[]{1}, Double.valueOf(2.5));

        assertLogicalValues(tensor, Integer.valueOf(1), Double.valueOf(2.5));
        assertEquals(Number.class, tensor.getData().getClass().getComponentType());
    }

    @Test
    public void initializerCannotRedirectWritesByMutatingItsIndexArray() {
        JTensor<Integer> tensor = new JTensor<>(Integer.class, new int[]{2, 2}, index -> {
            int value = 10 * index[0] + index[1];
            Arrays.fill(index, 100);
            return value;
        });

        assertLogicalValues(tensor, 0, 1, 10, 11);
    }

    @Test
    public void genericReductionSupportsNullIdentityWhenTypeIsKnown() {
        JTensor<String> words = new JTensor<>(String.class, new int[]{2, 2},
                new String[]{"a", "b", "c", "d"});

        JTensor<String> reduced = words.reduceAlong(
                null,
                (left, right) -> left == null ? right : left + right,
                1,
                false);

        assertLogicalValues(reduced, "ab", "cd");
    }

    @Test
    public void oneDimensionalReductionsReturnScalarLikeSingleton() {
        JTensor<Integer> integers = new JTensor<>(Integer.class, new int[]{3},
                new Integer[]{2, 3, 4});
        JTensor<Double> doubles = new JTensor<>(Double.class, new int[]{3},
                new Double[]{1.0, 2.0, 6.0});

        JTensor<Integer> sum = JTensor.sum(integers, 0, false);
        JTensor<Integer> product = JTensor.product(integers, 0, false);
        JTensor<Double> mean = JTensor.mean(doubles, 0, false);
        JTensor<Integer> reduceAll = integers.reduceAll(0, Integer::sum, 0, false);

        assertArrayEquals(new int[]{1}, sum.getShape());
        assertLogicalValues(sum, 9);
        assertLogicalValues(product, 24);
        assertLogicalValues(mean, 3.0);
        assertLogicalValues(reduceAll, 9);
    }

    @Test
    public void serializationRejectsNullPayloadWithLibraryException() {
        InvalidArgumentException exception = assertThrows(
                InvalidArgumentException.class,
                () -> new JTensor<>(Integer.class, new int[]{1}).toByteArray());

        assertTrue(exception.getMessage().contains("null values"));
    }

    @Test
    public void numericOperationsAndBuffersRejectNullValuesClearly() {
        JTensor<Integer> nullInteger = new JTensor<>(Integer.class, new int[]{1});
        JTensor<Integer> one = JTensor.singleValue(1);

        assertThrows(InvalidArgumentException.class,
                () -> JTensor.add(nullInteger, one));
        assertThrows(InvalidArgumentException.class,
                () -> JTensor.sum(nullInteger, 0, true));
        assertThrows(InvalidArgumentException.class, nullInteger::toIntBuffer);
        assertThrows(InvalidArgumentException.class,
                () -> JTensor.cast(nullInteger, Double.class));
        assertThrows(InvalidArgumentException.class,
                () -> JTensor.pow(nullInteger, one));
        assertThrows(InvalidArgumentException.class,
                () -> JTensor.sqrt(nullInteger));
    }

    @Test
    public void masksAndBooleanOperationsRejectNullValuesClearly() {
        JTensor<Boolean> nullBoolean = new JTensor<>(Boolean.class, new int[]{1});
        JTensor<Boolean> truth = JTensor.singleValue(true);

        assertThrows(InvalidArgumentException.class,
                () -> JTensor.singleValue(1).applyMask(nullBoolean));
        assertThrows(InvalidArgumentException.class,
                () -> JTensor.booleanAnd(nullBoolean, truth));
        assertThrows(InvalidArgumentException.class,
                () -> JTensor.booleanOr(truth, nullBoolean));
        assertThrows(InvalidArgumentException.class,
                () -> JTensor.booleanNot(nullBoolean));
        assertThrows(InvalidArgumentException.class,
                () -> JTensor.castFromBoolean(Integer.class, nullBoolean));
    }

    @Test
    public void arithmeticWrappersRejectNullTensorsBeforeDereferencingThem() {
        JTensor<Integer> one = JTensor.singleValue(1);

        assertThrows(InvalidArgumentException.class, () -> JTensor.add(null, one));
        assertThrows(InvalidArgumentException.class, () -> JTensor.pow(one, null));
        assertThrows(InvalidArgumentException.class, () -> JTensor.sqrt(null));
        assertThrows(InvalidArgumentException.class, () -> JTensor.sum(null, 0, true));
        assertThrows(InvalidArgumentException.class, () -> JTensor.argMax(null, 0, true));
    }

    @Test
    public void emptyRankZeroTensorIsNotMistakenForBroadcastScalar() {
        JTensor<Integer> empty = JTensor.empty(Integer.class);

        assertThrows(InvalidArgumentException.class,
                () -> JTensor.add(empty, JTensor.singleValue(1)));
        Pair<JTensor<Integer>, JTensor<Integer>> pair = JTensor.broadcast(empty, empty);
        assertSame(empty, pair.getFirst());
        assertSame(empty, pair.getSecond());
    }

    @Test
    public void argumentReductionsReturnFirstNaNLikeMinAndMaxPropagation() {
        JTensor<Double> values = new JTensor<>(Double.class, new int[]{4},
                new Double[]{1.0, Double.NaN, 5.0, Double.NaN});

        assertTrue(Double.isNaN(JTensor.max(values, 0, true).getItem(new int[]{0})));
        assertTrue(Double.isNaN(JTensor.min(values, 0, true).getItem(new int[]{0})));
        assertEquals(Integer.valueOf(1), JTensor.argMax(values, 0, true).getItem(new int[]{0}));
        assertEquals(Integer.valueOf(1), JTensor.argMin(values, 0, true).getItem(new int[]{0}));
    }

    @Test
    public void argumentReductionsDistinguishNegativeAndPositiveZero() {
        JTensor<Double> values = new JTensor<>(Double.class, new int[]{2},
                new Double[]{-0.0, 0.0});

        assertEquals(Integer.valueOf(1), JTensor.argMax(values, 0, true).getItem(new int[]{0}));
        assertEquals(Integer.valueOf(0), JTensor.argMin(values, 0, true).getItem(new int[]{0}));
    }

    @Test
    public void argumentReductionSentinelNeverWinsAgainstRealExtremeValue() {
        JTensor<Integer> minimums = new JTensor<>(Integer.class, new int[]{2},
                new Integer[]{Integer.MIN_VALUE, Integer.MIN_VALUE});
        JTensor<Integer> maximums = new JTensor<>(Integer.class, new int[]{2},
                new Integer[]{Integer.MAX_VALUE, Integer.MAX_VALUE});
        JTensor<Double> infinities = new JTensor<>(Double.class, new int[]{2},
                new Double[]{Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY});

        assertEquals(Integer.valueOf(0), JTensor.argMax(minimums, 0, true).getItem(new int[]{0}));
        assertEquals(Integer.valueOf(0), JTensor.argMin(maximums, 0, true).getItem(new int[]{0}));
        assertEquals(Integer.valueOf(0), JTensor.argMax(infinities, 0, true).getItem(new int[]{0}));
    }

    @Test
    public void packageBoundsExceptionIsAlsoAStandardBoundsException() {
        Throwable exception = assertThrows(
                java.lang.IndexOutOfBoundsException.class,
                () -> JTensor.singleValue(1).getItem(new int[]{1}));

        assertTrue(exception instanceof com.tensor.IndexOutOfBoundsException);
    }

    @Test
    public void nullCallbacksAndCoreOperandsAreRejectedClearly() {
        JTensor<Integer> tensor = JTensor.singleValue(1);

        assertThrows(InvalidArgumentException.class,
                () -> JTensor.applyFunction(Integer.class, tensor, null));
        assertThrows(InvalidArgumentException.class,
                () -> JTensor.applyBinaryOperation(Integer.class, tensor, tensor, null));
        assertThrows(InvalidArgumentException.class, () -> tensor.filter(null));
        assertThrows(InvalidArgumentException.class, () -> tensor.replace(null, value -> value));
        assertThrows(InvalidArgumentException.class,
                () -> tensor.reduceAlong(0, null, 0, true));
        assertThrows(InvalidArgumentException.class, () -> JTensor.broadcast(null, tensor));
        assertThrows(InvalidArgumentException.class, () -> JTensor.concatenate(null, tensor, 0));
    }
}
