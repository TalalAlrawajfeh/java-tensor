package com.tensor;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

import static org.junit.Assert.*;

public class JTensorViewEdgeCasesTest {
    private static JTensor<Integer> cube() {
        return new JTensor<>(Integer.class, new int[]{2, 3, 4},
                index -> 100 * index[0] + 10 * index[1] + index[2]);
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
    public void threeDimensionalSwapPreservesEveryLogicalCoordinate() {
        JTensor<Integer> swapped = cube().swapDimensions(0, 2);

        assertArrayEquals(new int[]{4, 3, 2}, swapped.getShape());
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 2; k++) {
                    assertEquals(Integer.valueOf(100 * k + 10 * j + i),
                            swapped.getItem(new int[]{i, j, k}));
                }
            }
        }
    }

    @Test
    public void chainedSwapSliceReverseReshapeUsesLogicalOrder() {
        JTensor<Integer> result = cube()
                .swapDimensions(0, 2)
                .slice(new int[][]{{1, 4}, {0, 2}, {0, 2}})
                .reverse(1)
                .reshape(new int[]{3, 4});

        assertFalse(result.isView());
        assertLogicalValues(result,
                11, 111, 1, 101,
                12, 112, 2, 102,
                13, 113, 3, 103);
    }

    @Test
    public void modifyingSliceWritesThroughToOriginalStorage() {
        JTensor<Integer> original = cube();
        JTensor<Integer> slice = original.slice(new int[][]{{1, 2}, {1, 3}, {1, 4}});

        slice.setItem(new int[]{0, 1, 2}, 999);

        assertEquals(Integer.valueOf(999), original.getItem(new int[]{1, 2, 3}));
    }

    @Test
    public void modifyingReversedViewWritesThroughToMappedElement() {
        JTensor<Integer> original = cube();
        JTensor<Integer> reversed = original.reverse(2);

        reversed.setItem(new int[]{0, 1, 0}, 777);

        assertEquals(Integer.valueOf(777), original.getItem(new int[]{0, 1, 3}));
    }

    @Test
    public void ravelOfContiguousTensorSharesStorage() {
        JTensor<Integer> original = cube();
        JTensor<Integer> ravelled = original.ravel();

        assertTrue(ravelled.isView());
        ravelled.setItem(new int[]{5}, 555);

        assertEquals(Integer.valueOf(555), original.getItem(new int[]{0, 1, 1}));
    }

    @Test
    public void ravelOfViewMaterializesAndDoesNotAliasOriginal() {
        JTensor<Integer> original = cube();
        JTensor<Integer> ravelled = original.swapDimensions(0, 2).ravel();

        assertFalse(ravelled.isView());
        ravelled.setItem(new int[]{0}, 999);

        assertEquals(Integer.valueOf(0), original.getItem(new int[]{0, 0, 0}));
    }

    @Test
    public void flattenAlwaysCopiesAndKeepsViewLogicalOrder() {
        JTensor<Integer> original = cube();
        JTensor<Integer> flattened = original.swapDimensions(0, 2).flatten();

        assertLogicalValues(flattened,
                0, 100, 10, 110, 20, 120,
                1, 101, 11, 111, 21, 121,
                2, 102, 12, 112, 22, 122,
                3, 103, 13, 113, 23, 123);
        flattened.setItem(new int[]{0}, 999);
        assertEquals(Integer.valueOf(0), original.getItem(new int[]{0, 0, 0}));
    }

    @Test
    public void iteratorVisitsAllCoordinatesInRowMajorOrder() {
        Iterator<int[]> iterator = new JTensor<>(Integer.class, new int[]{2, 2, 2}).indicesIterator();
        List<String> indices = new ArrayList<>();
        while (iterator.hasNext()) {
            indices.add(Arrays.toString(iterator.next()));
        }

        assertEquals(Arrays.asList(
                "[0, 0, 0]", "[0, 0, 1]", "[0, 1, 0]", "[0, 1, 1]",
                "[1, 0, 0]", "[1, 0, 1]", "[1, 1, 0]", "[1, 1, 1]"), indices);
        assertFalse(iterator.hasNext());
        assertThrows(NoSuchElementException.class, iterator::next);
    }

    @Test
    public void iteratorReturnsIndependentCoordinateArrays() {
        Iterator<int[]> iterator = new JTensor<>(Integer.class, new int[]{2, 2}).indicesIterator();
        int[] first = iterator.next();
        first[0] = 99;

        assertArrayEquals(new int[]{0, 1}, iterator.next());
    }

    @Test
    public void scalarIndexBoundsAreCheckedForGetsAndSets() {
        JTensor<Integer> tensor = cube();

        assertThrows(java.lang.IndexOutOfBoundsException.class, () -> tensor.getItem(new int[]{-1, 0, 0}));
        assertThrows(java.lang.IndexOutOfBoundsException.class, () -> tensor.getItem(new int[]{0, 3, 0}));
        assertThrows(java.lang.IndexOutOfBoundsException.class, () -> tensor.setItem(new int[]{0, 0, 4}, 1));
        assertThrows(InvalidArgumentException.class, () -> tensor.getItem(null));
    }

    @Test
    public void invalidSliceIntervalsAreRejectedBeforeIndexMapping() {
        JTensor<Integer> tensor = cube();

        assertThrows(InvalidArgumentException.class,
                () -> tensor.slice(new int[][]{{0, 2}, {1, 1}, {0, 4}}));
        assertThrows(InvalidArgumentException.class,
                () -> tensor.slice(new int[][]{{0, 2}, {-1, 2}, {0, 4}}));
        assertThrows(InvalidArgumentException.class,
                () -> tensor.slice(new int[][]{{0, 2}, {0, 4}, {0, 4}}));
        assertThrows(InvalidArgumentException.class,
                () -> tensor.slice(new int[][]{{0, 2}, null, {0, 4}}));
    }

    @Test
    public void expandAndSqueezeAtEveryPositionRoundTripValues() {
        JTensor<Integer> base = new JTensor<>(Integer.class, new int[]{2, 3},
                new Integer[]{1, 2, 3, 4, 5, 6});

        for (int dimension = 0; dimension <= 2; dimension++) {
            JTensor<Integer> expanded = base.expand(dimension);
            assertEquals(3, expanded.getShape().length);
            assertEquals(1, expanded.getShape()[dimension]);
            assertEquals(base, expanded.squeeze(dimension));
        }
        assertThrows(InvalidArgumentException.class, () -> base.squeeze(0));
    }

    @Test
    public void reshapeRejectsMismatchedAndInvalidShapes() {
        JTensor<Integer> tensor = cube();

        assertThrows(InvalidArgumentException.class, () -> tensor.reshape(new int[]{23}));
        assertThrows(InvalidShapeException.class, () -> tensor.reshape(new int[]{2, 0, 12}));
        assertThrows(InvalidShapeException.class, () -> tensor.reshape(new int[]{-1, 24}));
    }
}
