package com.tensor;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import static org.junit.Assert.*;

public class JTensorPropertyAuditTest {
    private static int flatIndex(int[] index, int[] shape) {
        int flat = 0;
        for (int i = 0; i < shape.length; i++) {
            flat = flat * shape[i] + index[i];
        }
        return flat;
    }

    private static JTensor<Integer> indexedTensor(int[] shape) {
        return new JTensor<>(Integer.class, shape,
                index -> flatIndex(index, shape) + 1);
    }

    private static Object[] logicalValues(JTensor<?> tensor) {
        Object[] values = new Object[tensor.getSize()];
        Iterator<int[]> iterator = tensor.indicesIterator();
        int position = 0;
        while (iterator.hasNext()) {
            values[position++] = tensor.getItem(iterator.next());
        }
        return values;
    }

    @Test
    public void everySwapReverseAndTransposeCoordinateMapsToSource() {
        int[][] shapes = {
                {2}, {2, 3}, {2, 3, 4}, {2, 2, 3, 2}
        };

        for (int[] shape : shapes) {
            JTensor<Integer> source = indexedTensor(shape);
            JTensor<Integer> transposed = source.transpose();
            Iterator<int[]> transposeIndices = transposed.indicesIterator();
            while (transposeIndices.hasNext()) {
                int[] output = transposeIndices.next();
                int[] input = new int[output.length];
                for (int i = 0; i < output.length; i++) {
                    input[i] = output[output.length - 1 - i];
                }
                assertEquals(source.getItem(input), transposed.getItem(output));
            }

            for (int dimension = 0; dimension < shape.length; dimension++) {
                JTensor<Integer> reversed = source.reverse(dimension);
                Iterator<int[]> iterator = reversed.indicesIterator();
                while (iterator.hasNext()) {
                    int[] output = iterator.next();
                    int[] input = Arrays.copyOf(output, output.length);
                    input[dimension] = shape[dimension] - 1 - input[dimension];
                    assertEquals(source.getItem(input), reversed.getItem(output));
                }
            }

            for (int first = 0; first < shape.length; first++) {
                for (int second = 0; second < shape.length; second++) {
                    JTensor<Integer> swapped = source.swapDimensions(first, second);
                    Iterator<int[]> iterator = swapped.indicesIterator();
                    while (iterator.hasNext()) {
                        int[] output = iterator.next();
                        int[] input = Arrays.copyOf(output, output.length);
                        input[first] = output[second];
                        input[second] = output[first];
                        assertEquals(source.getItem(input), swapped.getItem(output));
                    }
                }
            }
        }
    }

    @Test
    public void broadcastingMatchesIndependentReferenceAcrossManyShapes() {
        int[][] shapes = {
                {1}, {2}, {3}, {1, 1}, {1, 2}, {2, 1}, {2, 3}, {3, 1},
                {1, 1, 1}, {2, 1, 3}, {1, 2, 1}, {2, 2, 3}
        };

        for (int[] leftShape : shapes) {
            for (int[] rightShape : shapes) {
                int[] outputShape = referenceBroadcastShape(leftShape, rightShape);
                JTensor<Integer> left = indexedTensor(leftShape);
                JTensor<Integer> right = indexedTensor(rightShape);

                if (outputShape == null) {
                    assertThrows(InvalidArgumentException.class,
                            () -> JTensor.add(left, right));
                    continue;
                }

                JTensor<Integer> actual = JTensor.add(left, right);
                assertArrayEquals(outputShape, actual.getShape());
                Iterator<int[]> iterator = actual.indicesIterator();
                while (iterator.hasNext()) {
                    int[] output = iterator.next();
                    int expected = left.getItem(projectBroadcastIndex(output, leftShape))
                            + right.getItem(projectBroadcastIndex(output, rightShape));
                    assertEquals(Integer.valueOf(expected), actual.getItem(output));
                }
            }
        }
    }

    private static int[] referenceBroadcastShape(int[] first, int[] second) {
        int rank = Math.max(first.length, second.length);
        int[] output = new int[rank];
        for (int outputDimension = rank - 1; outputDimension >= 0; outputDimension--) {
            int firstDimension = outputDimension - (rank - first.length);
            int secondDimension = outputDimension - (rank - second.length);
            int firstSize = firstDimension < 0 ? 1 : first[firstDimension];
            int secondSize = secondDimension < 0 ? 1 : second[secondDimension];
            if (firstSize != secondSize && firstSize != 1 && secondSize != 1) {
                return null;
            }
            output[outputDimension] = Math.max(firstSize, secondSize);
        }
        return output;
    }

    private static int[] projectBroadcastIndex(int[] output, int[] inputShape) {
        int[] input = new int[inputShape.length];
        int offset = output.length - inputShape.length;
        for (int i = 0; i < input.length; i++) {
            input[i] = inputShape[i] == 1 ? 0 : output[offset + i];
        }
        return input;
    }

    @Test
    public void reductionsMatchIndependentGroupingForMaterializedAndViewLayouts() {
        JTensor<Integer> source = indexedTensor(new int[]{2, 3, 4});
        List<JTensor<Integer>> layouts = new ArrayList<>();
        layouts.add(source);
        layouts.add(source.transpose());
        layouts.add(source.swapDimensions(0, 1));
        layouts.add(source.reverse(1));
        layouts.add(source.slice(new int[][]{{0, 2}, {1, 3}, {0, 4}}));
        layouts.add(source.transpose().slice(new int[][]{{1, 4}, {0, 3}, {0, 2}}));

        for (JTensor<Integer> tensor : layouts) {
            int[] shape = tensor.getShape();
            for (int dimension = 0; dimension < shape.length; dimension++) {
                int[] alongShape = Arrays.copyOf(shape, shape.length);
                alongShape[dimension] = 1;
                Integer[] expectedAlong = new Integer[product(alongShape)];
                Arrays.fill(expectedAlong, 0);

                int[] allShape = Arrays.copyOf(shape, shape.length);
                for (int i = dimension; i < allShape.length; i++) {
                    allShape[i] = 1;
                }
                Integer[] expectedAll = new Integer[product(allShape)];
                Arrays.fill(expectedAll, 0);

                Iterator<int[]> iterator = tensor.indicesIterator();
                while (iterator.hasNext()) {
                    int[] input = iterator.next();
                    int value = tensor.getItem(input);

                    int[] alongIndex = Arrays.copyOf(input, input.length);
                    alongIndex[dimension] = 0;
                    int alongFlat = flatIndex(alongIndex, alongShape);
                    expectedAlong[alongFlat] += value;

                    int[] allIndex = Arrays.copyOf(input, input.length);
                    for (int i = dimension; i < allIndex.length; i++) {
                        allIndex[i] = 0;
                    }
                    int allFlat = flatIndex(allIndex, allShape);
                    expectedAll[allFlat] += value;
                }

                JTensor<Integer> along = JTensor.sum(tensor, dimension, true);
                JTensor<Integer> all = tensor.reduceAll(0, Integer::sum, dimension, true);
                assertArrayEquals(alongShape, along.getShape());
                assertArrayEquals(expectedAlong, logicalValues(along));
                assertArrayEquals(allShape, all.getShape());
                assertArrayEquals(expectedAll, logicalValues(all));
            }
        }
    }

    private static int product(int[] shape) {
        int result = 1;
        for (int dimension : shape) {
            result *= dimension;
        }
        return result;
    }
}
