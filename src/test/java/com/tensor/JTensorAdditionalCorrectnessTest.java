package com.tensor;

import org.junit.Test;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;

import static org.junit.Assert.*;

public class JTensorAdditionalCorrectnessTest {
    private static JTensor<Double> doubleMatrix() {
        return new JTensor<>(
                Double.class,
                new int[]{2, 3},
                new Double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    }

    @Test
    public void replaceChangesOnlyMatchingItems() {
        JTensor<Integer> tensor = new JTensor<>(
                Integer.class, new int[]{4}, new Integer[]{1, 2, 3, 4});

        JTensor<Integer> replaced = tensor.replace(x -> x % 2 == 0, x -> x * 10);

        assertArrayEquals(new Integer[]{1, 20, 3, 40}, replaced.getData());
    }

    @Test
    public void meanUsesRequestedDimensionSize() {
        JTensor<Double> alongRows = JTensor.mean(doubleMatrix(), 0, true);
        JTensor<Double> alongColumns = JTensor.mean(doubleMatrix(), 1, false);

        assertArrayEquals(new int[]{1, 3}, alongRows.getShape());
        assertArrayEquals(new Double[]{2.5, 3.5, 4.5}, alongRows.getData());
        assertArrayEquals(new int[]{2}, alongColumns.getShape());
        assertArrayEquals(new Double[]{2.0, 5.0}, alongColumns.getData());
    }

    @Test
    public void varianceBroadcastsReducedMeanAlongRequestedDimension() {
        JTensor<Double> alongRows = JTensor.var(doubleMatrix(), 0, false);
        JTensor<Double> alongColumns = JTensor.var(doubleMatrix(), 1, false);

        assertArrayEquals(new Double[]{2.25, 2.25, 2.25}, alongRows.getData());
        assertArrayEquals(new Double[]{2.0 / 3.0, 2.0 / 3.0}, alongColumns.getData());
    }

    @Test
    public void floatingPointMaxHandlesAllNegativeValues() {
        JTensor<Double> tensor = new JTensor<>(
                Double.class, new int[]{3}, new Double[]{-5.0, -2.0, -7.0});

        assertEquals(Double.valueOf(-2.0), JTensor.max(tensor, 0, true).getItem(new int[]{0}));
    }

    @Test
    public void floatingPointMinHandlesPositiveInfinity() {
        JTensor<Float> tensor = new JTensor<>(
                Float.class, new int[]{1}, new Float[]{Float.POSITIVE_INFINITY});

        assertEquals(Float.valueOf(Float.POSITIVE_INFINITY),
                JTensor.min(tensor, 0, true).getItem(new int[]{0}));
    }

    @Test
    public void byteAndShortExtremaKeepTheirRuntimeTypes() {
        JTensor<Byte> bytes = new JTensor<>(
                Byte.class, new int[]{3}, new Byte[]{-3, 7, 2});
        JTensor<Short> shorts = new JTensor<>(
                Short.class, new int[]{3}, new Short[]{-30, 70, 20});

        assertEquals(Byte.valueOf((byte) 7), JTensor.max(bytes, 0, true).getItem(new int[]{0}));
        assertEquals(Byte.valueOf((byte) -3), JTensor.min(bytes, 0, true).getItem(new int[]{0}));
        assertEquals(Short.valueOf((short) 70), JTensor.max(shorts, 0, true).getItem(new int[]{0}));
        assertEquals(Short.valueOf((short) -30), JTensor.min(shorts, 0, true).getItem(new int[]{0}));
    }

    @Test
    public void oversizedShapeIsRejectedBeforeAllocation() {
        InvalidShapeException exception = assertThrows(
                InvalidShapeException.class,
                () -> new JTensor<>(Integer.class, new int[]{65_536, 65_536}));

        assertTrue(exception.getMessage().contains("too large"));
    }

    @Test
    public void dimensionSensitiveOperationsRejectInvalidDimensions() {
        JTensor<Double> tensor = doubleMatrix();

        assertThrows(InvalidArgumentException.class, () -> tensor.squeeze(-1));
        assertThrows(InvalidArgumentException.class, () -> tensor.swapDimensions(0, 2));
        assertThrows(InvalidArgumentException.class, () -> tensor.reverse(2));
        assertThrows(InvalidArgumentException.class, () -> tensor.expand(3));
        assertThrows(InvalidArgumentException.class, () -> tensor.concatenate(tensor, 2));
        assertThrows(InvalidArgumentException.class, () -> JTensor.argMax(tensor, 2, true));
    }

    @Test
    public void maskRankCannotExceedTensorRank() {
        JTensor<Boolean> mask = JTensor.repeat(new int[]{1, 1, 1}, true);

        assertThrows(InvalidArgumentException.class, () -> doubleMatrix().applyMask(mask));
    }

    @Test
    public void resizeCannotGrowTensor() {
        assertThrows(InvalidArgumentException.class, () -> doubleMatrix().resize(new int[]{7}));
    }

    @Test
    public void malformedNestedArraysAreRejectedClearly() {
        assertThrows(
                InvalidArgumentException.class,
                () -> JTensor.from2DArray(Integer.class, new Integer[0][0]));
        assertThrows(
                InvalidArgumentException.class,
                () -> JTensor.from3DArray(Integer.class, new Integer[][][]{{null}}));
    }

    @Test
    public void nullElementsCanBeFormatted() {
        assertEquals("[null, null]", new JTensor<>(String.class, new int[]{2}).toString());
    }

    @Test
    public void emptyTensorIteratorNeverProducesAnIndex() {
        Iterator<int[]> iterator = JTensor.empty(Integer.class).indicesIterator();

        assertFalse(iterator.hasNext());
        assertThrows(NoSuchElementException.class, iterator::next);
    }

    @Test
    public void deserializationRejectsNonContiguousStrides() {
        byte[] serialized = doubleMatrix().toByteArray();
        ByteBuffer.wrap(serialized).putInt(13, 99);

        assertThrows(InvalidArgumentException.class, () -> JTensor.fromByteArray(serialized));
    }

    @Test
    public void deserializationRejectsIncorrectPayloadLength() {
        byte[] serialized = doubleMatrix().toByteArray();

        assertThrows(
                InvalidArgumentException.class,
                () -> JTensor.fromByteArray(Arrays.copyOf(serialized, serialized.length - 1)));
        assertThrows(
                InvalidArgumentException.class,
                () -> JTensor.fromByteArray(Arrays.copyOf(serialized, serialized.length + 1)));
    }
}
