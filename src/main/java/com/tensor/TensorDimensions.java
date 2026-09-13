package com.tensor;

import java.util.Arrays;

/** Shared dimension normalization for the extended API. */
final class TensorDimensions {
    /** Prevents instantiation of the dimension utility. */
    private TensorDimensions() { }

    /**
     * Normalizes a negative dimension relative to the tensor rank.
     *
     * @param dimension the dimension, possibly negative
     * @param rank the number of valid dimension positions
     * @return the nonnegative normalized dimension
     * @throws InvalidArgumentException if the dimension is out of range
     */
    static int normalizeDimension(int dimension, int rank) {
        int normalizedDimension = dimension < 0 ? dimension + rank : dimension;
        if (normalizedDimension < 0 || normalizedDimension >= rank) {
            throw new InvalidArgumentException(
                    "dimension " + dimension + " is invalid for rank " + rank);
        }
        return normalizedDimension;
    }

    /**
     * Normalizes and sorts a set of unique dimensions. An empty array selects every dimension.
     * The caller's array is never modified.
     *
     * @param rank the tensor rank
     * @param dimensions the selected dimensions, possibly negative
     * @return a new array of normalized dimensions in ascending order
     * @throws InvalidArgumentException if dimensions is null, duplicated, or out of range
     */
    static int[] normalizeDimensions(int rank, int... dimensions) {
        if (dimensions == null) {
            throw new InvalidArgumentException("dimensions must not be null");
        }
        if (dimensions.length == 0) {
            int[] allDimensions = new int[rank];
            for (int dimension = 0; dimension < rank; dimension++) {
                allDimensions[dimension] = dimension;
            }
            return allDimensions;
        }

        int[] normalizedDimensions = normalizeOrderedDimensions(rank, dimensions);
        Arrays.sort(normalizedDimensions);
        return normalizedDimensions;
    }

    /**
     * Normalizes unique dimensions while retaining their supplied order.
     *
     * @param rank the tensor rank
     * @param dimensions the dimensions to normalize
     * @return normalized dimensions in their original order
     */
    static int[] normalizeOrderedDimensions(int rank, int... dimensions) {
        if (dimensions == null) {
            throw new InvalidArgumentException("dimensions must not be null");
        }

        int[] normalizedDimensions = dimensions.clone();
        boolean[] dimensionAlreadyUsed = new boolean[rank];
        for (int i = 0; i < normalizedDimensions.length; i++) {
            int normalizedDimension =
                    normalizeDimension(normalizedDimensions[i], rank);
            if (dimensionAlreadyUsed[normalizedDimension]) {
                throw new InvalidArgumentException("duplicate dimension");
            }
            normalizedDimensions[i] = normalizedDimension;
            dimensionAlreadyUsed[normalizedDimension] = true;
        }
        return normalizedDimensions;
    }
}
