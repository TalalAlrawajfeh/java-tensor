package com.tensor;

import java.lang.reflect.Array;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.DoubleUnaryOperator;

/**
 * Implementations for the extended API. Numeric casts are confined here because
 * JTensor also supports nonnumeric element types.
 */
final class TensorOperations {
    /** Prevents instantiation of this implementation utility. */
    private TensorOperations() { }

    /**
     * Validates the supported numeric runtime type and rejects null elements.
     *
     * @param tensor the input tensor
     * @return the validated numeric element class
     * @throws InvalidArgumentException if the tensor or any element is null
     * @throws IllegalArgumentException if its element type is not supported
     */
    @SuppressWarnings("unchecked")
    private static <T extends Number> Class<T> validateNumericTensor(JTensor<?> tensor) {
        if (tensor == null) {
            throw new InvalidArgumentException("tensor must not be null");
        }

        Class<T> numericType = (Class<T>) tensor.getType();
        NumberHelper.zero(numericType);
        Iterator<int[]> indicesIterator = tensor.indicesIterator();
        while (indicesIterator.hasNext()) {
            if (tensor.getItem(indicesIterator.next()) == null) {
                throw new InvalidArgumentException("numeric values must not be null");
            }
        }
        return numericType;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> JTensor<T> add(JTensor<T> tensor1, JTensor<T> tensor2) {
        validateMatchingNumericTypes(tensor1, tensor2);
        return (JTensor<T>) JTensor.add((JTensor) tensor1, (JTensor) tensor2);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> JTensor<T> subtract(JTensor<T> tensor1, JTensor<T> tensor2) {
        validateMatchingNumericTypes(tensor1, tensor2);
        return (JTensor<T>) JTensor.subtract((JTensor) tensor1, (JTensor) tensor2);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> JTensor<T> multiply(JTensor<T> tensor1, JTensor<T> tensor2) {
        validateMatchingNumericTypes(tensor1, tensor2);
        return (JTensor<T>) JTensor.multiply((JTensor) tensor1, (JTensor) tensor2);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> JTensor<T> divide(JTensor<T> tensor1, JTensor<T> tensor2) {
        validateMatchingNumericTypes(tensor1, tensor2);
        return (JTensor<T>) JTensor.divide((JTensor) tensor1, (JTensor) tensor2);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> JTensor<T> pow(JTensor<T> tensor1, JTensor<T> tensor2) {
        validateMatchingNumericTypes(tensor1, tensor2);
        return (JTensor<T>) JTensor.pow((JTensor) tensor1, (JTensor) tensor2);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> JTensor<T> mod(JTensor<T> tensor1, JTensor<T> tensor2) {
        validateMatchingNumericTypes(tensor1, tensor2);
        return (JTensor<T>) JTensor.mod((JTensor) tensor1, (JTensor) tensor2);
    }

    private static Class<? extends Number> validateMatchingNumericTypes(
            JTensor<?> tensor1,
            JTensor<?> tensor2) {
        Class<? extends Number> numericType = validateNumericTensor(tensor1);
        validateNumericTensor(tensor2);
        if (tensor1.getType() != tensor2.getType()) {
            throw new InvalidArgumentException("numeric types must match");
        }
        return numericType;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static <T> T castNumber(Class<? extends Number> type, Number value) {
        return (T) NumberHelper.cast((Class) type, value);
    }

    /**
     * Ensures two operands are present and have the same runtime element type.
     */
    private static void validateMatchingTensorTypes(JTensor<?> tensor1, JTensor<?> tensor2) {
        if (tensor1 == null || tensor2 == null || tensor1.getType() != tensor2.getType()) {
            throw new InvalidArgumentException("tensors must have matching types");
        }
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> JTensor<Boolean> isEqual(JTensor<T> tensor1, JTensor<T> tensor2) {
        validateMatchingTensorTypes(tensor1, tensor2);
        if (!Number.class.isAssignableFrom(tensor1.getType())) {
            return JTensor.applyBinaryOperation(
                    Boolean.class,
                    tensor1,
                    tensor2,
                    Objects::equals);
        }
        validateNumericTensor(tensor1);
        validateNumericTensor(tensor2);
        return JTensor.equals((JTensor) tensor1, (JTensor) tensor2);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> JTensor<Boolean> isNotEqual(JTensor<T> tensor1, JTensor<T> tensor2) {
        validateMatchingTensorTypes(tensor1, tensor2);
        if (!Number.class.isAssignableFrom(tensor1.getType())) {
            return JTensor.applyBinaryOperation(
                    Boolean.class,
                    tensor1,
                    tensor2,
                    (value1, value2) -> !Objects.equals(value1, value2));
        }
        validateNumericTensor(tensor1);
        validateNumericTensor(tensor2);
        return JTensor.notEquals((JTensor) tensor1, (JTensor) tensor2);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> JTensor<Boolean> isLessThan(JTensor<T> tensor1, JTensor<T> tensor2) {
        validateMatchingNumericTypes(tensor1, tensor2);
        return JTensor.lessThan((JTensor) tensor1, (JTensor) tensor2);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> JTensor<Boolean> isGreaterThan(JTensor<T> tensor1, JTensor<T> tensor2) {
        validateMatchingNumericTypes(tensor1, tensor2);
        return JTensor.greaterThan((JTensor) tensor1, (JTensor) tensor2);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> JTensor<Boolean> isLessThanOrEqual(JTensor<T> tensor1, JTensor<T> tensor2) {
        validateMatchingNumericTypes(tensor1, tensor2);
        return JTensor.lessThanOrEquals((JTensor) tensor1, (JTensor) tensor2);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> JTensor<Boolean> isGreaterThanOrEqual(JTensor<T> tensor1, JTensor<T> tensor2) {
        validateMatchingNumericTypes(tensor1, tensor2);
        return JTensor.greaterThanOrEquals((JTensor) tensor1, (JTensor) tensor2);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> JTensor<T> sqrt(JTensor<T> tensor) {
        validateNumericTensor(tensor);
        return JTensor.sqrt((JTensor) tensor);
    }

    static <T> JTensor<T> negate(JTensor<T> tensor) {
        Class<? extends Number> numericType = validateNumericTensor(tensor);
        Number zero = NumberHelper.zero(numericType);
        return tensor.map(
                tensor.getType(),
                value -> negateValue(numericType, (Number) value, zero));
    }

    static <T> JTensor<T> abs(JTensor<T> tensor) {
        Class<? extends Number> numericType = validateNumericTensor(tensor);
        Number zero = NumberHelper.zero(numericType);
        return tensor.map(
                tensor.getType(),
                value -> absoluteValue(numericType, (Number) value, zero));
    }

    static <T> JTensor<T> exp(JTensor<T> tensor) {
        return applyDoubleUnaryOperation(tensor, Math::exp);
    }

    static <T> JTensor<T> log(JTensor<T> tensor) {
        return applyDoubleUnaryOperation(tensor, Math::log);
    }

    static <T> JTensor<T> sin(JTensor<T> tensor) {
        return applyDoubleUnaryOperation(tensor, Math::sin);
    }

    static <T> JTensor<T> cos(JTensor<T> tensor) {
        return applyDoubleUnaryOperation(tensor, Math::cos);
    }

    static <T> JTensor<T> tanh(JTensor<T> tensor) {
        return applyDoubleUnaryOperation(tensor, Math::tanh);
    }

    static <T> JTensor<T> floor(JTensor<T> tensor) {
        return applyRoundingOperation(tensor, Math::floor);
    }

    static <T> JTensor<T> ceil(JTensor<T> tensor) {
        return applyRoundingOperation(tensor, Math::ceil);
    }

    static <T> JTensor<T> round(JTensor<T> tensor) {
        return applyRoundingOperation(tensor, Math::rint);
    }

    private static <T> JTensor<T> applyRoundingOperation(
            JTensor<T> tensor,
            DoubleUnaryOperator operation) {
        Class<? extends Number> numericType = validateNumericTensor(tensor);
        if (numericType != Float.class && numericType != Double.class) {
            return tensor.copy();
        }
        return applyDoubleUnaryOperation(tensor, numericType, operation);
    }

    private static <T> JTensor<T> applyDoubleUnaryOperation(
            JTensor<T> tensor,
            DoubleUnaryOperator operation) {
        Class<? extends Number> numericType = validateNumericTensor(tensor);
        return applyDoubleUnaryOperation(tensor, numericType, operation);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static <T> JTensor<T> applyDoubleUnaryOperation(
            JTensor<T> tensor,
            Class<? extends Number> numericType,
            DoubleUnaryOperator operation) {
        return tensor.map(
                tensor.getType(),
                value -> (T) NumberHelper.cast(
                        (Class) numericType,
                        operation.applyAsDouble(((Number) value).doubleValue())));
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static <T> T negateValue(
            Class<? extends Number> numericType,
            Number value,
            Number zero) {
        if (numericType == Double.class) {
            return (T) Double.valueOf(-value.doubleValue());
        }
        if (numericType == Float.class) {
            return (T) Float.valueOf(-value.floatValue());
        }
        return (T) NumberHelper.subtract((Class) numericType, zero, value);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static <T> T absoluteValue(
            Class<? extends Number> numericType,
            Number value,
            Number zero) {
        if (numericType == Double.class) {
            return (T) Double.valueOf(Math.abs(value.doubleValue()));
        }
        if (numericType == Float.class) {
            return (T) Float.valueOf(Math.abs(value.floatValue()));
        }
        if (NumberHelper.lessThan((Class) numericType, value, zero)) {
            return (T) NumberHelper.subtract((Class) numericType, zero, value);
        }
        return (T) value;
    }

    /**
     * Stores the axis order and shapes used to reduce one or more dimensions.
     * Unreduced dimensions are packed first and reduced dimensions follow them.
     */
    private static final class ReductionPlan {
        private final int[] reducedDimensions;
        private final int[] permutation;
        private final int[] inversePermutation;
        private final int[] reducedShape;
        private final int[] keptDimensionsShape;
        private final int numberOfGroups;
        private final int groupSize;

        /**
         * Builds a reduction plan for the selected dimensions.
         *
         * @param tensor the input tensor
         * @param dimensions the dimensions to reduce; empty selects every dimension
         */
        private ReductionPlan(JTensor<?> tensor, int... dimensions) {
            int[] tensorShape = tensor.getShape();
            reducedDimensions = TensorDimensions.normalizeDimensions(tensorShape.length, dimensions);
            boolean[] isReducedDimension = new boolean[tensorShape.length];
            for (int dimension : reducedDimensions) {
                isReducedDimension[dimension] = true;
            }

            permutation = new int[tensorShape.length];
            inversePermutation = new int[tensorShape.length];
            reducedShape = new int[Math.max(1, tensorShape.length - reducedDimensions.length)];
            keptDimensionsShape = tensorShape.clone();
            Arrays.fill(reducedShape, 1);

            int permutationIndex = 0;
            int currentNumberOfGroups = 1;
            int currentGroupSize = 1;
            for (int dimension = 0; dimension < tensorShape.length; dimension++) {
                if (!isReducedDimension[dimension]) {
                    permutation[permutationIndex] = dimension;
                    reducedShape[permutationIndex] = tensorShape[dimension];
                    permutationIndex++;
                    currentNumberOfGroups *= tensorShape[dimension];
                }
            }
            for (int dimension : reducedDimensions) {
                permutation[permutationIndex++] = dimension;
                keptDimensionsShape[dimension] = 1;
                currentGroupSize *= tensorShape[dimension];
            }
            for (int dimension = 0; dimension < permutation.length; dimension++) {
                inversePermutation[permutation[dimension]] = dimension;
            }

            numberOfGroups = currentNumberOfGroups;
            groupSize = currentGroupSize;
        }

        private <T> JTensor<T> pack(JTensor<T> tensor) {
            return tensor.permute(permutation).reshape(new int[]{numberOfGroups, groupSize});
        }

        private int[] getResultShape(boolean keepDimensions) {
            return keepDimensions ? keptDimensionsShape : reducedShape;
        }
    }

    private enum ReductionOperation {
        SUM,
        MEAN,
        MIN,
        MAX,
        PRODUCT,
        ARG_MAX,
        ARG_MIN
    }

    /**
     * Packs selected axes into one dimension and applies a single-axis reduction.
     *
     * @param <T> the input element type
     * @param <R> the result element type selected by the operation
     * @param tensor the input tensor
     * @param operation the reduction to apply
     * @param keepDimensions whether to retain reduced dimensions with size one
     * @param dimensions the dimensions to reduce; empty selects every dimension
     * @return a tensor reshaped to retained or removed reduction dimensions
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    private static <T, R> JTensor<R> reduce(
            JTensor<T> tensor,
            ReductionOperation operation,
            boolean keepDimensions,
            int... dimensions) {
        Class<? extends Number> numericType = validateNumericTensor(tensor);
        ReductionPlan reductionPlan = new ReductionPlan(tensor, dimensions);

        if (tensor.size() == 0) {
            if (operation == ReductionOperation.SUM) {
                return (JTensor<R>) JTensor.singleValue(
                        (Class) numericType,
                        NumberHelper.zero(numericType));
            }
            if (operation == ReductionOperation.PRODUCT) {
                return (JTensor<R>) JTensor.singleValue(
                        (Class) numericType,
                        NumberHelper.one(numericType));
            }
            throw new InvalidArgumentException(
                    operation.name().toLowerCase() + " requires a nonempty tensor");
        }

        JTensor packedTensor = reductionPlan.pack(tensor);
        JTensor reducedTensor;
        switch (operation) {
            case SUM:
                reducedTensor = JTensor.sum(packedTensor, 1, false);
                break;
            case MEAN:
                reducedTensor = meanGroups(
                        packedTensor,
                        numericType,
                        reductionPlan.numberOfGroups,
                        reductionPlan.groupSize);
                break;
            case MIN:
                reducedTensor = JTensor.min(packedTensor, 1, false);
                break;
            case MAX:
                reducedTensor = JTensor.max(packedTensor, 1, false);
                break;
            case PRODUCT:
                reducedTensor = JTensor.product(packedTensor, 1, false);
                break;
            case ARG_MAX:
                reducedTensor = JTensor.argMax(packedTensor, 1, false);
                break;
            case ARG_MIN:
                reducedTensor = JTensor.argMin(packedTensor, 1, false);
                break;
            default:
                throw new AssertionError(operation);
        }
        return reducedTensor.reshape(reductionPlan.getResultShape(keepDimensions));
    }

    static <T> JTensor<T> sum(JTensor<T> tensor, boolean keepDimensions, int... dimensions) {
        return reduce(tensor, ReductionOperation.SUM, keepDimensions, dimensions);
    }

    static <T> JTensor<T> mean(JTensor<T> tensor, boolean keepDimensions, int... dimensions) {
        return reduce(tensor, ReductionOperation.MEAN, keepDimensions, dimensions);
    }

    static <T> JTensor<T> min(JTensor<T> tensor, boolean keepDimensions, int... dimensions) {
        return reduce(tensor, ReductionOperation.MIN, keepDimensions, dimensions);
    }

    static <T> JTensor<T> max(JTensor<T> tensor, boolean keepDimensions, int... dimensions) {
        return reduce(tensor, ReductionOperation.MAX, keepDimensions, dimensions);
    }

    static <T> JTensor<T> product(JTensor<T> tensor, boolean keepDimensions, int... dimensions) {
        return reduce(tensor, ReductionOperation.PRODUCT, keepDimensions, dimensions);
    }

    static <T> JTensor<Integer> argMax(
            JTensor<T> tensor,
            boolean keepDimensions,
            int... dimensions) {
        return reduce(tensor, ReductionOperation.ARG_MAX, keepDimensions, dimensions);
    }

    static <T> JTensor<Integer> argMin(
            JTensor<T> tensor,
            boolean keepDimensions,
            int... dimensions) {
        return reduce(tensor, ReductionOperation.ARG_MIN, keepDimensions, dimensions);
    }

    /**
     * Clamps numeric elements to validated inclusive bounds.
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    static <T> JTensor<T> clip(JTensor<T> tensor, T lowerBound, T upperBound) {
        Class<? extends Number> numericType = validateNumericTensor(tensor);
        if (!tensor.getType().isInstance(lowerBound)
                || !tensor.getType().isInstance(upperBound)) {
            throw new InvalidArgumentException("bounds must match tensor type");
        }
        if (NumberHelper.compare(
                (Class) numericType,
                (Number) lowerBound,
                (Number) upperBound) > 0) {
            throw new InvalidArgumentException("lower bound exceeds upper bound");
        }

        return tensor.map(
                tensor.getType(),
                value -> (T) NumberHelper.min(
                        (Class) numericType,
                        (Number) upperBound,
                        NumberHelper.max(
                                (Class) numericType,
                                (Number) lowerBound,
                                (Number) value)));
    }

    /**
     * Broadcasts three operands and selects each result from the true or false tensor.
     */
    static <T> JTensor<T> where(
            JTensor<Boolean> condition,
            JTensor<T> tensorIfTrue,
            JTensor<T> tensorIfFalse) {
        if (condition == null || tensorIfTrue == null || tensorIfFalse == null) {
            throw new InvalidArgumentException("where operands must not be null");
        }
        if (tensorIfTrue.getType() != tensorIfFalse.getType()) {
            throw new InvalidArgumentException("where value tensors must have matching types");
        }

        Pair<JTensor<T>, JTensor<T>> broadcastValues =
                JTensor.broadcast(tensorIfTrue, tensorIfFalse);
        Pair<JTensor<Boolean>, JTensor<T>> broadcastConditionAndTrueValues =
                JTensor.broadcast(condition, broadcastValues.getFirst());
        JTensor<Boolean> broadcastCondition = broadcastConditionAndTrueValues.getFirst();
        JTensor<T> broadcastTrueValues = broadcastConditionAndTrueValues.getSecond();
        JTensor<T> broadcastFalseValues =
                JTensor.broadcast(tensorIfFalse, broadcastTrueValues).getFirst();

        JTensor<T> result = new JTensor<>(
                tensorIfTrue.getType(),
                broadcastCondition.getShape());
        Iterator<int[]> indicesIterator = broadcastCondition.indicesIterator();
        while (indicesIterator.hasNext()) {
            int[] indices = indicesIterator.next();
            Boolean conditionValue = broadcastCondition.getItem(indices);
            if (conditionValue == null) {
                throw new InvalidArgumentException("condition must not contain null");
            }
            T resultValue = conditionValue
                    ? broadcastTrueValues.getItem(indices)
                    : broadcastFalseValues.getItem(indices);
            result.setItem(indices, resultValue);
        }
        return result;
    }

    /**
     * Broadcasts a mask and collects selected elements in logical order.
     */
    static <T> JTensor<T> maskedSelect(JTensor<T> tensor, JTensor<Boolean> mask) {
        Pair<JTensor<T>, JTensor<Boolean>> broadcastTensors = JTensor.broadcast(tensor, mask);
        JTensor<T> broadcastTensor = broadcastTensors.getFirst();
        JTensor<Boolean> broadcastMask = broadcastTensors.getSecond();
        List<T> selectedValues = new ArrayList<>();

        Iterator<int[]> indicesIterator = broadcastTensor.indicesIterator();
        while (indicesIterator.hasNext()) {
            int[] indices = indicesIterator.next();
            Boolean isSelected = broadcastMask.getItem(indices);
            if (isSelected == null) {
                throw new InvalidArgumentException("mask must not contain null");
            }
            if (isSelected) {
                selectedValues.add(broadcastTensor.getItem(indices));
            }
        }
        return fromList(tensor.getType(), selectedValues);
    }

    /**
     * Copies a list into typed tensor storage, preserving the rank-zero empty convention.
     *
     * @param <T> the input element type
     * @param type the runtime element class
     * @param values the values to copy
     * @return a detached flat tensor
     */
    private static <T> JTensor<T> fromList(Class<T> type, List<T> values) {
        if (values.isEmpty()) {
            return JTensor.empty(type);
        }
        @SuppressWarnings("unchecked")
        T[] valuesArray = (T[]) Array.newInstance(type, values.size());
        return JTensor.from1DArray(type, values.toArray(valuesArray));
    }

    /**
     * Normalizes a possibly negative element index and validates its bounds.
     *
     * @param index the possibly negative element index
     * @param size the positive axis size
     * @return the nonnegative index
     * @throws InvalidArgumentException if the normalized index is out of range
     */
    private static int index(int index, int size) {
        int normalizedIndex = index < 0 ? index + size : index;
        if (normalizedIndex < 0 || normalizedIndex >= size) {
            throw new InvalidArgumentException("index out of bounds");
        }
        return normalizedIndex;
    }

    /**
     * Selects complete slices along one dimension using normalized indices.
     */
    static <T> JTensor<T> take(JTensor<T> tensor, int[] indices, int dimension) {
        if (indices == null) {
            throw new InvalidArgumentException("indices must not be null");
        }
        if (tensor.size() == 0 && indices.length == 0) {
            return JTensor.empty(tensor.getType());
        }

        int normalizedDimension = TensorDimensions.normalizeDimension(dimension, tensor.numberOfDimensions());
        int[] normalizedIndices = indices.clone();
        for (int i = 0; i < normalizedIndices.length; i++) {
            normalizedIndices[i] = index(normalizedIndices[i], tensor.size(normalizedDimension));
        }
        if (normalizedIndices.length == 0) {
            return JTensor.empty(tensor.getType());
        }

        int[] resultShape = tensor.getShape();
        resultShape[normalizedDimension] = normalizedIndices.length;
        return new JTensor<>(tensor.getType(), resultShape, resultIndices -> {
            int selectedIndex = normalizedIndices[resultIndices[normalizedDimension]];
            resultIndices[normalizedDimension] = selectedIndex;
            return tensor.getItem(resultIndices);
        });
    }

    /**
     * Selects individual values along one dimension using an index tensor.
     */
    static <T> JTensor<T> gather(
            JTensor<T> tensor,
            JTensor<Integer> indexTensor,
            int dimension) {
        int normalizedDimension = TensorDimensions.normalizeDimension(dimension, tensor.numberOfDimensions());
        if (indexTensor == null
                || indexTensor.numberOfDimensions() != tensor.numberOfDimensions()) {
            throw new InvalidArgumentException("gather indices must have the same rank");
        }
        for (int currentDimension = 0;
                currentDimension < tensor.numberOfDimensions();
                currentDimension++) {
            if (currentDimension != normalizedDimension
                    && indexTensor.size(currentDimension) != tensor.size(currentDimension)) {
                throw new InvalidArgumentException("non-gather dimensions must match");
            }
        }

        return new JTensor<>(tensor.getType(), indexTensor.getShape(), resultIndices -> {
            Integer selectedIndex = indexTensor.getItem(resultIndices);
            if (selectedIndex == null) {
                throw new InvalidArgumentException("indices must not contain null");
            }
            resultIndices[normalizedDimension] =
                    index(selectedIndex, tensor.size(normalizedDimension));
            return tensor.getItem(resultIndices);
        });
    }

    /**
     * Inserts a dimension into each input tensor and concatenates the results.
     */
    static <T> JTensor<T> stack(int dimension, JTensor<T>[] tensors) {
        validateTensors(tensors);
        int normalizedDimension = TensorDimensions.normalizeDimension(
                dimension,
                tensors[0].numberOfDimensions() + 1);
        JTensor<T> result = tensors[0].unsqueeze(normalizedDimension);
        for (int i = 1; i < tensors.length; i++) {
            if (!Arrays.equals(tensors[0].getShape(), tensors[i].getShape())) {
                throw new InvalidArgumentException("stack shapes must match");
            }
            result = result.concatenate(
                    tensors[i].unsqueeze(normalizedDimension),
                    normalizedDimension);
        }
        return result.copy();
    }

    /**
     * Validates a nonempty array of non-null tensors with matching element types.
     *
     * @param <T> the input element type
     * @param tensors the input tensors in output order
     * @throws InvalidArgumentException if inputs are missing, null, or have different types
     */
    private static <T> void validateTensors(JTensor<T>[] tensors) {
        if (tensors == null || tensors.length == 0) {
            throw new InvalidArgumentException("at least one tensor is required");
        }
        for (JTensor<T> tensor : tensors) {
            if (tensor == null || tensor.getType() != tensors[0].getType()) {
                throw new InvalidArgumentException(
                        "tensors must be non-null and have matching types");
            }
        }
    }

    /**
     * Promotes vectors for vertical joining and concatenates along the appropriate axis.
     *
     * @param <T> the input element type
     * @param vertical true for vertical stacking, false for horizontal stacking
     * @param tensors the input tensors in output order
     * @return a detached concatenation
     */
    private static <T> JTensor<T> concatenateStackedTensors(boolean vertical, JTensor<T>[] tensors) {
        validateTensors(tensors);
        JTensor<T> result = tensors[0];
        if (vertical && result.numberOfDimensions() == 1) {
            result = result.unsqueeze(0);
        }
        int concatenationDimension =
                vertical || result.numberOfDimensions() == 1 ? 0 : 1;
        for (int i = 1; i < tensors.length; i++) {
            JTensor<T> nextTensor = tensors[i];
            if (vertical && nextTensor.numberOfDimensions() == 1) {
                nextTensor = nextTensor.unsqueeze(0);
            }
            result = result.concatenate(nextTensor, concatenationDimension);
        }
        return result.copy();
    }

    static <T> JTensor<T> verticalStack(JTensor<T>[] tensors) {
        return concatenateStackedTensors(true, tensors);
    }

    static <T> JTensor<T> horizontalStack(JTensor<T>[] tensors) {
        return concatenateStackedTensors(false, tensors);
    }

    /**
     * Validates explicit section lengths and creates slices sharing the source storage.
     */
    static <T> List<JTensor<T>> split(
            JTensor<T> tensor,
            int[] sectionLengths,
            int dimension) {
        int normalizedDimension = TensorDimensions.normalizeDimension(dimension, tensor.numberOfDimensions());
        if (sectionLengths == null || sectionLengths.length == 0) {
            throw new InvalidArgumentException("lengths must not be empty");
        }

        long totalLength = 0;
        for (int sectionLength : sectionLengths) {
            if (sectionLength <= 0) {
                throw new InvalidArgumentException("split lengths must be positive");
            }
            totalLength += sectionLength;
        }
        if (totalLength != tensor.size(normalizedDimension)) {
            throw new InvalidArgumentException("split lengths must sum to axis size");
        }

        List<JTensor<T>> result = new ArrayList<>();
        int sectionStart = 0;
        for (int sectionLength : sectionLengths) {
            int[][] intervals = new int[tensor.numberOfDimensions()][2];
            for (int currentDimension = 0;
                    currentDimension < intervals.length;
                    currentDimension++) {
                intervals[currentDimension][1] = tensor.size(currentDimension);
            }
            intervals[normalizedDimension][0] = sectionStart;
            intervals[normalizedDimension][1] = sectionStart + sectionLength;
            result.add(tensor.slice(intervals));
            sectionStart += sectionLength;
        }
        return result;
    }

    /**
     * Contracts paired dimensions after moving them to the matrix multiplication positions.
     */
    static <T> JTensor<T> dotProduct(
            JTensor<T> tensor1,
            JTensor<T> tensor2,
            int[] tensor1Dimensions,
            int[] tensor2Dimensions) {
        validateMatchingNumericTypes(tensor1, tensor2);
        if (tensor1Dimensions == null
                || tensor2Dimensions == null
                || tensor1Dimensions.length != tensor2Dimensions.length
                || tensor1.size() == 0
                || tensor2.size() == 0) {
            throw new InvalidArgumentException(
                    "contraction requires nonempty matching tensors and paired axes");
        }

        int tensor1Rank = tensor1.numberOfDimensions();
        int tensor2Rank = tensor2.numberOfDimensions();
        int[] normalizedTensor1Dimensions =
                TensorDimensions.normalizeOrderedDimensions(
                        tensor1Rank,
                        tensor1Dimensions);
        int[] normalizedTensor2Dimensions =
                TensorDimensions.normalizeOrderedDimensions(
                        tensor2Rank,
                        tensor2Dimensions);
        boolean[] isTensor1Contracted = new boolean[tensor1Rank];
        boolean[] isTensor2Contracted = new boolean[tensor2Rank];
        int contractedSize = 1;
        for (int i = 0; i < normalizedTensor1Dimensions.length; i++) {
            if (tensor1.size(normalizedTensor1Dimensions[i])
                    != tensor2.size(normalizedTensor2Dimensions[i])) {
                throw new InvalidArgumentException("paired contraction sizes must match");
            }

            isTensor1Contracted[normalizedTensor1Dimensions[i]] = true;
            isTensor2Contracted[normalizedTensor2Dimensions[i]] = true;
            contractedSize *= tensor1.size(normalizedTensor1Dimensions[i]);
        }

        int[] tensor1Permutation = new int[tensor1Rank];
        int[] tensor2Permutation = new int[tensor2Rank];
        int[] resultShape = new int[Math.max(
                1,
                tensor1Rank + tensor2Rank - 2 * normalizedTensor1Dimensions.length)];
        Arrays.fill(resultShape, 1);

        int tensor1PermutationIndex = 0;
        int tensor2PermutationIndex = 0;
        int resultDimension = 0;
        int tensor1UncontractedSize = 1;
        int tensor2UncontractedSize = 1;

        for (int dimension = 0; dimension < tensor1Rank; dimension++) {
            if (!isTensor1Contracted[dimension]) {
                tensor1Permutation[tensor1PermutationIndex++] = dimension;
                resultShape[resultDimension++] = tensor1.size(dimension);
                tensor1UncontractedSize *= tensor1.size(dimension);
            }
        }
        for (int dimension : normalizedTensor1Dimensions) {
            tensor1Permutation[tensor1PermutationIndex++] = dimension;
        }
        for (int dimension : normalizedTensor2Dimensions) {
            tensor2Permutation[tensor2PermutationIndex++] = dimension;
        }
        for (int dimension = 0; dimension < tensor2Rank; dimension++) {
            if (!isTensor2Contracted[dimension]) {
                tensor2Permutation[tensor2PermutationIndex++] = dimension;
                resultShape[resultDimension++] = tensor2.size(dimension);
                tensor2UncontractedSize *= tensor2.size(dimension);
            }
        }

        JTensor<T> tensor1Matrix = tensor1
                .permute(tensor1Permutation)
                .reshape(new int[]{tensor1UncontractedSize, contractedSize});
        JTensor<T> tensor2Matrix = tensor2
                .permute(tensor2Permutation)
                .reshape(new int[]{contractedSize, tensor2UncontractedSize});
        return matrixMultiply(tensor1Matrix, tensor2Matrix).reshape(resultShape);
    }

    /**
     * Computes one mean per packed row.
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    private static JTensor meanGroups(
            JTensor packedTensor,
            Class<? extends Number> numericType,
            int numberOfGroups,
            int groupSize) {
        return new JTensor(
                numericType,
                new int[]{numberOfGroups},
                (java.util.function.Function<int[], Number>) indices -> {
                    if (numericType == Float.class || numericType == Double.class) {
                        double sum = 0;
                        for (int groupIndex = 0; groupIndex < groupSize; groupIndex++) {
                            sum += ((Number) packedTensor.getItem(
                                    new int[]{indices[0], groupIndex})).doubleValue();
                        }
                        return NumberHelper.cast(numericType, sum / groupSize);
                    }

                    BigInteger sum = BigInteger.ZERO;
                    for (int groupIndex = 0; groupIndex < groupSize; groupIndex++) {
                        Number value = (Number) packedTensor.getItem(
                                new int[]{indices[0], groupIndex});
                        sum = sum.add(BigInteger.valueOf(value.longValue()));
                    }
                    return NumberHelper.cast(
                            numericType,
                            sum.divide(BigInteger.valueOf(groupSize)));
                });
    }

    /**
     * Computes the dot product of two equal-length vectors.
     */
    static <T> JTensor<T> dotProduct(JTensor<T> tensor1, JTensor<T> tensor2) {
        validateMatchingNumericTypes(tensor1, tensor2);
        if (tensor1.numberOfDimensions() != 1
                || tensor2.numberOfDimensions() != 1
                || tensor1.size() != tensor2.size()) {
            throw new InvalidArgumentException("dotProduct requires equal-length vectors");
        }
        return tensor1.multiply(tensor2).sum();
    }

    /**
     * Multiplies vectors, matrices, or batches of matrices.
     */
    static <T> JTensor<T> matrixMultiply(JTensor<T> tensor1, JTensor<T> tensor2) {
        Class<? extends Number> numericType = validateMatchingNumericTypes(tensor1, tensor2);
        if (tensor1.size() == 0 || tensor2.size() == 0) {
            throw new InvalidArgumentException(
                    "matrixMultiply requires nonempty tensors of matching type");
        }

        boolean tensor1IsVector = tensor1.numberOfDimensions() == 1;
        boolean tensor2IsVector = tensor2.numberOfDimensions() == 1;
        JTensor<T> tensor1Matrix = tensor1IsVector ? tensor1.unsqueeze(0) : tensor1;
        JTensor<T> tensor2Matrix = tensor2IsVector ? tensor2.unsqueeze(1) : tensor2;

        int tensor1Rank = tensor1Matrix.numberOfDimensions();
        int tensor2Rank = tensor2Matrix.numberOfDimensions();
        int numberOfRows = tensor1Matrix.size(-2);
        int contractedSize = tensor1Matrix.size(-1);
        int numberOfColumns = tensor2Matrix.size(-1);
        if (contractedSize != tensor2Matrix.size(-2)) {
            throw new InvalidArgumentException("contracted dimensions must match");
        }

        int numberOfBatchDimensions = Math.max(tensor1Rank, tensor2Rank) - 2;
        int[] resultShape = new int[numberOfBatchDimensions + 2];
        for (int batchDimension = 0;
                batchDimension < numberOfBatchDimensions;
                batchDimension++) {
            int tensor1BatchDimension =
                    batchDimension - (numberOfBatchDimensions - tensor1Rank + 2);
            int tensor2BatchDimension =
                    batchDimension - (numberOfBatchDimensions - tensor2Rank + 2);
            int tensor1BatchSize = tensor1BatchDimension < 0
                    ? 1
                    : tensor1Matrix.size(tensor1BatchDimension);
            int tensor2BatchSize = tensor2BatchDimension < 0
                    ? 1
                    : tensor2Matrix.size(tensor2BatchDimension);
            if (tensor1BatchSize != tensor2BatchSize
                    && tensor1BatchSize != 1
                    && tensor2BatchSize != 1) {
                throw new InvalidArgumentException("batch shapes cannot broadcast");
            }
            resultShape[batchDimension] = Math.max(tensor1BatchSize, tensor2BatchSize);
        }
        resultShape[numberOfBatchDimensions] = numberOfRows;
        resultShape[numberOfBatchDimensions + 1] = numberOfColumns;

        JTensor<T> result = new JTensor<>(
                tensor1.getType(),
                resultShape,
                resultIndices -> matrixProductItem(
                        tensor1Matrix,
                        tensor2Matrix,
                        numericType,
                        resultIndices,
                        numberOfBatchDimensions,
                        contractedSize));

        if (tensor2IsVector) {
            result = result.squeeze(result.numberOfDimensions() - 1);
        }
        if (tensor1IsVector && result.numberOfDimensions() > 1) {
            result = result.squeeze(numberOfBatchDimensions);
        }
        return result;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static <T> T matrixProductItem(
            JTensor<T> tensor1,
            JTensor<T> tensor2,
            Class<? extends Number> numericType,
            int[] resultIndices,
            int numberOfBatchDimensions,
            int contractedSize) {
        int tensor1Rank = tensor1.numberOfDimensions();
        int tensor2Rank = tensor2.numberOfDimensions();
        int[] tensor1Indices = new int[tensor1Rank];
        int[] tensor2Indices = new int[tensor2Rank];

        for (int dimension = 0; dimension < tensor1Rank - 2; dimension++) {
            int resultDimension = numberOfBatchDimensions - tensor1Rank + 2 + dimension;
            tensor1Indices[dimension] =
                    tensor1.size(dimension) == 1 ? 0 : resultIndices[resultDimension];
        }
        for (int dimension = 0; dimension < tensor2Rank - 2; dimension++) {
            int resultDimension = numberOfBatchDimensions - tensor2Rank + 2 + dimension;
            tensor2Indices[dimension] =
                    tensor2.size(dimension) == 1 ? 0 : resultIndices[resultDimension];
        }

        tensor1Indices[tensor1Rank - 2] = resultIndices[numberOfBatchDimensions];
        tensor2Indices[tensor2Rank - 1] = resultIndices[numberOfBatchDimensions + 1];

        Number sum = NumberHelper.zero(numericType);
        for (int contractedIndex = 0; contractedIndex < contractedSize; contractedIndex++) {
            tensor1Indices[tensor1Rank - 1] = contractedIndex;
            tensor2Indices[tensor2Rank - 2] = contractedIndex;
            Number product = NumberHelper.multiply(
                    (Class) numericType,
                    (Number) tensor1.getItem(tensor1Indices),
                    (Number) tensor2.getItem(tensor2Indices));
            sum = NumberHelper.add((Class) numericType, sum, product);
        }
        return (T) sum;
    }

    /**
     * Copies an offset diagonal while retaining unselected dimensions.
     */
    static <T> JTensor<T> diagonal(
            JTensor<T> tensor,
            int offset,
            int dimension1,
            int dimension2) {
        int rank = tensor.numberOfDimensions();
        int normalizedDimension1 = TensorDimensions.normalizeDimension(dimension1, rank);
        int normalizedDimension2 = TensorDimensions.normalizeDimension(dimension2, rank);
        if (normalizedDimension1 == normalizedDimension2) {
            throw new InvalidArgumentException("diagonal axes must differ");
        }

        long dimension1Start = Math.max(0L, -(long) offset);
        long dimension2Start = Math.max(0L, (long) offset);
        int diagonalSize = (int) Math.max(
                0L,
                Math.min(
                        tensor.size(normalizedDimension1) - dimension1Start,
                        tensor.size(normalizedDimension2) - dimension2Start));
        if (diagonalSize == 0) {
            return JTensor.empty(tensor.getType());
        }

        int[] resultShape = new int[rank - 1];
        int resultDimension = 0;
        for (int dimension = 0; dimension < rank; dimension++) {
            if (dimension != normalizedDimension1 && dimension != normalizedDimension2) {
                resultShape[resultDimension++] = tensor.size(dimension);
            }
        }
        resultShape[resultDimension] = diagonalSize;

        return new JTensor<>(tensor.getType(), resultShape, resultIndices -> {
            int[] sourceIndices = new int[rank];
            int currentResultDimension = 0;
            for (int dimension = 0; dimension < rank; dimension++) {
                if (dimension != normalizedDimension1 && dimension != normalizedDimension2) {
                    sourceIndices[dimension] = resultIndices[currentResultDimension++];
                }
            }
            sourceIndices[normalizedDimension1] =
                    (int) dimension1Start + resultIndices[currentResultDimension];
            sourceIndices[normalizedDimension2] =
                    (int) dimension2Start + resultIndices[currentResultDimension];
            return tensor.getItem(sourceIndices);
        });
    }

    /**
     * Sums an offset diagonal while retaining unselected dimensions.
     */
    static <T> JTensor<T> trace(
            JTensor<T> tensor,
            int offset,
            int dimension1,
            int dimension2) {
        Class<? extends Number> numericType = validateNumericTensor(tensor);
        JTensor<T> diagonal = diagonal(tensor, offset, dimension1, dimension2);
        if (diagonal.size() != 0) {
            return diagonal.sum(-1);
        }

        int normalizedDimension1 =
                TensorDimensions.normalizeDimension(dimension1, tensor.numberOfDimensions());
        int normalizedDimension2 =
                TensorDimensions.normalizeDimension(dimension2, tensor.numberOfDimensions());
        int[] resultShape =
                new int[Math.max(1, tensor.numberOfDimensions() - 2)];
        Arrays.fill(resultShape, 1);
        int resultDimension = 0;
        for (int dimension = 0;
                dimension < tensor.numberOfDimensions();
                dimension++) {
            if (dimension != normalizedDimension1 && dimension != normalizedDimension2) {
                resultShape[resultDimension++] = tensor.size(dimension);
            }
        }
        return JTensor.repeat(
                tensor.getType(),
                resultShape,
                castNumber(numericType, NumberHelper.zero(numericType)));
    }

    /**
     * Computes stable Euclidean norms over selected dimensions.
     */
    static JTensor<Double> norm(
            JTensor<?> tensor,
            boolean keepDimensions,
            int... dimensions) {
        validateNumericTensor(tensor);
        ReductionPlan reductionPlan = new ReductionPlan(tensor, dimensions);
        if (tensor.size() == 0) {
            return JTensor.singleValue(0.0);
        }

        JTensor<?> packedTensor = reductionPlan.pack(tensor);
        JTensor<Double> result = new JTensor<>(
                Double.class,
                new int[]{reductionPlan.numberOfGroups},
                indices -> {
                    double norm = 0;
                    for (int groupIndex = 0;
                            groupIndex < reductionPlan.groupSize;
                            groupIndex++) {
                        Number value = (Number) packedTensor.getItem(
                                new int[]{indices[0], groupIndex});
                        norm = Math.hypot(norm, value.doubleValue());
                    }
                    return norm;
                });
        return result.reshape(reductionPlan.getResultShape(keepDimensions));
    }

    /**
     * Stably sorts selected subspaces and restores their original dimension order.
     */
    static <T> JTensor<T> sort(JTensor<T> tensor, int... dimensions) {
        return sort(tensor, false, dimensions);
    }

    static <T> JTensor<Integer> argsort(JTensor<T> tensor, int... dimensions) {
        return sort(tensor, true, dimensions);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static <T, R> JTensor<R> sort(
            JTensor<T> tensor,
            boolean returnIndices,
            int... dimensions) {
        Class<? extends Number> numericType = validateNumericTensor(tensor);
        ReductionPlan reductionPlan = new ReductionPlan(tensor, dimensions);
        if (tensor.size() == 0) {
            return (JTensor<R>) (returnIndices
                    ? JTensor.empty(Integer.class)
                    : tensor.copy());
        }

        JTensor<T> packedTensor = reductionPlan.pack(tensor);
        Class<?> resultType = returnIndices ? Integer.class : tensor.getType();
        JTensor result = new JTensor(
                resultType,
                new int[]{reductionPlan.numberOfGroups, reductionPlan.groupSize});

        for (int group = 0; group < reductionPlan.numberOfGroups; group++) {
            final int currentGroup = group;
            Integer[] sortedIndices = new Integer[reductionPlan.groupSize];
            for (int groupIndex = 0; groupIndex < sortedIndices.length; groupIndex++) {
                sortedIndices[groupIndex] = groupIndex;
            }
            Arrays.sort(
                    sortedIndices,
                    (index1, index2) -> NumberHelper.compare(
                            (Class) numericType,
                            (Number) packedTensor.getItem(
                                    new int[]{currentGroup, index1}),
                            (Number) packedTensor.getItem(
                                    new int[]{currentGroup, index2})));
            for (int groupIndex = 0; groupIndex < sortedIndices.length; groupIndex++) {
                Object value = returnIndices
                        ? sortedIndices[groupIndex]
                        : packedTensor.getItem(
                                new int[]{group, sortedIndices[groupIndex]});
                result.setItem(new int[]{group, groupIndex}, value);
            }
        }

        if (dimensions.length == 0) {
            return result.flatten();
        }
        int[] packedShape = new int[reductionPlan.permutation.length];
        for (int dimension = 0; dimension < packedShape.length; dimension++) {
            packedShape[dimension] = tensor.size(reductionPlan.permutation[dimension]);
        }
        return result
                .reshape(packedShape)
                .permute(reductionPlan.inversePermutation)
                .copy();
    }

    /**
     * Deduplicates elements in logical order.
     */
    static <T> JTensor<T> unique(JTensor<T> tensor) {
        LinkedHashSet<T> uniqueValues = new LinkedHashSet<>(tensor.toList());
        return fromList(tensor.getType(), new ArrayList<>(uniqueValues));
    }

    /**
     * Deduplicates complete slices along one dimension.
     */
    static <T> JTensor<T> unique(JTensor<T> tensor, int dimension) {
        int normalizedDimension = TensorDimensions.normalizeDimension(
                dimension,
                tensor.numberOfDimensions());
        List<Integer> uniqueSliceIndices = new ArrayList<>();
        Set<List<T>> uniqueSlices = new LinkedHashSet<>();

        for (int sliceIndex = 0;
                sliceIndex < tensor.size(normalizedDimension);
                sliceIndex++) {
            List<T> slice = take(
                    tensor,
                    new int[]{sliceIndex},
                    normalizedDimension).toList();
            if (uniqueSlices.add(slice)) {
                uniqueSliceIndices.add(sliceIndex);
            }
        }

        int[] selectedIndices = uniqueSliceIndices
                .stream()
                .mapToInt(Integer::intValue)
                .toArray();
        return take(tensor, selectedIndices, normalizedDimension);
    }

    /**
     * Creates an integer range including start and excluding stop.
     */
    static JTensor<Integer> arange(int start, int stop, int step) {
        if (step == 0) {
            throw new InvalidArgumentException("step must not be zero");
        }
        long distance = step > 0
                ? (long) stop - start
                : (long) start - stop;
        if (distance <= 0) {
            return JTensor.empty(Integer.class);
        }

        long numberOfValues = (distance - 1) / Math.abs((long) step) + 1;
        if (numberOfValues > Integer.MAX_VALUE) {
            throw new InvalidArgumentException("range is too large");
        }
        return new JTensor<>(
                Integer.class,
                new int[]{(int) numberOfValues},
                indices -> (int) (start + (long) indices[0] * step));
    }

    /**
     * Creates a Double range including start and excluding stop.
     */
    static JTensor<Double> arange(double start, double stop, double step) {
        if (!Double.isFinite(start)
                || !Double.isFinite(stop)
                || !Double.isFinite(step)
                || step == 0) {
            throw new InvalidArgumentException(
                    "range requires finite bounds and nonzero finite step");
        }
        double numberOfValues = Math.ceil((stop - start) / step);
        if (numberOfValues <= 0) {
            return JTensor.empty(Double.class);
        }
        if (numberOfValues > Integer.MAX_VALUE) {
            throw new InvalidArgumentException("range is too large");
        }
        return new JTensor<>(
                Double.class,
                new int[]{(int) numberOfValues},
                indices -> start + indices[0] * step);
    }

    /**
     * Creates evenly spaced samples with optional endpoint inclusion.
     */
    static JTensor<Double> linspace(
            double start,
            double stop,
            int count,
            boolean endpoint) {
        if (count < 0 || !Double.isFinite(start) || !Double.isFinite(stop)) {
            throw new InvalidArgumentException(
                    "linspace requires finite bounds and nonnegative count");
        }
        if (count == 0) {
            return JTensor.empty(Double.class);
        }

        return new JTensor<>(Double.class, new int[]{count}, indices -> {
            if (indices[0] == 0) {
                return start;
            }
            if (endpoint && indices[0] == count - 1) {
                return stop;
            }
            double stopWeight = (double) indices[0] / (endpoint ? count - 1 : count);
            return (1 - stopWeight) * start + stopWeight * stop;
        });
    }
}
