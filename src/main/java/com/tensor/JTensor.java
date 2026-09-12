package com.tensor;


import java.lang.reflect.Array;
import java.nio.*;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;

import static com.tensor.DataType.*;

/*
  @author talalalrawajfeh@gmail.com
 */


/**
 * <p>
 * A Tensor is a multi-dimensional array with arrays of equal lengths (dimension size or shape) in each dimension.
 * Tensors are equipped with many useful operations that could either be performed on the tensor itself or more than one
 * tensor. Many implementations of the same concept already exist in many different languages such as the well-known
 * NumPy (in Python) which is the main inspiration for this implementation. Tensors are represented in memory as a
 * single-dimensional array which makes them more efficient than language-specific multi-dimensional arrays. Moreover,
 * Tensors can map a multi-dimensional index (array of indices) to a flat or single-dimensional index that points to an
 * item within the single-dimensional array. In this implementation, strides and special mappings are used to map a
 * multi-dimensional index to a flat index. A stride is a fixed-size step such that each dimension of the tensor has its
 * own stride. The stride of a dimension is calculated by taking the product of all the dimension sizes (shapes)
 * proceeding it. For example, if the dimension shapes of a tensor are (2, 3, 4) which is also called the shape of the
 * tensor, then the first dimension has a stride of 3 * 4 = 12 and the second dimension has a stride of 4 and the last
 * dimension has a stride of 1. In general, given any multi-dimensional index represented as an array of indices, each
 * index within that array is multiplied with the stride at that dimension and then the results are summed together to
 * obtain a flat index. As with the previous of the tensor of shape (2, 3, 4), to obtain the item at the index
 * (1, 1, 1), we compute 1 * 12 + 1 * 4 + 1 = 17, then take the item in the flat array at the index 17. This is called
 * the row-major order of the indices since the strides get smaller from left to right. In this way, each dimension is
 * represented as non-overlapping fixed-size blocks of contiguous items of the array. Strides are always strictly
 * positive in this implementation unlike some other implementations where strides could be negative or zero. In some
 * cases where a more sophisticated mapping of the indices is required, we took a functional approach to meet them all
 * of which utilizes the generality and composability of mappings and made the implementation of these operations much
 * easier. When some operations are performed on a tensor, the data is not always copied which introduces the concept of
 * a view of a tensor, which is a shallow copy of the data in contrast to a deep copy of the data, to reduce memory
 * overhead. This implementation contains all essential operations such tensor manipulation operations, tensors
 * broadcasting, unary and binary operations, functional operations, etc. It could also be easily extended and
 * customized according to specific needs.
 * </p>
 */
public class JTensor<T> {
    private static final String NO_NEXT_ELEMENT = "indices iterator doesn't have a next element";
    private static final String INDEX_OUT_OF_BOUNDS = "index out of bounds";
    public static final String ARRAYS_DO_NOT_ALL_HAVE_THE_SAME_LENGTH = "arrays do not all have the same length in each dimension";

    private final Class<T> type;
    private final int[] shape;
    private final T[] data;
    private final int size;
    private final int[] strides;
    // indicesTable is used to map indices when applying certain transformations to a tensor
    private final int[] indicesTable;
    private final boolean isView;

    private JTensor(Class<T> type,
                    int[] shape,
                    T[] data,
                    int size,
                    int[] strides,
                    int[] indicesTable,
                    boolean isView) {
        this.type = type;
        this.shape = Arrays.copyOf(shape, shape.length);
        this.data = data;
        this.size = size;
        this.strides = Arrays.copyOf(strides, strides.length);
        this.indicesTable = indicesTable;
        this.isView = isView;
    }

    /**
     * Constructs a tensor with the given shape filled with default values of the given type.
     *
     * @param type  determines the type of the values of the tensor.
     * @param shape determines the shapes of the dimensions of the tensor where each shape should be a (strictly) positive integer.
     * @throws InvalidShapeException when a zero or a negative number is found in the shape array.
     */
    public JTensor(Class<T> type,
                   int[] shape) {
        validateTensorType(type);
        size = initializeSize(shape);
        this.shape = Arrays.copyOf(shape, shape.length);
        this.type = type;
        strides = initializeStrides(shape);
        data = (T[]) Array.newInstance(type, size);
        this.indicesTable = defaultIndicesTable(size);
        isView = false;
    }

    /**
     * Constructs a tensor with the given shape from a 1-dimensional array in row-major order.
     * The array values are copied into storage whose runtime component type matches {@code type}.
     *
     * @param type  determines the type of the values of the tensor.
     * @param shape determines the shapes of the dimensions of the tensor where each shape should be a (strictly) positive integer.
     * @param array the 1-dimensional array that holds the values of the tensor.
     * @throws DataSizeMismatchException when the length of the given array does not equal the calculated size of the tensor.
     */
    public JTensor(Class<T> type,
                   int[] shape,
                   T[] array) {
        validateTensorType(type);
        if (array == null) {
            throw new InvalidArgumentException("data array must not be null");
        }
        size = initializeSize(shape);
        this.shape = Arrays.copyOf(shape, shape.length);
        this.type = type;
        if (array.length != size) {
            throw new DataSizeMismatchException("given array size: " + array.length + ", but should be " + size);
        }
        for (T value : array) {
            validateValueType(type, value);
        }
        strides = initializeStrides(shape);
        this.data = (T[]) Array.newInstance(type, size);
        System.arraycopy(array, 0, this.data, 0, size);
        this.indicesTable = defaultIndicesTable(size);
        isView = false;
    }

    /**
     * Constructs a tensor and initializes its values by a given initializer
     *
     * @param type        determines the type of the values of the tensor.
     * @param shape       determines the shapes of the dimensions of the tensor where each shape should be a (strictly) positive integer.
     * @param initializer a lambda that returns a value for every index of the tensor.
     * @throws DataSizeMismatchException when the length of the given array does not equal the calculated size of the tensor.
     */
    public JTensor(Class<T> type, int[] shape, Function<int[], T> initializer) {
        validateTensorType(type);
        if (initializer == null) {
            throw new InvalidArgumentException("initializer must not be null");
        }
        size = initializeSize(shape);
        this.shape = Arrays.copyOf(shape, shape.length);
        this.type = type;
        strides = initializeStrides(shape);
        data = (T[]) Array.newInstance(type, size);
        this.indicesTable = defaultIndicesTable(size);
        isView = false;
        final Iterator<int[]> indicesIterator = createIndicesIterator(this.shape);
        while (indicesIterator.hasNext()) {
            final int[] indices = indicesIterator.next();
            final int destinationIndex = dataIndex(indices);
            T value = initializer.apply(indices);
            validateValueType(type, value);
            data[destinationIndex] = value;
        }
    }

    /**
     * Constructs a new tensor from an old one and the data of the old tensor is always
     * copied even if it is a view.
     *
     * @param tensor the tensor to be copied.
     */
    public JTensor(JTensor<T> tensor) {
        if (tensor == null) {
            throw new InvalidArgumentException("tensor must not be null");
        }
        if (tensor.isView()) {
            this.data = (T[]) Array.newInstance(tensor.type, tensor.size);
            final Iterator<int[]> indicesIterator = tensor.indicesIterator();
            int i = 0;
            while (indicesIterator.hasNext()) {
                this.data[i] = tensor.getItem(indicesIterator.next());
                i++;
            }
        } else {
            this.data = Arrays.copyOf(tensor.data, tensor.getSize());
        }
        this.shape = Arrays.copyOf(tensor.shape, tensor.shape.length);
        this.strides = initializeStrides(this.shape);
        this.type = tensor.type;
        this.isView = false;
        this.indicesTable = defaultIndicesTable(tensor.size);
        this.size = tensor.size;
    }

    /*
     * @param type determines the type of the values of the tensor.
     * @return a tensor with an empty shape.
     */
    public static <T> JTensor<T> empty(Class<T> type) {
        return new JTensor<>(type, new int[]{});
    }

    /**
     * @param shape determines the shapes of the dimensions of the tensor where each shape should be a (strictly) positive integer.
     * @param value the value to be repeated
     * @return a tensor of the given shape and its data consists of the repeated value.
     * The runtime class of {@code value} becomes the tensor type; use the overload
     * accepting {@link Class} when a wider declared type or a null value is required.
     */
    public static <T> JTensor<T> repeat(int[] shape, T value) {
        if (value == null) {
            throw new InvalidArgumentException(
                    "repeated value must not be null when no explicit type is provided");
        }
        return repeat((Class<T>) value.getClass(), shape, value);
    }

    /**
     * Returns a tensor filled with a repeated value using an explicit component type.
     * This overload can represent null values and avoids generic type inference from
     * the runtime class of {@code value}.
     */
    public static <T> JTensor<T> repeat(Class<T> type, int[] shape, T value) {
        validateValueType(type, value);
        return new JTensor<>(type, shape, a -> value);
    }

    /**
     * @param value
     * @return a tensor of shape (1) with only the given value.
     */
    public static <T> JTensor<T> singleValue(T value) {
        return repeat(new int[]{1}, value);
    }

    public static <T> JTensor<T> singleValue(Class<T> type, T value) {
        return repeat(type, new int[]{1}, value);
    }

    /**
     * @param type  determines the type of the values of the tensor but must be an instance of {@link Number}.
     * @param shape determines the shapes of the dimensions of the tensor where each shape should be a (strictly) positive integer.
     * @return a tensor of the given shape and type and its data consists of zeros.
     */
    public static <T extends Number> JTensor<T> zeros(Class<T> type, int[] shape) {
        return repeat(shape, NumberHelper.zero(type));
    }

    /**
     * @param type  determines the type of the values of the tensor but must be an instance of {@link Number}.
     * @param shape determines the shapes of the dimensions of the tensor where each shape should be a (strictly) positive integer.
     * @return a tensor of the given shape and type and its data consists of ones.
     */
    public static <T extends Number> JTensor<T> ones(Class<T> type, int[] shape) {
        return repeat(shape, NumberHelper.one(type));
    }

    /**
     * @param type           determines the type of the values of the tensor but must be an instance of {@link Number}.
     * @param dimensionShape the shape of each dimension of the square matrix.
     * @return an identity matrix.
     */
    public static <T extends Number> JTensor<T> identity(Class<T> type, int dimensionShape) {
        JTensor<T> identity = zeros(type, new int[]{dimensionShape, dimensionShape});
        for (int i = 0; i < dimensionShape; i++) {
            identity.setItem(new int[]{i, i}, NumberHelper.one(type));
        }
        return identity;
    }

    /**
     * @param type  determines the type of the values of the tensor
     * @param array a one-dimensional array to construct the tensor from
     * @return a tensor with the same shape and data as the one-dimensional array
     * The values are copied; an empty input produces the rank-zero empty tensor.
     */
    public static <T> JTensor<T> from1DArray(Class<T> type, T[] array) {
        if (array == null) {
            throw new InvalidArgumentException("array must not be null");
        }
        if (array.length == 0) {
            return empty(type);
        }
        return new JTensor<>(type, new int[]{array.length}, array);
    }

    /**
     * @param type  determines the type of the values of the tensor
     * @param array a two-dimensional array to construct the tensor from
     * @return a tensor with the same shape and data as the two-dimensional array
     * @throws InvalidArgumentException if the arrays in a dimension are not of the same length
     */
    public static <T> JTensor<T> from2DArray(Class<T> type, T[][] array) {
        if (array == null || array.length == 0 || array[0] == null || array[0].length == 0) {
            throw new InvalidArgumentException("array dimensions must be non-empty");
        }
        int secondDimension = array[0].length;
        final JTensor<T> result = new JTensor<>(type, new int[]{array.length, secondDimension});

        for (int i = 0; i < array.length; i++) {
            if (array[i] == null || array[i].length != secondDimension) {
                throw new InvalidArgumentException(ARRAYS_DO_NOT_ALL_HAVE_THE_SAME_LENGTH);
            }
            for (int j = 0; j < secondDimension; j++) {
                result.setItem(new int[]{i, j}, array[i][j]);
            }
        }
        return result;
    }

    /**
     * @param type  determines the type of the values of the tensor
     * @param array a three-dimensional array to construct the tensor from
     * @return a tensor with the same shape and data as the three-dimensional array
     * @throws InvalidArgumentException if the arrays in a dimension are not of the same length
     */
    public static <T> JTensor<T> from3DArray(Class<T> type, T[][][] array) {
        if (array == null || array.length == 0
                || array[0] == null || array[0].length == 0
                || array[0][0] == null || array[0][0].length == 0) {
            throw new InvalidArgumentException("array dimensions must be non-empty");
        }
        int secondDimension = array[0].length;
        int thirdDimension = array[0][0].length;
        final JTensor<T> result = new JTensor<>(type, new int[]{
                array.length,
                secondDimension,
                thirdDimension});

        for (int i = 0; i < array.length; i++) {
            if (array[i] == null || array[i].length != secondDimension) {
                throw new InvalidArgumentException(ARRAYS_DO_NOT_ALL_HAVE_THE_SAME_LENGTH);
            }
            for (int j = 0; j < secondDimension; j++) {
                if (array[i][j] == null || array[i][j].length != thirdDimension) {
                    throw new InvalidArgumentException(ARRAYS_DO_NOT_ALL_HAVE_THE_SAME_LENGTH);
                }
                for (int k = 0; k < thirdDimension; k++) {
                    result.setItem(new int[]{i, j, k}, array[i][j][k]);
                }
            }
        }
        return result;
    }

    /**
     * @param type  determines the type of the values of the tensor
     * @param array a four-dimensional array to construct the tensor from
     * @return a tensor with the same shape and data as the four-dimensional array
     * @throws InvalidArgumentException if the arrays in a dimension are not of the same length
     */
    public static <T> JTensor<T> from4DArray(Class<T> type, T[][][][] array) {
        if (array == null || array.length == 0
                || array[0] == null || array[0].length == 0
                || array[0][0] == null || array[0][0].length == 0
                || array[0][0][0] == null || array[0][0][0].length == 0) {
            throw new InvalidArgumentException("array dimensions must be non-empty");
        }
        int secondDimension = array[0].length;
        int thirdDimension = array[0][0].length;
        int fourthDimension = array[0][0][0].length;
        final JTensor<T> result = new JTensor<>(type, new int[]{
                array.length,
                secondDimension,
                thirdDimension,
                fourthDimension});

        for (int i = 0; i < array.length; i++) {
            if (array[i] == null || array[i].length != secondDimension) {
                throw new InvalidArgumentException(ARRAYS_DO_NOT_ALL_HAVE_THE_SAME_LENGTH);
            }
            for (int j = 0; j < secondDimension; j++) {
                if (array[i][j] == null || array[i][j].length != thirdDimension) {
                    throw new InvalidArgumentException(ARRAYS_DO_NOT_ALL_HAVE_THE_SAME_LENGTH);
                }
                for (int k = 0; k < thirdDimension; k++) {
                    if (array[i][j][k] == null || array[i][j][k].length != fourthDimension) {
                        throw new InvalidArgumentException(ARRAYS_DO_NOT_ALL_HAVE_THE_SAME_LENGTH);
                    }
                    for (int l = 0; l < fourthDimension; l++) {
                        result.setItem(new int[]{i, j, k, l}, array[i][j][k][l]);
                    }
                }
            }
        }
        return result;
    }

    private static int[] initializeStrides(int[] shape) {
        final int[] strides = new int[shape.length];

        int currentStride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = currentStride;
            try {
                currentStride = Math.multiplyExact(currentStride, shape[i]);
            } catch (ArithmeticException exception) {
                throw new InvalidShapeException("tensor shape is too large", exception);
            }
        }

        return strides;
    }

    private static int initializeSize(int[] shape) {
        if (shape == null) {
            throw new InvalidShapeException("shape must not be null");
        }
        if (shape.length == 0) {
            return 0;
        }

        int size = 1;
        for (int i = 0; i < shape.length; i++) {
            int dimension = shape[i];
            if (dimension <= 0) {
                throw new InvalidShapeException("given non-positive number as a dimension: " + dimension);
            }
            try {
                size = Math.multiplyExact(size, dimension);
            } catch (ArithmeticException exception) {
                throw new InvalidShapeException("tensor shape is too large", exception);
            }
        }

        return size;
    }

    /**
     * @return an iterator of all possible logical indices in row-major order.
     * For views, this order may differ from the backing data array order.
     * @throws NoNextElementException if next is called and no next item is present.
     */
    public Iterator<int[]> indicesIterator() {
        return createIndicesIterator(shape);
    }

    private static Iterator<int[]> createIndicesIterator(int[] iteratorShape) {
        return new Iterator<>() {
            private final int[] currentIndices = new int[iteratorShape.length];
            private boolean isFirstTime = true;
            private boolean isDone = false;

            @Override
            public boolean hasNext() {
                if (isDone) {
                    return false;
                }
                for (int i = currentIndices.length - 1; i >= 0; i--) {
                    if (currentIndices[i] < iteratorShape[i] - 1
                            || (isFirstTime && currentIndices[i] == iteratorShape[i] - 1)) {
                        return true;
                    }
                }
                isDone = true;
                return false;
            }

            @Override
            public int[] next() {
                if (!hasNext()) {
                    throw new NoNextElementException(NO_NEXT_ELEMENT);
                }
                if (isFirstTime) {
                    isFirstTime = false;
                    return Arrays.copyOf(currentIndices, currentIndices.length);
                }
                for (int i = currentIndices.length - 1; i >= 0; i--) {
                    currentIndices[i] += 1;
                    if (currentIndices[i] >= iteratorShape[i]) {
                        currentIndices[i] = 0;
                    } else {
                        return Arrays.copyOf(currentIndices, currentIndices.length);
                    }
                }
                isDone = true;
                throw new NoNextElementException(NO_NEXT_ELEMENT);
            }
        };
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        JTensor<?> tensor = (JTensor<?>) o;
        if (!Objects.equals(type, tensor.type) || !Arrays.equals(shape, tensor.shape)) {
            return false;
        }

        Iterator<int[]> iterator1 = indicesIterator();
        Iterator<int[]> iterator2 = tensor.indicesIterator();

        while (iterator1.hasNext()) {
            if (!Objects.equals(getItem(iterator1.next()), tensor.getItem(iterator2.next()))) {
                return false;
            }
        }

        return true;
    }

    /**
     * @param indices an array containing an index for each dimension of the tensor.
     * @return the item stored in the tensor at the given indices.
     * @throws InvalidArgumentException if the number of indices does not match the tensor rank.
     * @throws IndexOutOfBoundsException if one of the indices is not within the bound of its dimension.
     */
    public T getItem(int[] indices) {
        return this.data[dataIndex(indices)];
    }

    /**
     * @param indices an array containing an index for each dimension of the tensor.
     * @param number  the item to store in the tensor at the given indices.
     * @throws InvalidArgumentException if the number of indices does not match the tensor rank.
     */
    public void setItem(int[] indices, T number) {
        validateValueType(type, number);
        this.data[dataIndex(indices)] = number;
    }

    private static void validateValueType(Class<?> type, Object value) {
        validateTensorType(type);
        if (value != null && !type.isInstance(value)) {
            throw new InvalidTypeException("value type " + value.getClass().getName()
                    + " is not compatible with tensor type " + type.getName());
        }
    }

    private static void validateTensorType(Class<?> type) {
        if (type == null) {
            throw new InvalidArgumentException("type must not be null");
        }
        if (type.isPrimitive()) {
            throw new InvalidTypeException(
                    "primitive class tokens are not supported; use the corresponding wrapper class");
        }
    }

    private static <V> JTensor<V> requireTensor(JTensor<V> tensor) {
        if (tensor == null) {
            throw new InvalidArgumentException("tensor must not be null");
        }
        return tensor;
    }

    private void requireNonNullItems(String operation) {
        Iterator<int[]> iterator = indicesIterator();
        while (iterator.hasNext()) {
            if (getItem(iterator.next()) == null) {
                throw new InvalidArgumentException(
                        operation + " does not support null tensor values");
            }
        }
    }

    private int dataIndex(int[] indices) {
        if (indices == null) {
            throw new InvalidArgumentException("indices must not be null");
        }
        if (indices.length != this.shape.length) {
            throw new InvalidArgumentException("invalid number of indices: expected "
                    + this.shape.length + " but got " + indices.length);
        }
        if (this.shape.length == 0) {
            throw new IndexOutOfBoundsException(INDEX_OUT_OF_BOUNDS);
        }
        int index = 0;
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException(INDEX_OUT_OF_BOUNDS);
            }
            index += strides[i] * indices[i];
        }
        return this.indicesTable[index];
    }

    private void validateDimension(int dimension) {
        if (dimension < 0 || dimension >= shape.length) {
            throw new InvalidArgumentException("invalid dimension " + dimension
                    + " for tensor rank " + shape.length);
        }
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(shape);
        final Iterator<int[]> indicesIterator = this.indicesIterator();
        while (indicesIterator.hasNext()) {
            result = 31 * result + Objects.hashCode(getItem(indicesIterator.next()));
        }
        return result;
    }

    public Class<T> getType() {
        return type;
    }

    public int[] getShape() {
        return Arrays.copyOf(shape, shape.length);
    }

    /**
     * Returns the backing data array. For a view, this is the shared backing array
     * and is not necessarily in the view's logical element order.
     */
    public T[] getData() {
        return data;
    }

    public int getSize() {
        return size;
    }

    public int[] getStrides() {
        return Arrays.copyOf(strides, strides.length);
    }

    public boolean isView() {
        return isView;
    }

    /**
     * @param shape an array of dimensions of the new tensor.
     * @return a new tensor with the same data but with the given shape.
     * Does a shallow copy of the tensor and the new tensor becomes a view
     * of the old tensor except when the tensor is already a view. Existing
     * views are materialized in logical row-major order before reshaping.
     * @throws InvalidArgumentException if the size of the given shape is not equal to the size of the tensor.
     */
    public JTensor<T> reshape(int[] shape) {
        final int newSize = initializeSize(shape);
        if (newSize != this.getSize()) {
            throw new InvalidArgumentException("given size is not equal to the original size");
        }

        T[] data = this.data;
        boolean isView = true;

        if (this.isView) {
            data = (T[]) Array.newInstance(this.type, this.size);

            final Iterator<int[]> indicesIterator = indicesIterator();
            int i = 0;
            while (indicesIterator.hasNext()) {
                data[i] = this.data[dataIndex(indicesIterator.next())];
                i++;
            }

            isView = false;
        }

        return new TensorBuilder<T>()
                .setType(this.type)
                .setShape(shape)
                .setData(data)
                .setSize(newSize)
                .setStrides(initializeStrides(shape))
                .setIndicesTable(defaultIndicesTable(newSize))
                .setView(isView)
                .build();
    }

    /**
     * @return reshapes the tensor into a flat array which is equivalent to the data array.
     * May return a view of the tensor according to the rules of reshape.
     */
    public JTensor<T> ravel() {
        if (size == 0) {
            return reshape(new int[]{});
        }
        return this.reshape(new int[]{this.size});
    }

    /**
     * @return same as ravel() but always returns a deep copy of the tensor.
     */
    public JTensor<T> flatten() {
        if (size == 0) {
            return new JTensor<>(this);
        }
        T[] data = (T[]) Array.newInstance(this.type, this.size);

        final Iterator<int[]> indicesIterator = indicesIterator();
        int i = 0;
        while (indicesIterator.hasNext()) {
            data[i] = this.data[dataIndex(indicesIterator.next())];
            i++;
        }

        return new JTensor<>(type, new int[]{this.size}, data);
    }

    /**
     * @param dimension a dimension of the tensor.
     * @return a new tensor with the same shape but with the given dimension removed.
     * the shape of the given dimension must be 1 or otherwise an exception
     * is thrown.
     * @throws InvalidArgumentException if the given dimension is not of shape 1.
     */
    public JTensor<T> squeeze(int dimension) {
        validateDimension(dimension);
        if (shape[dimension] != 1) {
            throw new InvalidArgumentException("can only squeeze a dimension when its shape is 1");
        }
        if (shape.length == 1) {
            throw new InvalidArgumentException(
                    "cannot squeeze the only dimension because scalar tensors are not supported");
        }

        int[] squeezedShape = new int[shape.length - 1];
        int j = 0;
        for (int i = 0; i < shape.length; i++) {
            if (i == dimension) {
                continue;
            }
            squeezedShape[j] = shape[i];
            j++;
        }

        return reshape(squeezedShape);
    }

    /**
     * @return the transpose of the tensor by reversing its dimensions. The result tensor is a view of the original.
     */
    public JTensor<T> transpose() {
        return new TensorBuilder<T>()
                .setType(type)
                .setShape(reverseArray(this.shape))
                .setSize(size)
                .setData(data)
                .setStrides(reverseArray(this.strides))
                .setIndicesTable(transposedIndicesTable())
                .setView(true)
                .build();
    }

    private int[] transposedIndicesTable() {
        int[] currentIndices = new int[shape.length];
        int[] newIndicesTable = new int[size];
        boolean isFirstTime = true;

        for (int i = 0; i < size; i++) {
            for (int j = currentIndices.length - 1; j >= 0; j--) {
                if (!isFirstTime) {
                    currentIndices[j] += 1;
                } else {
                    isFirstTime = false;
                }
                if (currentIndices[j] >= shape[j]) {
                    currentIndices[j] = 0;
                } else {
                    int flatIndex = 0;
                    for (int k = 0; k < currentIndices.length; k++) {
                        flatIndex += strides[k] * currentIndices[k];
                    }

                    int transposedFlatIndex = 0;
                    for (int k = 0; k < currentIndices.length; k++) {
                        transposedFlatIndex += strides[shape.length - k - 1] * currentIndices[shape.length - k - 1];
                    }

                    newIndicesTable[flatIndex] = this.indicesTable[transposedFlatIndex];
                    break;
                }
            }
        }

        return newIndicesTable;
    }

    private int[] reverseArray(int[] array) {
        int[] reversedShape = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            reversedShape[i] = array[array.length - 1 - i];
        }
        return reversedShape;
    }

    /**
     * @param dimension1 the first dimension.
     * @param dimension2 the second dimension.
     * @return the tensor with two dimensions swapped. The result tensor is a view of the original.
     */
    public JTensor<T> swapDimensions(int dimension1, int dimension2) {
        validateDimension(dimension1);
        validateDimension(dimension2);
        int[] shape = Arrays.copyOf(this.shape, this.shape.length);
        int temp = shape[dimension1];
        shape[dimension1] = shape[dimension2];
        shape[dimension2] = temp;

        int[] strides = Arrays.copyOf(this.strides, this.strides.length);
        temp = this.strides[dimension1];
        strides[dimension1] = this.strides[dimension2];
        strides[dimension2] = temp;

        return new TensorBuilder<T>()
                .setType(type)
                .setShape(shape)
                .setSize(size)
                .setData(data)
                .setStrides(strides)
                .setIndicesTable(swappedIndicesTable(dimension1, dimension2))
                .setView(true)
                .build();
    }

    private int[] swappedIndicesTable(int dimension1, int dimension2) {
        int[] currentIndices = new int[shape.length];
        int[] newIndicesTable = new int[size];
        boolean isFirstTime = true;

        for (int i = 0; i < size; i++) {
            for (int j = currentIndices.length - 1; j >= 0; j--) {
                if (!isFirstTime) {
                    currentIndices[j] += 1;
                } else {
                    isFirstTime = false;
                }
                if (currentIndices[j] >= shape[j]) {
                    currentIndices[j] = 0;
                } else {
                    int flatIndex = 0;
                    for (int k = 0; k < currentIndices.length; k++) {
                        flatIndex += strides[k] * currentIndices[k];
                    }

                    int swappedFlatIndex = 0;
                    for (int k = 0; k < currentIndices.length; k++) {
                        int swappedDimensionIndex;
                        int stride;

                        if (k == dimension1) {
                            swappedDimensionIndex = currentIndices[dimension2];
                            stride = strides[dimension2];
                        } else if (k == dimension2) {
                            swappedDimensionIndex = currentIndices[dimension1];
                            stride = strides[dimension1];
                        } else {
                            swappedDimensionIndex = currentIndices[k];
                            stride = strides[k];
                        }

                        swappedFlatIndex += stride * swappedDimensionIndex;
                    }

                    newIndicesTable[flatIndex] = this.indicesTable[swappedFlatIndex];
                    break;
                }
            }
        }

        return newIndicesTable;
    }

    /**
     * @param intervals array of arrays of integers each which has exactly two integers representing an interval
     *                  (starting index, ending index) and which corresponds to a dimension of the tensor. The starting
     *                  index of each interval is inclusive; however, the ending index is exclusive.
     * @return a part of the tensor for which only the items within the intervals in each dimension are taken.
     * The result tensor is a view of the original tensor.
     * @throws InvalidArgumentException if one of the intervals is invalid, i.e. one of the intervals contains a
     *                                  negative index, the starting index is greater or equal the ending index, or one of the indices exceed the shape
     *                                  of the dimension, or the number of intervals does not match the tensor rank.
     */
    public JTensor<T> slice(int[][] intervals) {
        if (intervals == null) {
            throw new InvalidArgumentException("intervals must not be null");
        }
        if (intervals.length != this.shape.length) {
            throw new InvalidArgumentException("invalid number of intervals: expected "
                    + this.shape.length + " but got " + intervals.length);
        }

        int[] shape = new int[this.shape.length];
        int[] offsets = new int[this.shape.length];

        for (int i = 0; i < this.shape.length; i++) {
            if (intervals[i] == null || intervals[i].length != 2) {
                throw new InvalidArgumentException("each interval must contain exactly two indices");
            }
            final int rightExclusiveLimit = intervals[i][1];
            final int leftInclusiveLimit = intervals[i][0];

            if (leftInclusiveLimit < 0
                    || rightExclusiveLimit > this.shape[i]
                    || rightExclusiveLimit <= leftInclusiveLimit) {
                throw new InvalidArgumentException("invalid interval");
            }

            shape[i] = rightExclusiveLimit - leftInclusiveLimit;
            offsets[i] = leftInclusiveLimit;
        }

        int size = initializeSize(shape);
        int[] strides = initializeStrides(shape);

        return new TensorBuilder<T>()
                .setType(this.type)
                .setShape(shape)
                .setData(this.data)
                .setSize(size)
                .setStrides(strides)
                .setIndicesTable(slicedIndicesTable(shape, strides, offsets, size))
                .setView(true)
                .build();
    }

    private int[] slicedIndicesTable(int[] shape, int[] strides, int[] offsets, int size) {
        int[] currentIndices = new int[shape.length];
        int[] newIndicesTable = new int[size];
        boolean isFirstTime = true;

        for (int i = 0; i < size; i++) {
            for (int j = currentIndices.length - 1; j >= 0; j--) {
                if (!isFirstTime) {
                    currentIndices[j] += 1;
                } else {
                    isFirstTime = false;
                }
                if (currentIndices[j] >= shape[j]) {
                    currentIndices[j] = 0;
                } else {
                    int flatIndex = 0;
                    for (int k = 0; k < currentIndices.length; k++) {
                        flatIndex += this.strides[k] * (currentIndices[k] + offsets[k]);
                    }

                    int slicedFlatIndex = 0;
                    for (int k = 0; k < currentIndices.length; k++) {
                        slicedFlatIndex += strides[k] * currentIndices[k];
                    }

                    newIndicesTable[slicedFlatIndex] = this.indicesTable[flatIndex];
                    break;
                }
            }
        }

        return newIndicesTable;
    }

    /**
     * @param tensor1   first tensor
     * @param tensor2   second tensor
     * @param dimension the dimension to concatenate along
     * @return A tensor that is the concatenation of the first and second tensor along the given dimension. Both tensors
     * must have the same shape except on the dimension of concatenation.
     * @throws InvalidArgumentException when the number of dimensions of the first tensor is not equal to the number of
     *                                  dimensions of the second tensor. Or the shapes of the tensors are not equal on the dimensions other than the
     *                                  concatenation dimension.
     */
    public static <A> JTensor<A> concatenate(JTensor<A> tensor1, JTensor<A> tensor2, int dimension) {
        if (tensor1 == null || tensor2 == null) {
            throw new InvalidArgumentException("tensors to concatenate must not be null");
        }
        if (tensor1.shape.length != tensor2.shape.length) {
            throw new InvalidArgumentException("tensors must have the same number of dimensions");
        }
        tensor1.validateDimension(dimension);

        int[] newShape = new int[tensor1.shape.length];
        for (int i = 0; i < tensor1.shape.length; i++) {
            if (i == dimension) {
                try {
                    newShape[i] = Math.addExact(tensor1.shape[i], tensor2.shape[i]);
                } catch (ArithmeticException exception) {
                    throw new InvalidShapeException("concatenated tensor shape is too large", exception);
                }
                continue;
            }
            if (tensor1.shape[i] != tensor2.shape[i]) {
                throw new InvalidArgumentException("tensors must have the same shape except on the given dimension");
            }
            newShape[i] = tensor1.shape[i];
        }

        JTensor<A> result = new JTensor<>(tensor1.type, newShape);
        final Iterator<int[]> indicesIterator = result.indicesIterator();
        while (indicesIterator.hasNext()) {
            final int[] indices = indicesIterator.next();
            if (indices[dimension] < tensor1.shape[dimension]) {
                result.setItem(indices, tensor1.getItem(indices));
            } else {
                final int[] tensor2Indices = Arrays.copyOf(indices, indices.length);
                tensor2Indices[dimension] -= tensor1.shape[dimension];
                result.setItem(indices, tensor2.getItem(tensor2Indices));
            }
        }

        return result;
    }

    /**
     * @param tensor    another tensor
     * @param dimension the dimension to concatenate along
     * @return A tensor that is the concatenation of the original and another tensor along the given dimension. Both tensors
     * must have the same shape except on the dimension of concatenation.
     * @throws InvalidArgumentException when the number of dimensions of the first tensor is not equal to the number of
     *                                  dimensions of the second tensor. Or the shapes of the tensors are not equal on the dimensions other than the
     *                                  concatenation dimension.
     */
    public JTensor<T> concatenate(JTensor<T> tensor, int dimension) {
        return concatenate(this, tensor, dimension);
    }

    /**
     * @param mask a boolean tensor acting as a mask to extract the specific items/arrays from the tensor
     * @return a tensor that is the result of applying the mask to the tensor. The shapes of the first n
     * dimensions of the tensor must be equal to the shapes of the mask dimensions where n is the number of dimensions
     * the mask. If the mask selects no values, the result is the rank-zero empty tensor.
     * @throws InvalidArgumentException if the corresponding dimensions of the mask and the tensor are not equal.
     */
    public JTensor<T> applyMask(JTensor<Boolean> mask) {
        if (mask == null) {
            throw new InvalidArgumentException("mask must not be null");
        }
        mask.requireNonNullItems("Mask application");
        int[] maskShape = mask.getShape();
        int[] tensorShape = getShape();

        if (maskShape.length > tensorShape.length) {
            throw new InvalidArgumentException("mask rank must not exceed tensor rank");
        }

        for (int i = 0; i < maskShape.length; i++) {
            if (maskShape[i] != tensorShape[i]) {
                throw new InvalidArgumentException("the corresponding dimensions of the mask and the tensor should be equal");
            }
        }

        int[] maskedTensorShape = new int[tensorShape.length - maskShape.length + 1];
        if (tensorShape.length - maskShape.length >= 0) {
            System.arraycopy(tensorShape,
                    maskShape.length,
                    maskedTensorShape,
                    1,
                    tensorShape.length - maskShape.length);
        }

        boolean hasAnyTrueItem = false;
        Iterator<int[]> maskIterator = mask.indicesIterator();
        while (maskIterator.hasNext()) {
            if (mask.getItem(maskIterator.next())) {
                hasAnyTrueItem = true;
                maskedTensorShape[0]++;
            }
        }

        if (!hasAnyTrueItem) {
            return new JTensor<>(getType(), new int[]{});
        }

        JTensor<T> maskedTensor = new JTensor<>(getType(), maskedTensorShape);
        Iterator<int[]> tensorIterator = indicesIterator();
        int i = 0;
        while (tensorIterator.hasNext()) {
            int[] tensorIndex = tensorIterator.next();
            int[] maskIndex = Arrays.copyOfRange(tensorIndex, 0, maskShape.length);
            if (mask.getItem(maskIndex)) {
                int[] maskedIndex = new int[maskedTensorShape.length];
                System.arraycopy(tensorIndex,
                        maskShape.length,
                        maskedIndex,
                        1,
                        maskedIndex.length - 1);
                maskedIndex[0] = i / maskedTensor.getStrides()[0];
                maskedTensor.setItem(maskedIndex, getItem(tensorIndex));
                i++;
            }
        }

        return maskedTensor;
    }

    /**
     * @param shape the shape to resize the tensor to which could be equal to or smaller than the original tensor's shape
     * @return takes only the data items of the tensor in row-major order up to the same size of the new tensor and
     * discards the data items afterwards. Shape {@code []} returns an independent rank-zero empty tensor.
     */
    public JTensor<T> resize(int[] shape) {
        final int newSize = initializeSize(shape);
        if (newSize > size) {
            throw new InvalidArgumentException("new tensor size must not exceed the original size");
        }
        if (newSize == 0) {
            return new JTensor<>(type, new int[]{});
        }
        if (newSize == size) {
            return reshape(shape);
        }
        return ravel()
                .slice(new int[][]{{0, newSize}})
                .reshape(shape);
    }

    /**
     * @param value the value to search for
     * @return true if the value is found within the data of the tensor and returns false otherwise
     */
    public boolean contains(T value) {
        final Iterator<int[]> indicesIterator = indicesIterator();
        while (indicesIterator.hasNext()) {
            final T item = getItem(indicesIterator.next());
            if (Objects.equals(value, item)) {
                return true;
            }
        }
        return false;
    }

    /**
     * @param tensor1         first tensor
     * @param tensor2         second tensor
     * @param binaryOperation a function that takes the class of the generic type and the two values and returns a new
     *                        value. The class of the generic type is needed since the Java language doesn't provide any
     *                        mechanism to check the given generic type at runtime. For example, this is needed when the
     *                        add operation is required and the numbers could be of class Integer or Double, etc. for
     *                        which the specific type must be known to perform the addition operation for that type.
     * @return a new tensor as a result of applying the binary operation on the corresponding items of the two tensors.
     * tensors are broadcasted if there shapes are not equal but are compatible.
     */
    public static <A, B> JTensor<B> applyBinaryOperation(Class<B> resultType,
                                                         JTensor<A> tensor1,
                                                         JTensor<A> tensor2,
                                                         BiFunction<A, A, B> binaryOperation) {
        if (resultType == null) {
            throw new InvalidArgumentException("result type must not be null");
        }
        if (tensor1 == null || tensor2 == null) {
            throw new InvalidArgumentException("operands must not be null");
        }
        if (binaryOperation == null) {
            throw new InvalidArgumentException("binary operation must not be null");
        }
        final Pair<JTensor<A>, JTensor<A>> broadcast = broadcast(tensor1, tensor2);

        final JTensor<A> first = broadcast.getFirst();
        final JTensor<A> second = broadcast.getSecond();
        JTensor<B> result = new JTensor<>(resultType, first.shape);

        final Iterator<int[]> indicesIterator = result.indicesIterator();
        while (indicesIterator.hasNext()) {
            final int[] indices = indicesIterator.next();
            result.setItem(indices, binaryOperation.apply(first.getItem(indices), second.getItem(indices)));
        }
        return result;
    }

    /**
     * @param tensor          another tensor
     * @param binaryOperation a function that takes the class of the generic type and the two values and returns a new
     *                        value. The class of the generic type is needed since the Java language doesn't provide any
     *                        mechanism to check the given generic type at runtime. For example, this is needed when the
     *                        add operation is required and the numbers could be of class Integer or Double, etc. for
     *                        which the specific type must be known to perform the addition operation for that type.
     * @return a new tensor as a result of applying the binary operation on the corresponding items of the two tensors.
     * tensors are broadcasted if there shapes are not equal but are compatible.
     */
    public <B> JTensor<B> applyBinaryOperation(Class<B> resultType,
                                               JTensor<T> tensor,
                                               BiFunction<T, T, B> binaryOperation) {
        return applyBinaryOperation(resultType, this, tensor, binaryOperation);
    }

    /**
     * @param resultType the result tensor type.
     * @param tensor     the tensor to apply the function on.
     * @param function   a function that takes a value of the the type of the tensor and returns a value of the same
     *                   generic type of the result type.
     * @return a tensor as a result of applying the function on all the items of the tensor with the possibility of the
     * new tensor having a type other than the tensor's type.
     */
    public static <A, B> JTensor<B> applyFunction(Class<B> resultType,
                                                  JTensor<A> tensor,
                                                  Function<A, B> function) {
        if (resultType == null) {
            throw new InvalidArgumentException("result type must not be null");
        }
        if (tensor == null) {
            throw new InvalidArgumentException("tensor must not be null");
        }
        if (function == null) {
            throw new InvalidArgumentException("function must not be null");
        }
        JTensor<B> result = new JTensor<>(resultType, tensor.shape);

        final Iterator<int[]> indicesIterator = result.indicesIterator();
        while (indicesIterator.hasNext()) {
            final int[] indices = indicesIterator.next();
            result.setItem(indices, function.apply(tensor.getItem(indices)));
        }
        return result;
    }

    /**
     * @param tensor1 a tensor of numbers
     * @param tensor2 a tensor of numbers
     * @return a new tensor as a result of adding the corresponding items of each tensor. If the shapes of the two
     * tensors are not equal, broadcasting is performed if they are compatible.
     */
    public static <A extends Number> JTensor<A> add(JTensor<A> tensor1,
                                                    JTensor<A> tensor2) {
        requireTensor(tensor1);
        requireTensor(tensor2);
        return applyBinaryOperation(
                tensor1.type,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.add(tensor1.type, x, y));
    }

    /**
     * @param tensor1 a tensor of numbers
     * @param tensor2 a tensor of number
     * @return a new tensor that is a result of raising the items of the first tensor to the corresponding items
     * of the second tensor. If the shapes of the tensors are not equal, broadcasting is performed if they are compatible.
     */
    public static <A extends Number> JTensor<A> pow(JTensor<A> tensor1,
                                                    JTensor<A> tensor2) {
        requireTensor(tensor1);
        requireTensor(tensor2);
        tensor1.requireNonNullItems("Power");
        tensor2.requireNonNullItems("Power");
        return applyBinaryOperation(
                tensor1.type,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.cast(tensor1.type, Math.pow(x.doubleValue(), y.doubleValue())));
    }

    /**
     * @param tensor a tensor of numbers
     * @return a new tensor that is a result of taking the square root of all of its elements.
     */
    public static <A extends Number> JTensor<A> sqrt(JTensor<A> tensor) {
        requireTensor(tensor);
        tensor.requireNonNullItems("Square root");
        return applyFunction(
                tensor.type,
                tensor,
                (x) -> NumberHelper.cast(tensor.type, Math.sqrt(x.doubleValue())));
    }

    /**
     * @param tensor1 a tensor of numbers
     * @param tensor2 a tensor of numbers
     * @return a new tensor as a result of subtracting the corresponding items of each tensor. If the shapes of the two
     * tensors are not equal, broadcasting is performed if they are compatible.
     */
    public static <A extends Number> JTensor<A> subtract(JTensor<A> tensor1,
                                                         JTensor<A> tensor2) {
        requireTensor(tensor1);
        requireTensor(tensor2);
        return applyBinaryOperation(
                tensor1.type,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.subtract(tensor1.type, x, y));
    }

    /**
     * @param tensor1 a tensor of numbers
     * @param tensor2 a tensor of numbers
     * @return a new tensor as a result of multiplying the corresponding items of each tensor. If the shapes of the two
     * tensors are not equal, broadcasting is performed if they are compatible.
     */
    public static <A extends Number> JTensor<A> multiply(JTensor<A> tensor1,
                                                         JTensor<A> tensor2) {
        requireTensor(tensor1);
        requireTensor(tensor2);
        return applyBinaryOperation(
                tensor1.type,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.multiply(tensor1.type, x, y));
    }

    /**
     * @param tensor1 a tensor of numbers
     * @param tensor2 a tensor of numbers
     * @return a new tensor as a result of dividing the corresponding items of each tensor. If the shapes of the two
     * tensors are not equal, broadcasting is performed if they are compatible.
     */
    public static <A extends Number> JTensor<A> divide(JTensor<A> tensor1,
                                                       JTensor<A> tensor2) {
        requireTensor(tensor1);
        requireTensor(tensor2);
        return applyBinaryOperation(
                tensor1.type,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.divide(tensor1.type, x, y));
    }

    /**
     * @param tensor1 a tensor of numbers
     * @param tensor2 a tensor of numbers
     * @return a new tensor as a result of performing the modulo operation on corresponding items of each tensor. If the shapes of the two
     * tensors are not equal, broadcasting is performed if they are compatible.
     */
    public static <A extends Number> JTensor<A> mod(JTensor<A> tensor1,
                                                    JTensor<A> tensor2) {
        requireTensor(tensor1);
        requireTensor(tensor2);
        return applyBinaryOperation(
                tensor1.type,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.mod(tensor1.type, x, y));
    }

    /**
     * @param tensor1 a tensor of numbers
     * @param tensor2 a tensor of numbers
     * @return a new tensor as a result of performing the binary and operation on corresponding items of each tensor. If the shapes of the two
     * tensors are not equal, broadcasting is performed if they are compatible.
     */
    public static <A extends Number> JTensor<A> and(JTensor<A> tensor1,
                                                    JTensor<A> tensor2) {
        requireTensor(tensor1);
        requireTensor(tensor2);
        return applyBinaryOperation(
                tensor1.type,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.and(tensor1.type, x, y));
    }

    /**
     * @param tensor1 a tensor of numbers
     * @param tensor2 a tensor of numbers
     * @return a new tensor as a result of performing the binary or operation on corresponding items of each tensor. If the shapes of the two
     * tensors are not equal, broadcasting is performed if they are compatible.
     */
    public static <A extends Number> JTensor<A> or(JTensor<A> tensor1,
                                                   JTensor<A> tensor2) {
        requireTensor(tensor1);
        requireTensor(tensor2);
        return applyBinaryOperation(
                tensor1.type,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.or(tensor1.type, x, y));
    }

    /**
     * @param tensor1 a tensor of numbers
     * @param tensor2 a tensor of numbers
     * @return a new tensor as a result of performing the binary xor operation on corresponding items of each tensor. If the shapes of the two
     * tensors are not equal, broadcasting is performed if they are compatible.
     */
    public static <A extends Number> JTensor<A> xor(JTensor<A> tensor1,
                                                    JTensor<A> tensor2) {
        requireTensor(tensor1);
        requireTensor(tensor2);
        return applyBinaryOperation(
                tensor1.type,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.xor(tensor1.type, x, y));
    }

    /**
     * @param tensor1 a tensor of numbers
     * @param tensor2 a tensor of numbers
     * @return a new tensor as a result of performing the binary left shift operation on corresponding items of each tensor
     * where the items of the first tensor are. If the shapes of the two tensors are not equal, broadcasting is performed if they are compatible.
     */
    public static <A extends Number> JTensor<A> leftShift(JTensor<A> tensor1,
                                                          JTensor<A> tensor2) {
        requireTensor(tensor1);
        requireTensor(tensor2);
        return applyBinaryOperation(
                tensor1.type,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.leftShift(tensor1.type, x, y));
    }

    /**
     * @param tensor1 a tensor of numbers
     * @param tensor2 a tensor of numbers
     * @return a new tensor as a result of performing the binary right shift operation on corresponding items of each tensor
     * where the items of the first tensor are. If the shapes of the two tensors are not equal, broadcasting is performed if they are compatible.
     */
    public static <A extends Number> JTensor<A> rightShift(JTensor<A> tensor1,
                                                           JTensor<A> tensor2) {
        requireTensor(tensor1);
        requireTensor(tensor2);
        return applyBinaryOperation(
                tensor1.type,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.rightShift(tensor1.type, x, y));
    }

    /**
     * @param tensor a tensor of numbers
     * @return a new tensor as a result of performing the binary not operation on items the tensor.
     */
    public static <A extends Number> JTensor<A> not(JTensor<A> tensor) {
        requireTensor(tensor);
        return applyFunction(
                tensor.type,
                tensor,
                (x) -> NumberHelper.not(tensor.type, x));
    }

    /**
     * @param mappedType the type of the returned tensor
     * @param mapper     the element-wise mapper which takes each value to a new value of possibly a different type.
     * @return a tensor as a result of applying the element-wise mapper.
     */
    public <U> JTensor<U> map(Class<U> mappedType, Function<T, U> mapper) {
        return applyFunction(mappedType, this, mapper);
    }

    /**
     * @param predicate
     * @return a flat tensor by applying the mask constructed from applying the predicate on all the items of the tensor.
     */
    public JTensor<T> filter(Predicate<T> predicate) {
        if (predicate == null) {
            throw new InvalidArgumentException("predicate must not be null");
        }
        return applyMask(map(Boolean.class, predicate::test));
    }

    /**
     * @param replacePredicate
     * @param replacer
     * @return a tensor as a result of replacing items from the original tensor that satisfy the predicate.
     */
    public JTensor<T> replace(Predicate<T> replacePredicate, Function<T, T> replacer) {
        if (replacePredicate == null || replacer == null) {
            throw new InvalidArgumentException("replace predicate and replacer must not be null");
        }
        return map(type, x -> replacePredicate.test(x) ? replacer.apply(x) : x);
    }

    /**
     * @param dimension the dimension at witch to insert a new dimension of shape 1.
     * @return a tensor with an additional dimension of shape 1 inserted at the given dimension index.
     */
    public JTensor<T> expand(int dimension) {
        if (dimension < 0 || dimension > shape.length) {
            throw new InvalidArgumentException("invalid dimension " + dimension
                    + " for expansion of tensor rank " + shape.length);
        }
        if (shape.length == 0) {
            throw new InvalidArgumentException(
                    "cannot expand an empty rank-zero tensor because scalar tensors are not supported");
        }
        int[] expandedShape = new int[this.shape.length + 1];

        System.arraycopy(this.shape, 0, expandedShape, 0, dimension);
        expandedShape[dimension] = 1;
        System.arraycopy(this.shape, dimension, expandedShape, dimension + 1, this.shape.length - dimension);

        return this.reshape(expandedShape);
    }

    /**
     * @param identity
     * @param accumulator    a binary operation
     * @param dimension      the dimension to reduce along
     * @param keepDimensions if true, then the result tensor will have the same number of dimensions as the original
     * @return a tensor as a result of reducing all elements along the given dimension using the accumulator and
     * starting from the identity. If keepDimensions is true the new tensor will have the same number of dimensions as
     * the original one; otherwise, the new tensor will have a number of dimensions less than one from the original
     * dimensions. Since scalar tensors are not supported, reducing a rank-one tensor returns shape {@code [1]}.
     */
    public JTensor<T> reduceAlong(T identity,
                                  BiFunction<T, T, T> accumulator,
                                  int dimension,
                                  boolean keepDimensions) {
        validateDimension(dimension);
        if (accumulator == null) {
            throw new InvalidArgumentException("accumulator must not be null");
        }

        int[] tensorShape = getShape();
        int[] resultShape = Arrays.copyOf(tensorShape, tensorShape.length);
        resultShape[dimension] = 1;

        JTensor<T> result = repeat(type, resultShape, identity);

        Iterator<int[]> iterator = indicesIterator();
        while (iterator.hasNext()) {
            int[] indices = iterator.next();
            T number = getItem(indices);
            indices[dimension] = 0;
            result.setItem(indices,
                    accumulator.apply(result.getItem(indices), number));
        }

        if (keepDimensions) {
            return result;
        }

        if (tensorShape.length == 1) {
            // The library has no scalar representation; [1] is its scalar-like result.
            return result;
        }

        int[] newShape = new int[tensorShape.length - 1];
        int j = 0;
        for (int i = 0; i < tensorShape.length; i++) {
            if (i == dimension) {
                continue;
            }
            newShape[j] = tensorShape[i];
            j++;
        }

        return result.reshape(newShape);
    }

    /**
     * @param identity
     * @param accumulator    a binary operation
     * @param dimension
     * @param keepDimensions if true, then the result tensor will have the same number of dimensions as the original
     * @return a tensor as a result of reducing all items that the given dimension contains within the data. If
     * keepDimensions is true the new tensor will have the same number of dimensions as the original one; otherwise,
     * the new tensor will have only the dimensions before the given dimension of the original tensor. When no
     * dimensions remain, the scalar-like result has shape {@code [1]}.
     */
    public JTensor<T> reduceAll(T identity,
                                BiFunction<T, T, T> accumulator,
                                int dimension,
                                boolean keepDimensions) {
        validateDimension(dimension);
        if (accumulator == null) {
            throw new InvalidArgumentException("accumulator must not be null");
        }

        JTensor<T> reduced = reshape(getTensorShapeForReducing(dimension, shape))
                .reduceAlong(identity,
                        accumulator,
                        dimension,
                        true);

        if (keepDimensions) {
            int[] newShape = Arrays.copyOf(shape, shape.length);
            for (int i = dimension; i < shape.length; i++) {
                newShape[i] = 1;
            }
            return reduced.reshape(newShape);
        }

        if (shape.length == 1) {
            // The library has no scalar representation; [1] is its scalar-like result.
            return reduced;
        }

        if (reduced.getShape().length == 1) {
            return reduced;
        }

        return reduced.squeeze(dimension);
    }

    /**
     * @param dimension the dimension to reverse the indices along.
     * @return a tensor with indices reversed along a given dimension.
     */
    public JTensor<T> reverse(int dimension) {
        validateDimension(dimension);
        return new JTensor<T>(this.type,
                this.shape,
                this.data,
                this.size,
                this.strides,
                reversedIndicesTable(dimension),
                true);
    }

    private int[] reversedIndicesTable(int dimension) {
        int[] currentIndices = new int[shape.length];
        int[] newIndicesTable = new int[size];
        boolean isFirstTime = true;

        for (int i = 0; i < size; i++) {
            for (int j = currentIndices.length - 1; j >= 0; j--) {
                if (!isFirstTime) {
                    currentIndices[j] += 1;
                } else {
                    isFirstTime = false;
                }
                if (currentIndices[j] >= shape[j]) {
                    currentIndices[j] = 0;
                } else {
                    int flatIndex = 0;
                    for (int k = 0; k < currentIndices.length; k++) {
                        flatIndex += strides[k] * currentIndices[k];
                    }

                    int reversedFlatIndex = 0;
                    for (int k = 0; k < currentIndices.length; k++) {
                        int reversedDimensionIndex;
                        if (k == dimension) {
                            reversedDimensionIndex = this.shape[k] - currentIndices[k] - 1;
                        } else {
                            reversedDimensionIndex = currentIndices[k];
                        }
                        reversedFlatIndex += strides[k] * reversedDimensionIndex;
                    }

                    newIndicesTable[flatIndex] = this.indicesTable[reversedFlatIndex];
                    break;
                }
            }
        }

        return newIndicesTable;
    }

    private int[] getTensorShapeForReducing(int dimension, int[] shape) {
        int[] reducedShape = Arrays.copyOf(shape, dimension + 1);
        int collapsedDimension = 1;
        for (int i = dimension; i < shape.length; i++) {
            try {
                collapsedDimension = Math.multiplyExact(collapsedDimension, shape[i]);
            } catch (ArithmeticException exception) {
                throw new InvalidShapeException("tensor shape is too large", exception);
            }
        }
        reducedShape[dimension] = collapsedDimension;
        return reducedShape;
    }

    public static <T extends Number> JTensor<T> sum(JTensor<T> tensor,
                                                    int dimension,
                                                    boolean keepDimensions) {
        requireTensor(tensor);
        return tensor.reduceAlong(
                NumberHelper.zero(tensor.getType()),
                (x, y) -> NumberHelper.add(tensor.getType(), x, y),
                dimension,
                keepDimensions);
    }

    public static <T extends Number> JTensor<T> product(JTensor<T> tensor,
                                                        int dimension,
                                                        boolean keepDimensions) {
        requireTensor(tensor);
        return tensor.reduceAlong(
                NumberHelper.one(tensor.getType()),
                (x, y) -> NumberHelper.multiply(tensor.getType(), x, y),
                dimension,
                keepDimensions);
    }

    public static <T extends Number> JTensor<T> max(JTensor<T> tensor,
                                                    int dimension,
                                                    boolean keepDimensions) {
        requireTensor(tensor);
        return tensor.reduceAlong(
                NumberHelper.minValue(tensor.getType()),
                (x, y) -> NumberHelper.max(tensor.getType(), x, y),
                dimension,
                keepDimensions);
    }

    public static <T extends Number> JTensor<T> min(JTensor<T> tensor,
                                                    int dimension,
                                                    boolean keepDimensions) {
        requireTensor(tensor);
        return tensor.reduceAlong(
                NumberHelper.maxValue(tensor.getType()),
                (x, y) -> NumberHelper.min(tensor.getType(), x, y),
                dimension,
                keepDimensions);
    }

    public static <T extends Number> JTensor<T> mean(JTensor<T> tensor,
                                                     int dimension,
                                                     boolean keepDimensions) {
        requireTensor(tensor);
        tensor.validateDimension(dimension);
        JTensor<T> countTensor = repeat(
                new int[]{1},
                NumberHelper.cast(tensor.getType(), tensor.shape[dimension]));
        return JTensor.divide(
                sum(tensor, dimension, keepDimensions),
                countTensor);
    }

    public static <T extends Number> JTensor<T> var(JTensor<T> tensor, int dimension, boolean keepDimensions) {
        requireTensor(tensor);
        tensor.validateDimension(dimension);
        JTensor<T> countTensor = repeat(
                new int[]{1},
                NumberHelper.cast(tensor.getType(), tensor.shape[dimension]));
        JTensor<T> powerTensor = repeat(
                new int[]{1},
                NumberHelper.cast(tensor.type, 2));

        JTensor<T> meanTensor = mean(tensor, dimension, true);
        JTensor<T> squares = JTensor.pow(subtract(tensor, meanTensor), powerTensor);
        return divide(JTensor.sum(squares, dimension, keepDimensions), countTensor);
    }

    public static <T extends Number> JTensor<T> std(JTensor<T> tensor, int dimension, boolean keepDimensions) {
        requireTensor(tensor);
        return sqrt(var(tensor, dimension, keepDimensions));
    }

    public static <A extends Number, B extends Number> JTensor<B> cast(JTensor<A> tensor, Class<B> newType) {
        requireTensor(tensor);
        return tensor.map(newType, x -> NumberHelper.cast(newType, x));
    }

    public static <A extends Number> JTensor<Boolean> castToBoolean(JTensor<A> tensor) {
        requireTensor(tensor);
        return tensor.map(Boolean.class, x -> NumberHelper.booleanValue(tensor.getType(), x));
    }

    public static <A extends Number> JTensor<A> castFromBoolean(Class<A> newType,
                                                                JTensor<Boolean> tensor) {
        requireTensor(tensor);
        tensor.requireNonNullItems("Boolean cast");
        return tensor.map(newType, x -> NumberHelper.cast(newType, x ? 1 : 0));
    }

    public static <A extends Number> JTensor<Boolean> greaterThan(JTensor<A> tensor1, JTensor<A> tensor2) {
        return applyBinaryOperation(
                Boolean.class,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.greaterThan(tensor1.type, x, y));
    }

    public static <A extends Number> JTensor<Boolean> lessThan(JTensor<A> tensor1, JTensor<A> tensor2) {
        return applyBinaryOperation(
                Boolean.class,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.lessThan(tensor1.type, x, y));
    }

    public static <A extends Number> JTensor<Boolean> equals(JTensor<A> tensor1, JTensor<A> tensor2) {
        return applyBinaryOperation(
                Boolean.class,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.equals(tensor1.type, x, y));
    }

    public static <A extends Number> JTensor<Boolean> notEquals(JTensor<A> tensor1, JTensor<A> tensor2) {
        return applyBinaryOperation(
                Boolean.class,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.notEquals(tensor1.type, x, y));
    }

    public static <A extends Number> JTensor<Boolean> greaterThanOrEquals(JTensor<A> tensor1, JTensor<A> tensor2) {
        return applyBinaryOperation(
                Boolean.class,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.greaterThanOrEquals(tensor1.type, x, y));
    }

    public static <A extends Number> JTensor<Boolean> lessThanOrEquals(JTensor<A> tensor1, JTensor<A> tensor2) {
        return applyBinaryOperation(
                Boolean.class,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.lessThanOrEquals(tensor1.type, x, y));
    }

    public static <A extends Number> JTensor<Boolean> booleanAnd(JTensor<Boolean> tensor1, JTensor<Boolean> tensor2) {
        requireTensor(tensor1).requireNonNullItems("Boolean and");
        requireTensor(tensor2).requireNonNullItems("Boolean and");
        return applyBinaryOperation(
                Boolean.class,
                tensor1,
                tensor2,
                (x, y) -> x & y);
    }

    public static <A extends Number> JTensor<Boolean> booleanOr(JTensor<Boolean> tensor1, JTensor<Boolean> tensor2) {
        requireTensor(tensor1).requireNonNullItems("Boolean or");
        requireTensor(tensor2).requireNonNullItems("Boolean or");
        return applyBinaryOperation(
                Boolean.class,
                tensor1,
                tensor2,
                (x, y) -> x | y);
    }


    public static <A extends Number> JTensor<Boolean> booleanXor(JTensor<Boolean> tensor1, JTensor<Boolean> tensor2) {
        requireTensor(tensor1).requireNonNullItems("Boolean xor");
        requireTensor(tensor2).requireNonNullItems("Boolean xor");
        return applyBinaryOperation(
                Boolean.class,
                tensor1,
                tensor2,
                (x, y) -> x ^ y);
    }


    public static <A extends Number> JTensor<Boolean> booleanNot(JTensor<Boolean> tensor) {
        requireTensor(tensor).requireNonNullItems("Boolean not");
        return applyFunction(
                Boolean.class,
                tensor,
                (x) -> !x);
    }

    public static <A extends Number> JTensor<Integer> compare(JTensor<A> tensor1, JTensor<A> tensor2) {
        return applyBinaryOperation(
                Integer.class,
                tensor1,
                tensor2,
                (x, y) -> NumberHelper.compare(tensor1.type, x, y));
    }

    private static class IndexNumberPair extends Pair<Integer, Number> {
        public IndexNumberPair(Integer first, Number second) {
            super(first, second);
        }
    }

    private static boolean isNaN(Number number) {
        return number instanceof Float && Float.isNaN(number.floatValue())
                || number instanceof Double && Double.isNaN(number.doubleValue());
    }

    public static <A extends Number> JTensor<Integer> argMax(JTensor<A> tensor, int dimension, boolean keepDimensions) {
        requireTensor(tensor);
        return indexBasedReduction(tensor,
                new IndexNumberPair(-1, NumberHelper.minValue(tensor.type)),
                (x, y) -> {
                    if (x.getFirst() < 0) {
                        return y;
                    }
                    if (isNaN(x.getSecond())) {
                        return x;
                    }
                    if (isNaN(y.getSecond())) {
                        return y;
                    }
                    return NumberHelper.compare(
                            tensor.type, x.getSecond(), y.getSecond()) >= 0 ? x : y;
                },
                dimension,
                keepDimensions);
    }

    public static <A extends Number> JTensor<Integer> argMin(JTensor<A> tensor, int dimension, boolean keepDimensions) {
        requireTensor(tensor);
        return indexBasedReduction(tensor,
                new IndexNumberPair(-1, NumberHelper.maxValue(tensor.type)),
                (x, y) -> {
                    if (x.getFirst() < 0) {
                        return y;
                    }
                    if (isNaN(x.getSecond())) {
                        return x;
                    }
                    if (isNaN(y.getSecond())) {
                        return y;
                    }
                    return NumberHelper.compare(
                            tensor.type, x.getSecond(), y.getSecond()) <= 0 ? x : y;
                },
                dimension,
                keepDimensions);
    }

    private static <A extends Number> JTensor<Integer> indexBasedReduction(JTensor<A> tensor,
                                                                           IndexNumberPair identity,
                                                                           BiFunction<IndexNumberPair, IndexNumberPair, IndexNumberPair> accumulator,
                                                                           int dimension,
                                                                           boolean keepDimensions) {
        tensor.validateDimension(dimension);
        Iterator<int[]> iterator = tensor.indicesIterator();
        JTensor<IndexNumberPair> indicesWithNumbersTensor = tensor.map(
                IndexNumberPair.class,
                (x -> new IndexNumberPair(iterator.next()[dimension], x)));

        return indicesWithNumbersTensor.reduceAlong(identity,
                accumulator,
                dimension,
                keepDimensions).map(Integer.class, Pair::getFirst);
    }

    /**
     * @return a pair of tensors with the same shape as a result of broadcasting either one of the tensors or both
     * according to broadcasting rules. The pair of tensors correspond to the parameters with the same order.
     * For more on the broadcasting rules used, see NumPy's general broadcasting rules:
     * https://numpy.org/doc/stable/user/basics.broadcasting.html.
     * @throws InvalidArgumentException if the shapes of the tensors are not compatible.
     */
    public static <A, B> Pair<JTensor<A>, JTensor<B>> broadcast(JTensor<A> tensor1,
                                                                JTensor<B> tensor2) {
        if (tensor1 == null || tensor2 == null) {
            throw new InvalidArgumentException("tensors to broadcast must not be null");
        }
        final int[] shape1 = tensor1.shape;
        final int[] shape2 = tensor2.shape;

        if (Arrays.equals(shape1, shape2)) {
            return new Pair<>(tensor1, tensor2);
        }
        if (tensor1.size == 0 || tensor2.size == 0) {
            throw new InvalidArgumentException(
                    "an empty rank-zero tensor cannot be broadcast as a scalar");
        }

        if (!areShapesCompatible(shape1, shape2)) {
            throw new InvalidArgumentException("could not broadcast operands together");
        }

        final Pair<JTensor<A>, JTensor<B>> tensorsPair = makeLargerTensorTrailingShapeEqualToSmallerTensorShape(tensor1, tensor2);
        JTensor<A> compatibleTensor1 = tensorsPair.getFirst();
        JTensor<B> compatibleTensor2 = tensorsPair.getSecond();

        final int dimensions1 = shape1.length;
        final int dimensions2 = shape2.length;

        if (dimensions2 < dimensions1) {
            return new Pair<>(compatibleTensor1, stretchTensor(compatibleTensor2, compatibleTensor1.shape));
        } else if (dimensions1 < dimensions2) {
            return new Pair<>(stretchTensor(compatibleTensor1, compatibleTensor2.shape), compatibleTensor2);
        } else {
            return tensorsPair;
        }
    }

    /**
     * @return a pair of tensors with the same shape as a result of broadcasting either one of the tensors or both
     * according to broadcasting rules. The pair of tensors correspond to the parameters with the same order.
     * For more on the broadcasting rules used, see NumPy's general broadcasting rules:
     * https://numpy.org/doc/stable/user/basics.broadcasting.html.
     * @throws InvalidArgumentException if the shapes of the tensors are not compatible.
     */
    public <B> Pair<JTensor<T>, JTensor<B>> broadcast(JTensor<B> tensor) {
        return broadcast(this, tensor);
    }

    /**
     * Checks for the compatibility of the shapes according to NumPy's general broadcasting rules:
     * https://numpy.org/doc/stable/user/basics.broadcasting.html.
     */
    private static boolean areShapesCompatible(int[] shape1, int[] shape2) {
        int[] larger;
        int[] smaller;

        if (shape1.length > shape2.length) {
            larger = shape1;
            smaller = shape2;
        } else {
            larger = shape2;
            smaller = shape1;
        }

        final int largerDimensions = larger.length;
        final int smallerDimensions = smaller.length;

        int j = smallerDimensions - 1;
        for (int i = largerDimensions - 1; i >= largerDimensions - smallerDimensions; i--) {
            if (larger[i] != smaller[j] && larger[i] != 1 && smaller[j] != 1) {
                return false;
            }
            j--;
        }

        return true;
    }

    /**
     * Larger tensor trailing shape is the sub-array containing the shapes of the last k dimensions where k is the
     * number of dimensions of the smaller tensor. This method makes the larger tensor trailing shape equal to the
     * shape of the smaller tensor either by stretching the larger tensor or the smaller tensor or both along the
     * dimensions of shape1.
     */
    private static <A, B> Pair<JTensor<A>, JTensor<B>> makeLargerTensorTrailingShapeEqualToSmallerTensorShape(JTensor<A> tensor1,
                                                                                                              JTensor<B> tensor2) {
        int[] shape1 = tensor1.shape;
        int[] shape2 = tensor2.shape;

        int[] larger;
        int[] smaller;

        boolean isFirstTensorLarger = false;

        if (shape1.length > shape2.length) {
            larger = shape1;
            smaller = shape2;
            isFirstTensorLarger = true;
        } else {
            larger = shape2;
            smaller = shape1;
        }

        final int largerDimensions = larger.length;
        final int smallerDimensions = smaller.length;

        JTensor<A> stretched1 = tensor1;
        JTensor<B> stretched2 = tensor2;

        int j = smallerDimensions - 1;
        for (int i = largerDimensions - 1; i >= largerDimensions - smallerDimensions; i--) {
            if (larger[i] == 1) {
                if (isFirstTensorLarger) {
                    stretched1 = stretchTensorAlongDimension(stretched1, i, smaller[j]);
                } else {
                    stretched2 = stretchTensorAlongDimension(stretched2, i, smaller[j]);
                }
            }
            if (smaller[j] == 1) {
                if (isFirstTensorLarger) {
                    stretched2 = stretchTensorAlongDimension(stretched2, j, larger[i]);
                } else {
                    stretched1 = stretchTensorAlongDimension(stretched1, j, larger[i]);
                }
            }
            j--;
        }

        return new Pair<>(stretched1, stretched2);
    }

    /**
     * Stretches the tensor by copying each item along the given dimension according to the correspondingDimensionShape.
     */
    private static <A> JTensor<A> stretchTensorAlongDimension(JTensor<A> tensor,
                                                              int dimension,
                                                              int correspondingDimensionShape) {
        final int[] originalShape = tensor.shape;
        int[] newShape = Arrays.copyOf(originalShape, originalShape.length);
        newShape[dimension] = correspondingDimensionShape;
        JTensor<A> adjusted = new JTensor<>(tensor.getType(), newShape);

        final Iterator<int[]> indicesIterator = adjusted.indicesIterator();
        while (indicesIterator.hasNext()) {
            final int[] indices = indicesIterator.next();
            int i = indices[dimension];

            indices[dimension] = 0;
            A item = tensor.getItem(indices);

            indices[dimension] = i;
            adjusted.setItem(indices, item);
        }
        return adjusted;
    }

    /**
     * Stretches the tensor by copying each item in the tensor along the the initial part the destinationShape that is
     * different from the shape of the tensor. In other words, the trailing dimensions of the new tensor are equivalent
     * to the dimensions of the tensor and two different indices of the new tensor contain the same item the last k
     * indices (where k is the number of dimensions in the original tensor) are equal.
     */
    private static <A> JTensor<A> stretchTensor(JTensor<A> tensor, int[] destinationShape) {
        JTensor<A> stretched = new JTensor<>(tensor.getType(), destinationShape);
        final Iterator<int[]> iterator = stretched.indicesIterator();
        while (iterator.hasNext()) {
            final int[] destinationIndices = iterator.next();
            stretched.setItem(
                    destinationIndices,
                    tensor.getItem(getTrailingIndices(destinationIndices, tensor.shape.length)));
        }
        return stretched;
    }

    private static int[] getTrailingIndices(int[] indices, int trailingDimensions) {
        return Arrays.copyOfRange(
                indices,
                indices.length - trailingDimensions,
                indices.length);
    }

    /**
     * @return returns a FloatBuffer object containing the tensor's data
     */
    public FloatBuffer toFloatBuffer() {
        if (Float.class.equals(this.type)) {
            requireNonNullItems("Float buffer conversion");
            FloatBuffer floatBuffer = FloatBuffer.allocate(size);
            Iterator<int[]> iterator = indicesIterator();
            while (iterator.hasNext()) {
                floatBuffer.put((Float) getItem(iterator.next()));
            }
            floatBuffer.position(0);
            return floatBuffer;
        }
        throw new InvalidTypeException("type of the tensor is not Float");
    }

    /**
     * @return returns a LongBuffer object containing the tensor's data
     */
    public LongBuffer toLongBuffer() {
        if (Long.class.equals(this.type)) {
            requireNonNullItems("Long buffer conversion");
            LongBuffer longBuffer = LongBuffer.allocate(size);
            Iterator<int[]> iterator = indicesIterator();
            while (iterator.hasNext()) {
                longBuffer.put((Long) getItem(iterator.next()));
            }
            longBuffer.position(0);
            return longBuffer;
        }
        throw new InvalidTypeException("type of the tensor is not Long");
    }

    /**
     * @return returns a DoubleBuffer object containing the tensor's data
     */
    public DoubleBuffer toDoubleBuffer() {
        if (Double.class.equals(this.type)) {
            requireNonNullItems("Double buffer conversion");
            DoubleBuffer doubleBuffer = DoubleBuffer.allocate(size);
            Iterator<int[]> iterator = indicesIterator();
            while (iterator.hasNext()) {
                doubleBuffer.put((Double) getItem(iterator.next()));
            }
            doubleBuffer.position(0);
            return doubleBuffer;
        }
        throw new InvalidTypeException("type of the tensor is not Double");
    }

    /**
     * @return returns a IntBuffer object containing the tensor's data
     */
    public IntBuffer toIntBuffer() {
        if (Integer.class.equals(this.type)) {
            requireNonNullItems("Integer buffer conversion");
            IntBuffer intBuffer = IntBuffer.allocate(size);
            Iterator<int[]> iterator = indicesIterator();
            while (iterator.hasNext()) {
                intBuffer.put((Integer) getItem(iterator.next()));
            }
            intBuffer.position(0);
            return intBuffer;
        }
        throw new InvalidTypeException("type of the tensor is not Integer");
    }

    /**
     * @return byte array representation of the tensor
     */
    public byte[] toByteArray() {
        DataType dataType;
        if (Boolean.class.equals(this.type)) {
            dataType = BOOLEAN;
        } else if (Byte.class.equals(this.type)) {
            dataType = BYTE;
        } else if (Short.class.equals(this.type)) {
            dataType = SHORT;
        } else if (Integer.class.equals(this.type)) {
            dataType = INTEGER;
        } else if (Float.class.equals(this.type)) {
            dataType = FLOAT;
        } else if (Long.class.equals(this.type)) {
            dataType = LONG;
        } else if (Double.class.equals(this.type)) {
            dataType = DOUBLE;
        } else {
            throw new InvalidTypeException("not implemented for type " + this.type.getName());
        }

        Iterator<int[]> nullCheckIterator = indicesIterator();
        while (nullCheckIterator.hasNext()) {
            if (getItem(nullCheckIterator.next()) == null) {
                throw new InvalidArgumentException(
                        "cannot serialize a tensor containing null values");
            }
        }

        long totalBits = (long) dataType.getSize() * size;
        long numberOfBytes = (totalBits + 7L) / 8L;
        long headerBytes = 5L + 8L * shape.length;
        long serializedBytes = numberOfBytes + headerBytes;
        if (serializedBytes > Integer.MAX_VALUE) {
            throw new InvalidArgumentException("serialized tensor is too large");
        }

        ByteBuffer byteBuffer = ByteBuffer.allocate((int) serializedBytes);
        byteBuffer.put(dataType.getValue());
        byteBuffer.putInt(this.shape.length);
        for (int x : shape) {
            byteBuffer.putInt(x);
        }
        for (int x : initializeStrides(shape)) {
            byteBuffer.putInt(x);
        }

        Iterator<int[]> iterator = indicesIterator();

        if (Boolean.class.equals(this.type)) {
            int currentBitIndex = 0;
            int currentByte = 0;

            while (iterator.hasNext()) {
                Boolean x = (Boolean) getItem(iterator.next());
                if (x) {
                    currentByte += (1 << currentBitIndex);
                }
                currentBitIndex++;

                if (currentBitIndex == 8) {
                    byteBuffer.put(Integer.valueOf(currentByte).byteValue());
                    currentByte = 0;
                    currentBitIndex = 0;
                }
            }

            if (currentBitIndex > 0) {
                byteBuffer.put(Integer.valueOf(currentByte).byteValue());
            }
        } else if (Byte.class.equals(this.type)) {
            while (iterator.hasNext()) {
                byteBuffer.put((Byte) getItem(iterator.next()));
            }
        } else if (Short.class.equals(this.type)) {
            while (iterator.hasNext()) {
                byteBuffer.putShort((Short) getItem(iterator.next()));
            }
        } else if (Integer.class.equals(this.type)) {
            while (iterator.hasNext()) {
                byteBuffer.putInt((Integer) getItem(iterator.next()));
            }
        } else if (Float.class.equals(this.type)) {
            while (iterator.hasNext()) {
                byteBuffer.putFloat((Float) getItem(iterator.next()));
            }
        } else if (Long.class.equals(this.type)) {
            while (iterator.hasNext()) {
                byteBuffer.putLong((Long) getItem(iterator.next()));
            }
        } else {
            while (iterator.hasNext()) {
                byteBuffer.putDouble((Double) getItem(iterator.next()));
            }
        }

        byteBuffer.position(0);
        return byteBuffer.array();
    }

    public static JTensor<?> fromByteArray(byte[] byteArray) {
        if (byteArray == null || byteArray.length < 5) {
            throw new InvalidArgumentException("given byte array is invalid");
        }
        ByteBuffer byteBuffer = ByteBuffer.wrap(byteArray);

        DataType dataType;
        try {
            dataType = DataType.fromValue(byteBuffer.get());
        } catch (IllegalArgumentException exception) {
            throw new InvalidArgumentException("given byte array has an invalid data type", exception);
        }
        int shapeLength = byteBuffer.getInt();
        long headerSize = 5L + 8L * shapeLength;
        if (shapeLength < 0 || headerSize > byteArray.length) {
            throw new InvalidArgumentException("given byte array has an invalid shape rank");
        }

        int[] shape = new int[shapeLength];
        for (int i = 0; i < shapeLength; i++) {
            shape[i] = byteBuffer.getInt();
        }

        int size;
        try {
            size = initializeSize(shape);
        } catch (InvalidShapeException exception) {
            throw new InvalidArgumentException("given byte array has an invalid tensor shape", exception);
        }

        int[] serializedStrides = new int[shapeLength];
        for (int i = 0; i < shapeLength; i++) {
            serializedStrides[i] = byteBuffer.getInt();
        }
        int[] strides = initializeStrides(shape);
        if (!Arrays.equals(serializedStrides, strides)) {
            throw new InvalidArgumentException("given byte array has non-contiguous tensor strides");
        }

        long payloadBits = (long) dataType.getSize() * size;
        long expectedPayloadBytes = (payloadBits + 7L) / 8L;
        if (byteBuffer.remaining() != expectedPayloadBytes) {
            throw new InvalidArgumentException("given byte array has an invalid payload size");
        }

        if (BOOLEAN == dataType) {
            Boolean[] data = new Boolean[size];

            int currentIndex = 0;
            while (currentIndex < size) {
                byte x = byteBuffer.get();

                for (int i = 0; i < 8 && currentIndex < size; i++) {
                    data[currentIndex] = (x & 1) == 1;
                    x >>= 1;
                    currentIndex++;
                }
            }

            return new JTensor<>(Boolean.class,
                    shape,
                    data,
                    size,
                    strides,
                    defaultIndicesTable(size),
                    false);
        } else if (BYTE == dataType) {
            Byte[] data = new Byte[size];

            for (int i = 0; i < size; i++) {
                data[i] = byteBuffer.get();
            }

            return new JTensor<>(Byte.class,
                    shape,
                    data,
                    size,
                    strides,
                    defaultIndicesTable(size),
                    false);
        } else if (SHORT == dataType) {
            Short[] data = new Short[size];

            for (int i = 0; i < size; i++) {
                data[i] = byteBuffer.getShort();
            }

            return new JTensor<>(Short.class,
                    shape,
                    data,
                    size,
                    strides,
                    defaultIndicesTable(size),
                    false);
        } else if (INTEGER == dataType) {
            Integer[] data = new Integer[size];

            for (int i = 0; i < size; i++) {
                data[i] = byteBuffer.getInt();
            }

            return new JTensor<>(Integer.class,
                    shape,
                    data,
                    size,
                    strides,
                    defaultIndicesTable(size),
                    false);
        } else if (FLOAT == dataType) {
            Float[] data = new Float[size];

            for (int i = 0; i < size; i++) {
                data[i] = byteBuffer.getFloat();
            }

            return new JTensor<>(Float.class,
                    shape,
                    data,
                    size,
                    strides,
                    defaultIndicesTable(size),
                    false);
        } else if (LONG == dataType) {
            Long[] data = new Long[size];

            for (int i = 0; i < size; i++) {
                data[i] = byteBuffer.getLong();
            }

            return new JTensor<>(Long.class,
                    shape,
                    data,
                    size,
                    strides,
                    defaultIndicesTable(size),
                    false);
        } else if (DOUBLE == dataType) {
            Double[] data = new Double[size];

            for (int i = 0; i < size; i++) {
                data[i] = byteBuffer.getDouble();
            }

            return new JTensor<>(Double.class,
                    shape,
                    data,
                    size,
                    strides,
                    defaultIndicesTable(size),
                    false);
        } else {
            throw new InvalidArgumentException("given byte array is invalid");
        }
    }

    @Override
    public String toString() {
        if (shape.length == 0) {
            return "[]";
        }

        int maxNumberLength = -1;
        final T[] raveledData = this.ravel().data;

        for (T number : raveledData) {
            final int numberLength = String.valueOf(number).length();
            if (numberLength > maxNumberLength) {
                maxNumberLength = numberLength;
            }
        }

        StringBuilder representation = new StringBuilder();

        int[] indices = new int[shape.length];
        boolean[] openedParentheses = new boolean[shape.length];

        for (int n = 0; n < size; n++) {
            for (int i = 0; i < indices.length; i++) {
                if (!openedParentheses[i]) {
                    representation.append("[");
                    openedParentheses[i] = true;
                }
            }

            final String numberRepresentation = String.valueOf(getItem(indices));
            representation
                    .append(" ".repeat(maxNumberLength - numberRepresentation.length()))
                    .append(numberRepresentation);

            final int lastDimensionIndex = shape.length - 1;
            if (indices[lastDimensionIndex] < shape[lastDimensionIndex] - 1) {
                representation.append(", ");
            }

            boolean closedAnyParentheses = false;
            int numberOfClosedParentheses = 0;

            for (int i = indices.length - 1; i >= 0; i--) {
                indices[i] += 1;
                if (indices[i] >= shape[i]) {
                    indices[i] = 0;

                    representation.append("]");
                    if (i > 0 && indices[i - 1] < shape[i - 1] - 1) {
                        representation.append(",");
                    }

                    openedParentheses[i] = false;
                    closedAnyParentheses = true;
                    numberOfClosedParentheses++;
                } else {
                    break;
                }
            }

            if (closedAnyParentheses) {
                if (numberOfClosedParentheses > 1) {
                    representation.append("\n");
                }
                representation.append("\n");
                representation.append(" ".repeat(shape.length - numberOfClosedParentheses));
            }
        }

        return representation.toString().trim();
    }

    private static int[] defaultIndicesTable(int size) {
        int[] indices = new int[size];
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        return indices;
    }

    private static class TensorBuilder<T> {
        private Class<T> type;
        private int[] shape;
        private T[] data;
        private int size;
        private int[] strides;
        private int[] indicesTable;
        private boolean isView;

        public TensorBuilder<T> setType(Class<T> type) {
            this.type = type;
            return this;
        }

        public TensorBuilder<T> setShape(int[] shape) {
            this.shape = shape;
            return this;
        }

        public TensorBuilder<T> setData(T[] data) {
            this.data = data;
            return this;
        }

        public TensorBuilder<T> setSize(int size) {
            this.size = size;
            return this;
        }

        public TensorBuilder<T> setStrides(int[] strides) {
            this.strides = strides;
            return this;
        }

        public TensorBuilder<T> setIndicesTable(int[] indicesTable) {
            this.indicesTable = indicesTable;
            return this;
        }

        public TensorBuilder<T> setView(boolean view) {
            isView = view;
            return this;
        }

        public JTensor<T> build() {
            return new JTensor<>(
                    type,
                    shape,
                    data,
                    size,
                    strides,
                    indicesTable,
                    isView);
        }
    }
}
