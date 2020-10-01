# java-tensor
![alt text](logo.png)

A Tensor is a multi-dimensional array with arrays of equal lengths (dimension size or shape) in each dimension. Tensors are equipped with many useful operations that could either be performed on the tensor itself or more than one tensor. Many implementations of the same concept already exist in many different languages such as the well-known NumPy (in Python) which is the main inspiration for this implementation. Tensors are represented in memory as a single-dimensional array which makes them more efficient than language-specific multi-dimensional arrays. Moreover, Tensors can map a multi-dimensional index (array of indices) to a flat or single-dimensional index that points to an item within the single-dimensional array. In this implementation, strides and special mappings are used to map a multi-dimensional index to a flat index. A stride is a fixed-size step such that each dimension of the tensor has its own stride. The stride of a dimension is calculated by taking the product of all the dimension sizes (shapes) proceeding it. For example, if the dimension shapes of a tensor are (2, 3, 4) which is also called the shape of the tensor, then the first dimension has a stride of 3 * 4 = 12 and the second dimension has a stride of 4 and the last dimension has a stride of 1. In general, given any multi-dimensional index represented as an array of indices, each index within that array is multiplied with the stride at that dimension and then the results are summed together to obtain a flat index. As with the previous of the tensor of shape (2, 3, 4), to obtain the item at the index (1, 1, 1), we compute 1 * 12 + 1 * 4 + 1 = 17, then take the item in the flat array at the index 17. This is called the row-major order of the indices since the strides get smaller from left to right. In this way, each dimension is represented as non-overlapping fixed-size blocks of contiguous items of the array. Strides are always strictly positive in this implementation unlike some other implementations where strides could be negative or zero. In some cases where a more sophisticated mapping of the indices is required, we took a functional approach to meet them all of which utilizes the generality and composability of mappings and made the implementation of these operations much easier. When some operations are performed on a tensor, the data is not always copied which introduces the concept of a view of a tensor, which is a shallow copy of the data in contrast to a deep copy of the data, to reduce memory overhead. This implementation contains all essential operations such tensor manipulation operations, tensors broadcasting, unary and binary operations, functional operations, etc. It could also be easily extended and customized according to specific needs. The reason behind this implementation is the need for a very clear and simple implementation of tensors and their operations that could be customized when required.

## Examples

#### Example 1
code:
```java
Tensor<Integer> tensor = Tensor.ones(Integer.class, new int[]{2, 2, 2});
System.out.println(tensor);
```

result:
```
[[[1, 1],
  [1, 1]],

 [[1, 1],
  [1, 1]]]
```

#### Example 2
code:
```java
Tensor<Double> tensor = new Tensor<>(
        Double.class,
        new int[]{2, 3},
        new Double[]{
                5.0, 6.0, 1.0,
                -1.0, 0.0, 2.0});
System.out.println(tensor.reshape(new int[]{3, 2}));
```

result:
```
[[ 5.0,  6.0],
 [ 1.0, -1.0],
 [ 0.0,  2.0]]
```

#### Example 3
code:
```java
Tensor<Double> tensor = new Tensor<>(
        Double.class,
        new int[]{2, 3},
        new Double[]{
                5.0, 6.0, 1.0,
                -1.0, 0.0, 2.0});
System.out.println(tensor.transpose());
```

result:
```
[[ 5.0, -1.0],
 [ 6.0,  0.0],
 [ 1.0,  2.0]]
```

#### Example 4
code:
```java
Tensor<Double> tensor = new Tensor<>(
        Double.class,
        new int[]{2, 3},
        new Double[]{
                5.0, 6.0, 1.0,
                -1.0, 0.0, 2.0});
Tensor<Boolean> mask = Tensor.greaterThan(tensor, Tensor.singleValue(0.0));
System.out.println(tensor.applyMask(mask));
```

result:
```
[5.0, 6.0, 1.0, 2.0]
```

#### Example 5
code:
```java
Tensor<Double> tensor = new Tensor<>(
        Double.class,
        new int[]{2, 3},
        new Double[]{
                5.0, 6.0, 1.0,
                -1.0, 0.0, 2.0});
System.out.println(tensor.slice(new int[][]{{0, 2}, {1, 3}}));
```

result:
```
[[6.0, 1.0],
 [0.0, 2.0]]
```

#### Example 6
code:
```java
Tensor<Double> tensor1 = new Tensor<>(
        Double.class,
        new int[]{3, 1},
        new Double[]{
                1.0, 2.0, 3.0});

Tensor<Double> tensor2 = new Tensor<>(
        Double.class,
        new int[]{1, 3},
        new Double[]{
                4.0, 5.0, 6.0});

System.out.println(Tensor.multiply(tensor1, tensor2));
```

```
[[ 4.0,  5.0,  6.0],
 [ 8.0, 10.0, 12.0],
 [12.0, 15.0, 18.0]]
```

#### Example 7
code:
```java
Tensor<Double> tensor = new Tensor<>(
        Double.class,
        new int[]{2, 2, 2},
        new Double[]{-1.0, 5.0, 2.0, 4.0, -6.0, 9.0, 1.5, 7.2});

System.out.println(Tensor.max(tensor, 1, false));
```

result:
```
[[2.0, 5.0],
 [1.5, 9.0]]
```