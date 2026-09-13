package com.tensor;

import org.junit.Test;
import java.util.*;
import static org.junit.Assert.*;

public class JTensorExtendedOperationsTest {
    private static JTensor<Integer> ints(int... values) {
        Integer[] boxed = Arrays.stream(values).boxed().toArray(Integer[]::new);
        return values.length == 0 ? JTensor.empty(Integer.class) : JTensor.from1DArray(Integer.class, boxed);
    }
    private static JTensor<Double> doubles(double... values) {
        Double[] boxed = Arrays.stream(values).boxed().toArray(Double[]::new);
        return JTensor.from1DArray(Double.class, boxed);
    }
    private static void values(JTensor<?> actual, Object... expected) { assertArrayEquals(expected, actual.toArray()); }
    private static void shape(JTensor<?> actual, int... expected) { assertArrayEquals(expected, actual.getShape()); }
    private static void invalid(Runnable action) {
        assertThrows(InvalidArgumentException.class, action::run);
    }

    @Test public void shapeAndConversionHelpersUseLogicalOrder() {
        JTensor<Integer> a = ints(0,1,2,3,4,5).reshape(new int[]{2,3}).transpose();
        assertEquals(2, a.numberOfDimensions()); assertEquals(6, a.size()); assertEquals(2, a.size(-1));
        values(a, 0,3,1,4,2,5); assertEquals(Arrays.asList(0,3,1,4,2,5), a.toList());
        a.toArray()[0] = 99; a.toList().set(0,99); assertEquals(Integer.valueOf(0), a.item(0,0));
        assertEquals(Integer.valueOf(7), ints(7).item()); invalid(a::item); invalid(() -> a.size(-3));
    }

    @Test public void flattenCopiesAndRavelPreservesValues() {
        JTensor<Integer> a = ints(1,2,3,4).reshape(new int[]{2,2});
        JTensor<Integer> flat = a.flatten(); flat.setItem(new int[]{0}, 90);
        assertEquals(Integer.valueOf(1), a.item(0,0));
        values(a.transpose().ravel(), 1,3,2,4); shape(flat,4);
    }

    @Test public void squeezeAndUnsqueezeHandleAxes() {
        JTensor<Integer> a = ints(1,2).reshape(new int[]{1,2,1,1});
        shape(a.squeeze(), 2); shape(a.squeeze(0,3), 2,1);
        shape(a.unsqueeze(-1),1,2,1,1,1); shape(a.unsqueeze(0),1,1,2,1,1);
        shape(ints(1).squeeze(),1);
        invalid(() -> a.squeeze(1)); invalid(() -> a.squeeze(0,0)); invalid(() -> a.unsqueeze(6));
    }

    @Test public void permutationSharesDataAndComposesWithViews() {
        JTensor<Integer> a = JTensor.arange(24).reshape(new int[]{2,3,4});
        JTensor<Integer> p = a.permute(2,0,1);
        shape(p,4,2,3); assertEquals(a.item(1,2,3),p.item(3,1,2));
        p.setItem(new int[]{3,1,2},99); assertEquals(Integer.valueOf(99),a.item(1,2,3));
        assertEquals(a, p.permute(1,2,0));
        assertEquals(a.swapDimensions(0,2), a.swapAxes(-3,-1));
        invalid(() -> a.permute(0,0,2)); invalid(() -> a.permute(0,1));
        JTensor<Integer> slice = a.slice(new int[][]{{1,2},{0,3},{1,4}});
        assertEquals(slice.transpose(), slice.permute(2,1,0));
    }

    @Test public void contiguityAndCopiesRespectPhysicalOrder() {
        JTensor<Integer> a = JTensor.arange(12).reshape(new int[]{3,4});
        assertTrue(a.isContiguous()); assertSame(a,a.contiguous());
        JTensor<Integer> p = a.transpose(); assertFalse(p.isContiguous());
        JTensor<Integer> c = p.contiguous(); assertTrue(c.isContiguous()); assertEquals(p,c);
        c.setItem(new int[]{0,0},90); assertEquals(Integer.valueOf(0),a.item(0,0));
        assertTrue(a.slice(new int[][]{{1,3},{0,4}}).isContiguous());
        assertFalse(a.slice(new int[][]{{0,3},{1,3}}).isContiguous());
        JTensor<Integer> copy = a.copy(); copy.setItem(new int[]{0,0},42);
        assertEquals(Integer.valueOf(0),a.item(0,0));
        assertTrue(JTensor.empty(Integer.class).isContiguous());
    }

    @Test public void reductionsAllSingleAndMultipleAxes() {
        JTensor<Integer> a = JTensor.arange(1,25).reshape(new int[]{2,3,4});
        values(a.sum(),300); shape(a.sum(),1);
        values(a.sum(0,2),68,100,132); shape(a.sum(new int[]{0,2},true),1,3,1);
        values(a.mean(0,2),8,12,16);
        values(a.min(0,2),1,5,9); values(a.max(0,2),16,20,24);
        values(a.sum(-1),10,26,42,58,74,90);
        values(a.mean(),12); values(a.min(),1); values(a.max(),24);
        values(ints(1,2,3,4).prod(),24);
        values(ints(1,2,3,4).reshape(new int[]{2,2}).prod(0),3,8);
        assertEquals(a.sum(0,2),a.sum(2,0));
        assertEquals(a.sum(0,2),a.transpose().sum(0,2));
        invalid(() -> a.sum(0,-3)); invalid(() -> a.mean(3)); invalid(() -> a.sum((int[])null));
    }

    @Test public void everyReductionMatchesIndependentSubspaceReference() {
        JTensor<Double> a = JTensor.arange(1.0,25.0,1.0).reshape(new int[]{2,3,4}).swapAxes(0,2);
        for (int mask = 1; mask < 8; mask++) {
            List<Integer> selected = new ArrayList<>(), remaining = new ArrayList<>();
            for (int axis = 0; axis < 3; axis++) ( (mask & (1 << axis)) != 0 ? selected : remaining).add(axis);
            int[] axes = selected.stream().mapToInt(Integer::intValue).toArray();
            Map<List<Integer>,List<Double>> groups = new LinkedHashMap<>();
            Iterator<int[]> it = a.indicesIterator();
            while(it.hasNext()) {
                int[] ix = it.next(); List<Integer> key = new ArrayList<>();
                for(int axis : remaining) key.add(ix[axis]);
                groups.computeIfAbsent(key,k -> new ArrayList<>()).add(a.getItem(ix));
            }
            int row = 0;
            Double[] sum = a.sum(axes).toArray(), mean = a.mean(axes).toArray();
            Double[] min = a.min(axes).toArray(), max = a.max(axes).toArray(), prod = a.prod(axes).toArray();
            Integer[] argmin = a.argmin(axes).toArray(), argmax = a.argmax(axes).toArray();
            for(List<Double> group : groups.values()) {
                double s = 0, p = 1;
                for(double v : group) { s += v; p *= v; }
                assertEquals(s,sum[row],0); assertEquals(s/group.size(),mean[row],1e-12);
                assertEquals(Collections.min(group),min[row]); assertEquals(Collections.max(group),max[row]);
                assertEquals(p,prod[row],Math.abs(p)*1e-14);
                assertEquals(Integer.valueOf(group.indexOf(Collections.min(group))),argmin[row]);
                assertEquals(Integer.valueOf(group.indexOf(Collections.max(group))),argmax[row]); row++;
            }
            assertEquals(3,a.sum(axes,true).numberOfDimensions());
        }
    }

    @Test public void argReductionsUseFirstTieAndFlattenedSelectedCoordinates() {
        JTensor<Integer> a = ints(5,1,5,2,9,9).reshape(new int[]{2,3});
        values(a.argmax(),4); values(a.argmin(),1);
        values(a.argmax(1),0,1); values(a.argmin(0),1,0,0);
        shape(a.argmax(new int[]{0,1},true),1,1);
        values(doubles(1,Double.NaN,Double.NaN).argmax(),1);
        values(doubles(1,Double.NaN,Double.NaN).argmin(),1);
        values(doubles(Double.NEGATIVE_INFINITY,Double.NEGATIVE_INFINITY).argmax(),0);
    }

    @Test public void reductionsHaveDefinedEmptyIdentities() {
        JTensor<Integer> empty = JTensor.empty(Integer.class);
        values(empty.sum(),0); values(empty.prod(),1); values(empty.norm(),0.0);
        invalid(empty::mean); invalid(empty::min); invalid(empty::max); invalid(empty::argmax); invalid(empty::argmin);
    }

    @Test public void arithmeticBroadcastsAndPreservesType() {
        JTensor<Integer> a = ints(1,2,3,4).reshape(new int[]{2,2}), b = ints(10,20);
        values(a.add(b),11,22,13,24); values(a.subtract(b),-9,-18,-7,-16);
        values(a.multiply(b),10,40,30,80); values(a.divide(2),0,1,1,2);
        values(a.pow(2),1,4,9,16); values(a.mod(2),1,0,1,0); values(a.negate(),-1,-2,-3,-4);
        values(JTensor.from1DArray(Long.class,new Long[]{9007199254740993L}).add(1L),9007199254740994L);
        invalid(() -> a.add(ints(1,2,3)));
    }

    @Test public void comparisonsBroadcastAndHonorNaN() {
        JTensor<Integer> a = ints(1,2,3);
        values(a.isEqual(2),false,true,false); values(a.isNotEqual(2),true,false,true);
        values(a.isLessThan(2),true,false,false); values(a.isGreaterThan(2),false,false,true);
        values(a.isLessThanOrEqual(2),true,true,false); values(a.isGreaterThanOrEqual(2),false,true,true);
        values(doubles(Double.NaN).isEqual(Double.NaN),false);
        values(JTensor.from1DArray(String.class,new String[]{"a","b"}).isEqual("a"),true,false);
    }

    @Test public void unaryFunctionsMatchMathAndKeepShape() {
        JTensor<Double> a = doubles(0.25,0.5,1,2).reshape(new int[]{2,2}).transpose();
        JTensor<?>[] results = {a.sqrt(),a.exp(),a.log(),a.sin(),a.cos(),a.tanh(),a.floor(),a.ceil(),a.round(),a.abs(),a.negate()};
        for(int i=0;i<a.size();i++) {
            double x = a.toArray()[i];
            double[] expected = {Math.sqrt(x),Math.exp(x),Math.log(x),Math.sin(x),Math.cos(x),Math.tanh(x),
                Math.floor(x),Math.ceil(x),Math.rint(x),Math.abs(x),-x};
            for(int j=0;j<results.length;j++) {
                assertEquals(expected[j],((Number)results[j].toArray()[i]).doubleValue(),1e-12);
                shape(results[j],2,2);
            }
        }
        values(doubles(-2.5,-1.5,0.5,1.5,Double.NaN,Double.POSITIVE_INFINITY).round(),
            -2.0,-2.0,0.0,2.0,Double.NaN,Double.POSITIVE_INFINITY);
        values(ints(-3,2).abs(),3,2);
        values(doubles(0.0,-0.0).negate(),-0.0,0.0);
        values(JTensor.from1DArray(Float.class,new Float[]{0.0f,-0.0f}).negate(),-0.0f,0.0f);
        values(JTensor.from1DArray(Long.class,new Long[]{Long.MAX_VALUE}).floor(),Long.MAX_VALUE);
        assertTrue(Double.isNaN(doubles(-1).sqrt().item()));
    }

    @Test public void numericMethodsRejectNonNumbersAndNulls() {
        JTensor<String> strings = JTensor.singleValue("x");
        assertThrows(IllegalArgumentException.class,strings::sum);
        assertThrows(IllegalArgumentException.class,strings::exp);
        invalid(() -> new JTensor<>(Integer.class,new int[]{1}).sum());
        invalid(() -> ints(1).add((JTensor<Integer>)null));
    }

    @Test public void clipValidatesBoundsAndPropagatesNaN() {
        values(ints(-3,1,5).clip(0,3),0,1,3);
        values(doubles(Double.NaN,-2,5).clip(0.0,3.0),Double.NaN,0.0,3.0);
        invalid(() -> ints(1).clip(4,2)); invalid(() -> ints(1).clip(null,2));
    }

    @Test public void whereBroadcastsAllThreeOperands() {
        JTensor<Boolean> mask = JTensor.from1DArray(Boolean.class,new Boolean[]{true,false,true}).unsqueeze(0);
        JTensor<Integer> yes = ints(10,20).unsqueeze(1);
        JTensor<Integer> result = JTensor.where(mask,yes,ints(-1));
        shape(result,2,3); values(result,10,-1,10,20,-1,20);
        invalid(() -> JTensor.where(JTensor.singleValue(Boolean.class,null),ints(1),ints(2)));
    }

    @Test public void maskedSelectBroadcastsAndFlattensViews() {
        JTensor<Integer> a = ints(1,2,3,4,5,6).reshape(new int[]{2,3}).transpose();
        values(a.maskedSelect(JTensor.from1DArray(Boolean.class,new Boolean[]{true,false})),1,2,3);
        assertEquals(0,a.maskedSelect(JTensor.singleValue(false)).size());
        invalid(() -> a.maskedSelect(JTensor.singleValue(Boolean.class,null)));
    }

    @Test public void takeAndGatherHandleDuplicatesAndNegativeIndices() {
        JTensor<Integer> a = ints(1,2,3,4,5,6).reshape(new int[]{2,3});
        values(a.take(5,0,-1),6,1,6);
        JTensor<Integer> selected = a.take(new int[]{2,0,2},-1);
        shape(selected,2,3); values(selected,3,1,3,6,4,6);
        JTensor<Integer> indices = ints(2,0,0,-1).reshape(new int[]{2,2});
        values(a.gather(indices,1),3,1,4,6);
        invalid(() -> a.take(6)); invalid(() -> a.gather(ints(0),0));
        assertEquals(0,a.take(new int[0],0).size());
    }

    @Test public void stackAndJoinsHaveExpectedRanks() {
        JTensor<Integer> a = ints(1,2), b = ints(3,4);
        shape(JTensor.stack(a,b),2,2); values(JTensor.stack(1,a,b),1,3,2,4);
        values(JTensor.stack(-1,a,b),1,3,2,4);
        values(JTensor.verticalStack(a,b),1,2,3,4); shape(JTensor.verticalStack(a,b),2,2);
        shape(JTensor.horizontalStack(a,b),4);
        values(JTensor.horizontalStack(a.unsqueeze(1),b.unsqueeze(1)),1,3,2,4);
        invalid(() -> JTensor.stack(a,ints(3))); invalid(() -> JTensor.stack());
    }

    @Test public void splitsAndChunksCoverEachElementAndShareViews() {
        JTensor<Integer> a = JTensor.arange(10).reshape(new int[]{2,5});
        List<JTensor<Integer>> parts = a.split(new int[]{2,3},1);
        values(parts.get(0),0,1,5,6); values(parts.get(1),2,3,4,7,8,9);
        parts.get(0).setItem(new int[]{0,0},99); assertEquals(Integer.valueOf(99),a.item(0,0));
        List<JTensor<Integer>> chunks = a.chunk(3,-1);
        assertEquals(3,chunks.size()); shape(chunks.get(2),2,1);
        assertEquals(5,a.chunk(20,1).size()); assertEquals(2,a.split(2,0).size());
        invalid(() -> a.split(3,1)); invalid(() -> a.split(new int[]{2,2},1));
        invalid(() -> a.split(new int[]{0,5},1)); invalid(() -> a.chunk(0));
    }

    @Test public void linearAlgebraHandlesVectorsMatricesAndBatches() {
        JTensor<Integer> a = ints(1,2,3,4,5,6).reshape(new int[]{2,3});
        JTensor<Integer> b = ints(7,8,9,10,11,12).reshape(new int[]{3,2});
        values(a.matrixMultiply(b),58,64,139,154);
        values(ints(1,2,3).dotProduct(ints(4,5,6)),32);
        values(ints(1,2,3).matrixMultiply(ints(4,5,6)),32);
        values(a.matrixMultiply(ints(1,2,3)),14,32);
        values(ints(1,2).matrixMultiply(a),9,12,15);
        values(ints(1,2).outerProduct(ints(3,4,5)),3,4,5,6,8,10);
        JTensor<Integer> batch = JTensor.stack(a,a.multiply(2));
        shape(batch.matrixMultiply(b),2,2,2);
        values(batch.matrixMultiply(b),58,64,139,154,116,128,278,308);
        assertEquals(a.matrixMultiply(b),b.transpose().matrixMultiply(a.transpose()).transpose());
        invalid(() -> a.matrixMultiply(a)); invalid(() -> ints(1).dotProduct(ints(1,2)));
    }

    @Test public void matrixMultiplyBroadcastsDifferentBatchRanks() {
        JTensor<Integer> a = JTensor.ones(Integer.class,new int[]{2,1,3,4});
        JTensor<Integer> b = JTensor.full(Integer.class,new int[]{5,4,2},2);
        JTensor<Integer> result = a.matrixMultiply(b); shape(result,2,5,3,2);
        for(Integer value:result.toArray()) assertEquals(Integer.valueOf(8),value);
        invalid(() -> JTensor.ones(Integer.class,new int[]{2,3,4})
            .matrixMultiply(JTensor.ones(Integer.class,new int[]{3,4,2})));
    }

    @Test public void diagonalsAndTracesSupportOffsetsAndAxisPairs() {
        JTensor<Integer> a = JTensor.arange(1,13).reshape(new int[]{3,4});
        values(a.diagonal(),1,6,11); values(a.diagonal(1,0,1),2,7,12);
        values(a.diagonal(-1,0,1),5,10); values(a.trace(),18);
        values(a.trace(1,0,1),21); values(a.trace(99,0,1),0);
        assertEquals(0,a.diagonal(Integer.MIN_VALUE,0,1).size());
        JTensor<Integer> b = JTensor.arange(24).reshape(new int[]{2,3,4});
        shape(b.diagonal(0,0,2),3,2); values(b.diagonal(0,0,2),0,13,4,17,8,21);
        values(b.trace(0,0,2),13,21,29); values(b.trace(100,0,2),0,0,0);
        invalid(() -> a.diagonal(0,0,0));
    }

    @Test public void normIsStableAndSupportsSubsets() {
        values(ints(3,4).norm(),5.0);
        JTensor<Double> a = doubles(3,0,4,0).reshape(new int[]{2,2});
        values(a.norm(0),5.0,0.0); shape(a.norm(0,true),1,2);
        assertEquals(1e300*Math.sqrt(2),doubles(1e300,1e300).norm().item(),1e285);
        assertEquals(Math.sqrt(14),ints(1,2,3).norm().item(),1e-12);
    }

    @Test public void sortingSupportsAxesSubsetsAndStableIndices() {
        JTensor<Integer> a = ints(3,1,2,6,4,5).reshape(new int[]{2,3});
        values(a.sort(),1,2,3,4,5,6); values(a.argsort(),1,2,0,4,5,3);
        values(a.sort(-1),1,2,3,4,5,6); shape(a.sort(-1),2,3);
        values(a.argsort(1),1,2,0,1,2,0);
        values(ints(2,1,2,1).argsort(),1,3,0,2);
        JTensor<Integer> cube = ints(8,1,7,2,6,3,5,4).reshape(new int[]{2,2,2});
        values(cube.sort(0,2),1,3,2,4,6,8,5,7);
        values(cube.argsort(0,2),1,3,1,3,2,0,2,0);
        assertEquals(cube.sort(0,2),cube.sort(2,0));
        values(doubles(Double.NaN,1,Double.NEGATIVE_INFINITY).sort(),Double.NEGATIVE_INFINITY,1.0,Double.NaN);
        invalid(() -> a.sort(0,0));
        assertEquals(0,JTensor.empty(Integer.class).argsort().size());
    }

    @Test public void uniquePreservesFirstOccurrenceAndCanDeduplicateSlices() {
        values(ints(3,1,3,2,1).unique(),3,1,2);
        JTensor<Integer> a = ints(1,2,1,2,3,4).reshape(new int[]{3,2});
        shape(a.unique(0),2,2); values(a.unique(0),1,2,3,4);
        assertEquals(a.unique(0).transpose(),a.transpose().unique(1));
        values(JTensor.from1DArray(String.class,new String[]{"a",null,"a",null}).unique(),"a",null);
    }

    @Test public void factoriesHaveDefinedEndpointsTypesAndSeeds() {
        values(JTensor.zeros(Integer.class,new int[]{2}),0,0);
        values(JTensor.ones(Double.class,new int[]{2}),1.0,1.0);
        values(JTensor.full(new int[]{2},"x"),"x","x");
        values(JTensor.eye(Integer.class,2),1,0,0,1);
        values(JTensor.eye(Integer.class,2,3,1),0,1,0,0,0,1);
        values(JTensor.arange(5),0,1,2,3,4); values(JTensor.arange(5,-1,-2),5,3,1);
        values(JTensor.arange(0.0,1.0,0.25),0.0,0.25,0.5,0.75);
        assertEquals(0,JTensor.arange(0,5,-1).size()); invalid(() -> JTensor.arange(0,5,0));
        values(JTensor.linspace(0,1,3),0.0,0.5,1.0);
        values(JTensor.linspace(0,1,4,false),0.0,0.25,0.5,0.75);
        values(JTensor.linspace(2,9,1),2.0); assertEquals(0,JTensor.linspace(0,1,0).size());
        invalid(() -> JTensor.linspace(0,1,-1)); invalid(() -> JTensor.arange(0.0,1.0,Double.NaN));
        JTensor<Double> random = JTensor.random(new Random(7),2,3);
        assertEquals(random,JTensor.random(new Random(7),2,3)); shape(random,2,3);
        for(double v:random.toArray()) assertTrue(v>=0 && v<1);
    }

    @Test public void contractionsSupportOneMultipleAllAndNoAxes() {
        JTensor<Integer> a = JTensor.arange(1,7).reshape(new int[]{2,3});
        JTensor<Integer> b = JTensor.arange(7,13).reshape(new int[]{3,2});
        assertEquals(a.matrixMultiply(b),a.dotProduct(b,1,0));
        values(a.dotProduct(a,new int[]{0,1},new int[]{0,1}),91);
        values(a.dotProduct(a.transpose(),new int[]{0,1},new int[]{1,0}),91);
        JTensor<Integer> cube = JTensor.ones(Integer.class,new int[]{2,3,4});
        JTensor<Integer> other = JTensor.full(Integer.class,new int[]{4,5,2},2);
        JTensor<Integer> result = cube.dotProduct(other,new int[]{0,2},new int[]{2,0});
        shape(result,3,5);
        for(Integer value:result.toArray()) assertEquals(Integer.valueOf(16),value);
        JTensor<Integer> outer = ints(1,2).dotProduct(ints(3,4),new int[0],new int[0]);
        values(outer,3,4,6,8);
        invalid(() -> a.dotProduct(b,new int[]{0,0},new int[]{1,1}));
        invalid(() -> a.dotProduct(b,0,0));
    }

    @Test public void meansDoNotOverflowSmallTypeCountsOrIntegralSums() {
        values(JTensor.ones(Byte.class,new int[]{256}).mean(),(byte)1);
        values(JTensor.from1DArray(Long.class,new Long[]{Long.MAX_VALUE,Long.MAX_VALUE}).mean(),Long.MAX_VALUE);
        values(ints(-2,-1).mean(),-1);
        assertEquals(ints(1,2).sum(0,true),ints(1,2).sum(0,true));
        shape(ints(1,2).norm(new int[]{0},true),1);
    }

    @Test
    public void allAdmissibleNumericTypesAreSupportedByNewOperations() {
        assertNumericTypeSupport(
                JTensor.from1DArray(Byte.class, new Byte[]{1, 2, 4}),
                (byte) 1,
                (byte) 2,
                (byte) 4);
        assertNumericTypeSupport(
                JTensor.from1DArray(Short.class, new Short[]{1, 2, 4}),
                (short) 1,
                (short) 2,
                (short) 4);
        assertNumericTypeSupport(
                JTensor.from1DArray(Integer.class, new Integer[]{1, 2, 4}),
                1,
                2,
                4);
        assertNumericTypeSupport(
                JTensor.from1DArray(Long.class, new Long[]{1L, 2L, 4L}),
                1L,
                2L,
                4L);
        assertNumericTypeSupport(
                JTensor.from1DArray(Float.class, new Float[]{1F, 2F, 4F}),
                1F,
                2F,
                4F);
        assertNumericTypeSupport(
                JTensor.from1DArray(Double.class, new Double[]{1D, 2D, 4D}),
                1D,
                2D,
                4D);
    }

    private static <T extends Number> void assertNumericTypeSupport(
            JTensor<T> tensor,
            T one,
            T two,
            T four) {
        JTensor<T> scalar = JTensor.singleValue(tensor.getType(), two);

        assertElementType(tensor.add(scalar), tensor.getType());
        assertElementType(tensor.subtract(scalar), tensor.getType());
        assertElementType(tensor.multiply(scalar), tensor.getType());
        assertElementType(tensor.divide(scalar), tensor.getType());
        assertElementType(tensor.pow(scalar), tensor.getType());
        assertElementType(tensor.mod(scalar), tensor.getType());
        assertElementType(tensor.negate(), tensor.getType());
        assertElementType(tensor.abs(), tensor.getType());
        assertElementType(tensor.sqrt(), tensor.getType());
        assertElementType(tensor.exp(), tensor.getType());
        assertElementType(tensor.log(), tensor.getType());
        assertElementType(tensor.sin(), tensor.getType());
        assertElementType(tensor.cos(), tensor.getType());
        assertElementType(tensor.tanh(), tensor.getType());
        assertElementType(tensor.floor(), tensor.getType());
        assertElementType(tensor.ceil(), tensor.getType());
        assertElementType(tensor.round(), tensor.getType());
        assertElementType(tensor.clip(one, four), tensor.getType());

        assertElementType(tensor.sum(), tensor.getType());
        assertElementType(tensor.mean(), tensor.getType());
        assertElementType(tensor.min(), tensor.getType());
        assertElementType(tensor.max(), tensor.getType());
        assertElementType(tensor.prod(), tensor.getType());
        assertElementType(tensor.outerProduct(tensor), tensor.getType());

        JTensor<T> matrix1 = JTensor.repeat(
                tensor.getType(),
                new int[]{1, 2},
                one);
        JTensor<T> matrix2 = JTensor.repeat(
                tensor.getType(),
                new int[]{2, 1},
                two);
        assertElementType(matrix1.matrixMultiply(matrix2), tensor.getType());

        assertElementType(
                JTensor.zeros(tensor.getType(), new int[]{2}),
                tensor.getType());
        assertElementType(
                JTensor.ones(tensor.getType(), new int[]{2}),
                tensor.getType());
        assertElementType(JTensor.eye(tensor.getType(), 2), tensor.getType());

        assertElementType(tensor.isEqual(scalar), Boolean.class);
        assertElementType(tensor.isLessThan(scalar), Boolean.class);
        assertElementType(tensor.argmax(), Integer.class);
        assertElementType(tensor.norm(), Double.class);
    }

    private static void assertElementType(JTensor<?> tensor, Class<?> expectedType) {
        assertSame(expectedType, tensor.getType());
        for (Object value : tensor.toArray()) {
            if (value != null) {
                assertSame(expectedType, value.getClass());
            }
        }
    }

    @Test
    public void integralPowerPreservesLongPrecisionAndWrapperOverflow() {
        JTensor<Long> preciseLong =
                JTensor.from1DArray(Long.class, new Long[]{9007199254740993L});
        values(preciseLong.pow(1L), 9007199254740993L);
        values(JTensor.pow(preciseLong, JTensor.singleValue(1L)), 9007199254740993L);

        values(JTensor.singleValue(Long.MAX_VALUE).pow(2L), 1L);
        values(JTensor.singleValue(-1L).pow(-Long.MAX_VALUE), -1L);
        values(JTensor.singleValue(-1L).pow(Long.MIN_VALUE), 1L);
        values(JTensor.singleValue(2).pow(-1), 0);
        values(JTensor.singleValue(4F).pow(0.5F), 2F);
        values(JTensor.singleValue(4D).pow(0.5D), 2D);
    }
}
