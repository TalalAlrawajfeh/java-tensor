import com.tensor.JTensor;

public class Main {

    private static JTensor<Integer> removeWhiteBorder(JTensor<Integer> tensor) {
        var black = JTensor.compare(tensor, JTensor.singleValue(0))
                .map(Integer.class, x -> x == 0 ? 1 : 0);

        var alongY = black.reduceAlong(0,
                (x, y) -> x | y,
                1,
                false);

        var alongX = black.reduceAlong(0,
                (x, y) -> x | y,
                0,
                false);

        var shape = tensor.getShape();
        var height = shape[0];
        var width = shape[1];

        int x1 = JTensor.argMax(alongX, 0, true).getItem(new int[]{0});
        int y1 = JTensor.argMax(alongY, 0, true).getItem(new int[]{0});
        int x2 = width - JTensor.argMax(alongX.reverse(0), 0, true).getItem(new int[]{0});
        int y2 = height - JTensor.argMax(alongY.reverse(0), 0, true).getItem(new int[]{0});

        return tensor.slice(new int[][]{{y1, y2}, {x1, x2}});
    }


    public static void main(String[] args) {
        Tensor<Integer> tensor = Tensor.ones(Integer.class, new int[]{2, 2, 2});
        System.out.println(tensor);
    }
}
