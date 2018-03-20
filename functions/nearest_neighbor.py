import chainer.functions as F

def nearest_neighbor(x, r=2):
    return F.reshape(
        F.transpose(
            F.reshape(
                F.tile(x.reshape((x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1)), r * r),
                (x.shape[0], x.shape[1], x.shape[2], x.shape[3], r, r),
            ),
            (0, 1, 2, 4, 3, 5),
        ),
        (x.shape[0], x.shape[1], r * x.shape[2], r * x.shape[3]),
    )

if __name__ == '__main__':
    import math
    import numpy as np

    batchsize = 7
    r = 2
    n_channels = 3
    in_height = 4
    in_width = 5

    shape = (batchsize, n_channels, in_height, in_width)
    in_map = np.arange(np.prod(shape)).reshape(shape)

    print(in_map)
    out_map = nearest_neighbor(in_map, r).data
    print(out_map)

    # test
    expected_shape = (batchsize, n_channels, r * in_height, r * in_width)
    actual_shape = out_map.shape
    assert expected_shape == actual_shape, \
        'out_map shape: expected {}, but {}'.format(
            expected_shape,
            actual_shape,
        )
    for b in range(batchsize):
        for k in range(n_channels):
            for y in range(in_height):
                for x in range(in_width):
                    for i in range(r):
                        for j in range(r):
                            idx = (b, k, r * y + i, r * x + j)
                            expected_value_idx = (b, k, y, x)
                            expected_value = in_map[expected_value_idx]
                            actual_value = out_map[idx]
                            assert expected_value == actual_value, \
                                'expected <{};in_map[{}]> but <{};out_map[{}]>'.format(
                                    expected_value,
                                    expected_value_idx,
                                    actual_value,
                                    idx,
                                )
