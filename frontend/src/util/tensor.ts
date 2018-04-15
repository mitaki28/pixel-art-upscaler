import * as tf from "@tensorflow/tfjs";
export function upsample_nearest_neighbor(tensor: tf.Tensor) {
    const [h, w, c] = [tensor.shape[0], tensor.shape[1], tensor.shape[2], 1];
    return tensor
        .reshape([h, w, c, 1])
        .tile([1, 1, 1, 2])
        .reshape([h, w, c, 2])
        .transpose([0, 3, 1, 2])
        .reshape([2 * h, w, c, 1])
        .tile([1, 1, 1, 2])
        .reshape([2 * h, w, c, 2])
        .transpose([0, 1, 3, 2])
        .reshape([2 * h, 2 * w, c]);
}
export function downsample_nearest_neighbor(tensor: tf.Tensor) {
    const [h, w, c] = [tensor.shape[0], tensor.shape[1], tensor.shape[2], 1];
    return tensor
        .reshape([h / 2, 2, w, c])
        .transpose([0, 2, 3, 1])
        .reshape([h / 2, w, c, 2])
        .slice([0, 0, 0, 0], [h / 2, w, c, 1])
        .reshape([h / 2, w / 2, 2, c])
        .transpose([0, 1, 3, 2])
        .reshape([h / 2, w / 2, c, 2])
        .slice([0, 0, 0, 0], [h / 2, w / 2, c, 1])
        .reshape([h / 2, w / 2, c]);
}