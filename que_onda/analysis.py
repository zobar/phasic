import tensorflow as tf

def rms(tensor, axis=None):
    square = tf.pow(tensor, 2)
    mean = tf.reduce_mean(square, axis)
    return tf.sqrt(mean)

def rms_by_level(levels):
    rmses = [rms(l) for l in levels]
    return tf.stack(rmses)

def rms_by_frame_and_level(frames):
    rmses = [rms(l, axis=[1, 2]) for l in frames]
    return tf.stack(rmses, axis=1)

def rms_by_frame(by_frame_and_level):
    return rms(by_frame_and_level, axis=1)


# Don't use squared distance- we really need a linear metric if we're converting to int for or-tools
# https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/distances/euclidean.py#L42
def distance_matrix(frame_level_rms):
    squared_norm = tf.math.square(frame_level_rms)
    squared_norm = tf.math.reduce_sum(squared_norm, axis=1, keepdims=True)

    distances = 2.0 * tf.linalg.matmul(frame_level_rms, frame_level_rms, transpose_b=True)
    distances = squared_norm - distances + tf.transpose(squared_norm)

    dist_mask = tf.math.greater_equal(distances, 1e-18)
    distances = tf.math.maximum(distances, 1e-18)
    distances = tf.math.sqrt(distances) * tf.cast(dist_mask, tf.float32)

    return distances
