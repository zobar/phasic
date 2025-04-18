import tensorflow as tf


def energy(tensor, axis=None):
    square = tf.pow(tensor, 2)
    return tf.reduce_sum(square, axis)


def energy_by_level(levels):
    energies = [energy(l) for l in levels]
    return tf.stack(energies)


def energy_by_frame_and_level(frames):
    energies = [energy(l, axis=[1, 2]) for l in frames]
    return tf.stack(energies, axis=1)


def power_by_frame(frame_level_energies):
    sum = tf.reduce_sum(frame_level_energies, axis=1)
    return tf.sqrt(sum)


# Don't use squared distance- we really need a linear metric if we're converting to int for or-tools
# https://github.com/tensorflow/similarity/blob/master/tensorflow_similarity/distances/euclidean.py#L42
def euclidean_distance_matrix(beat_level_power):
    squared_norm = tf.math.square(beat_level_power)
    squared_norm = tf.math.reduce_sum(squared_norm, axis=1, keepdims=True)

    distances = 2.0 * tf.linalg.matmul(
        beat_level_power, beat_level_power, transpose_b=True
    )
    distances = squared_norm - distances + tf.transpose(squared_norm)

    dist_mask = tf.math.greater_equal(distances, 1e-18)
    distances = tf.math.maximum(distances, 1e-18)
    distances = tf.math.sqrt(distances) * tf.cast(dist_mask, tf.float32)

    return distances


def manhattan_distance_matrix(beat_level_power):
    rs = tf.reshape(beat_level_power, shape=[tf.shape(beat_level_power)[0], -1])
    deltas = tf.expand_dims(rs, axis=1) - tf.expand_dims(rs, axis=0)
    distances = tf.norm(deltas, 1, axis=2)
    return distances
