import tensorflow as tf
import tensorflow_probability as tfp


def frames_to_levels(frames):
    """Flatten framed levels.

    Input dimensions:
        - frames
        - batches
        - frame length
    """
    levels = []
    for framed in frames:
        shape = list(framed.shape)
        new_shape = shape[1:-1] + [-1]
        level = tf.reshape(framed, new_shape)
        levels.append(level)
    return levels


def levels_to_frames(levels):
    """Convert flat levels (where each level is a different length) to framed levels (where each level has the same number of frames).

    Input dimensions:
        - batches
        - level length
    Output dimensions:
        - frames
        - batches
        - frame length
    """
    frame_count = levels[0].shape[1]
    frames = []
    for level in levels:
        shape = list(level.shape)
        new_shape = [frame_count] + shape[:-1] + [-1]
        framed = tf.reshape(level, new_shape)
        frames.append(framed)
    return frames


def stretch_linear(new_length, frames):
    """Time-stretch framed levels, using linear interpolation.

    To lengthen n times with precise alignment, new_length = (n * frame_length) + 1"""

    dtype = frames[0].dtype
    old_length = frames[0].shape[0]
    x = tf.linspace(
        tf.constant(0, dtype=dtype), tf.constant(old_length, dtype=dtype), new_length
    )

    new_frames = []
    for level in frames:
        new = tfp.math.interp_regular_1d_grid(
            x=x, x_ref_min=0, x_ref_max=old_length, y_ref=level, axis=0
        )
        new_frames.append(new)
    return new_frames


def stretch_neighbor(new_length, frames):
    """Time-stretch framed levels. No interpolation is performed- this uses the previous neighbor.

    To lengthen n times with precise alignment, new_length = (n * frame_length) - 1"""
    old_length = frames[0].shape[0]
    offsets = tf.linspace(
        tf.constant(0, dtype=tf.float32),
        tf.constant(old_length - 1, dtype=tf.float32),
        new_length,
    )
    indexes = tf.cast(offsets, dtype=tf.int32)
    return [tf.gather(l, indexes) for l in frames]
