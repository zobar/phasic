import tensorflow as tf
import tensorflow_probability as tfp


def frames_to_levels(frames):
    """Flatten framed levels.

    Input dimensions:
        - frames
        - frame length
        - audio channels
    Output dimensions:
        - audio channels
        - length
    """
    channel_count = frames[0].shape[2]
    levels = []
    for framed in frames:
        samples = tf.reshape(framed, [-1, channel_count])
        level = tf.transpose(samples)
        levels.append(level)
    return levels


def levels_to_frames(levels):
    """Convert flat levels (where each level is a different length) to framed levels (where each level has the same number of frames).

    Input dimensions:
        - audio channels
        - length
    Output dimensions:
        - frames
        - frame length
        - audio channels
    """
    channel_count, frame_count = levels[0].shape
    frames = []
    for level in levels:
        channels = tf.transpose(level)
        framed = tf.reshape(channels, [frame_count, -1, channel_count])
        frames.append(framed)
    return frames


def remix(frames, indexes):
    return [tf.gather(l, indexes) for l in frames]


def remix_linear(frames, offsets):
    x_ref_max = frames[0].shape[0] - 1
    new_frames = []
    for framed in frames:
        new_framed = tfp.math.interp_regular_1d_grid(
            x=offsets, x_ref_min=0, x_ref_max=x_ref_max, y_ref=framed, axis=0
        )
        new_frames.append(new_framed)
    return new_frames


def stretch_linear(new_length, frames):
    """Time-stretch framed levels, using linear interpolation.

    To lengthen n times with precise alignment, new_length = (n * frame_length) - 1"""

    dtype = frames[0].dtype
    x_ref_max = frames[0].shape[0] - 1
    return tf.linspace(
        start=tf.constant(0, dtype=dtype),
        stop=tf.constant(x_ref_max, dtype=dtype),
        num=new_length,
    )


def stretch_neighbor(new_length, frames):
    """Time-stretch framed levels. No interpolation is performed- this uses the previous neighbor.

    To lengthen n times with precise alignment, new_length = (n * frame_length) - 1"""
    offsets = stretch_linear(new_length, frames)
    return tf.cast(offsets, dtype=tf.int32)
