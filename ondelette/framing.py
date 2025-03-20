import tensorflow as tf

def frames_to_levels(frames):
    '''Flatten framed levels.

    Input dimensions:
        - frames
        - batches
        - frame length
    '''
    levels = []
    for framed in frames:
        shape = list(framed.shape)
        new_shape = shape[1:-1] + [-1]
        level = tf.reshape(framed, new_shape)
        levels.append(level)
    return levels

def levels_to_frames(levels):
    '''Convert flat levels (where each level is a different length) to framed levels (where each level has the same number of frames).

    Input dimensions:
        - batches
        - level length
    Output dimensions:
        - frames
        - batches
        - frame length
    '''
    frame_count = levels[0].shape[1]
    frames = []
    for level in levels:
        shape = list(level.shape)
        new_shape = [frame_count] + shape[:-1] + [-1]
        framed = tf.reshape(level, new_shape)
        frames.append(framed)
    return frames

def stretch_frame(factor, frame):
    old_length = frame.shape[0]
    indexes = tf.range(0, old_length * factor, dtype=tf.float32)
    scaled = indexes / factor
    int_indexes = tf.cast(scaled, dtype=tf.int32)
    return tf.gather(frame, int_indexes)

def stretch_frames(length, frames):
    '''Time-stretch framed levels. There's no interpolation- this uses the previous neighbor.

    Investigate using https://www.tensorflow.org/probability/api_docs/python/tfp/math/interp_regular_1d_grid instead.
    '''
    return [stretch_frame(length, l) for l in frames]

# This transform sounds terrible.
def stretch_level(factor, level):
    old_length = level.shape[1]
    indexes = tf.range(0, old_length * factor, dtype=tf.float32)
    scaled = indexes / factor
    int_indexes = tf.cast(scaled, dtype=tf.int32)
    return tf.gather(level, int_indexes, axis=1)

def stretch_levels(factor, levels):
    return [stretch_level(factor, l) for l in levels]
