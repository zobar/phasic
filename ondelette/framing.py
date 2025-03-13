import tensorflow as tf

def frames_to_levels(frames):
    levels = []
    for framed in frames:
        shape = list(framed.shape)
        new_shape = shape[1:-1] + [-1]
        level = tf.reshape(framed, new_shape)
        levels.append(level)
    return levels

def levels_to_frames(levels):
    frame_count = levels[0].shape[1]
    frames = []
    for level in levels:
        shape = list(level.shape)
        new_shape = [frame_count] + shape[:-1] + [-1]
        framed = tf.reshape(level, new_shape)
        frames.append(framed)
    return frames

def stretch_tensor(length, tensor):
    old_length = tensor.shape[0]
    factor = old_length / length
    indexes = tf.range(0, length, dtype=tf.float32)
    scaled = indexes * factor
    int_indexes = tf.cast(scaled, dtype=tf.int32)
    return tf.gather(tensor, int_indexes)

def stretch(length, frames):
    return [stretch_tensor(length, l) for l in frames]
