import tensorflow as tf

def rms(batches):
    square = tf.pow(batches, 2)
    mean = tf.reduce_mean(square)
    return tf.sqrt(mean)

def rms_by_level(levels):
    rmses = [rms(batch) for batch in levels]
    return tf.stack(rmses)
