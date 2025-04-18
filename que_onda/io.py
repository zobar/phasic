import soundfile as sf
import tensorflow as tf


def read_channels(filename, dtype=tf.float32):
    np_samples, samplerate = sf.read(filename, dtype=dtype.name)
    tf_samples = tf.constant(np_samples)
    channels = tf.transpose(tf_samples)
    return channels, samplerate


def write_channels(filename, channels, samplerate):
    samples = tf.transpose(channels)
    np_samples = samples.numpy()
    sf.write(filename, np_samples, samplerate)
