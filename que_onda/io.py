from samplerate import resample
import soundfile as sf
import tensorflow as tf

vorbis_chunk_size = 128 * 1024


def read_channels(filename, dtype=tf.float32):
    np_samples, samplerate = sf.read(filename, dtype=dtype.name)
    tf_samples = tf.constant(np_samples)
    channels = tf.transpose(tf_samples)
    return channels, samplerate


def write_channels_vorbis(filename, channels, samplerate):
    samples = tf.transpose(channels)
    np_samples = samples.numpy()
    with sf.SoundFile(
        filename,
        mode="w",
        channels=channels.shape[0],
        format="OGG",
        samplerate=samplerate,
        subtype="VORBIS",
    ) as file:
        for i in range(0, np_samples.shape[0], vorbis_chunk_size):
            chunk = np_samples[i : i + vorbis_chunk_size]
            file.write(chunk)


def write_channels_opus(filename, channels, samplerate):
    samples = tf.transpose(channels)
    np_samples = resampled.numpy()
    resampled = resample(np_samples, 48000 / samplerate)
    sf.write(filename, resampled, 48000, format="OGG", subtype="OPUS")
