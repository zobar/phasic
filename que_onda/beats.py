import tensorflow as tf
import tensorflow_probability as tfp
from scipy.signal import find_peaks
from . import analysis


def frames_to_beats(beat_length, frames, offset=0):
    all_frames, _, n_channels = frames[0].shape
    n_beats = (all_frames - offset) // beat_length
    n_frames = beat_length * n_beats
    new_shape = [n_beats, beat_length, -1, n_channels]
    beats = []
    for framed in frames:
        cropped = framed[offset : n_frames + offset]
        beated = tf.reshape(cropped, new_shape)
        beats.append(beated)
    return beats


def beats_to_frames(beats):
    n_beats, beat_length, _, n_channels = beats[0].shape
    n_frames = n_beats * beat_length
    new_shape = [n_frames, -1, n_channels]
    frames = []
    for beated in beats:
        framed = tf.reshape(beated, new_shape)
        frames.append(framed)
    return frames


def find_length(
    frame_power,
    height=0,
    include=range(16, 52),  # 50-160 bpm
    max_lags=100,
    prominence=0.011,
    width=2,
):
    autocorrelation = tfp.stats.auto_correlation(frame_power, max_lags=max_lags)
    np_autocorrelation = autocorrelation.numpy()
    peaks, peak_properties = find_peaks(
        np_autocorrelation, height=height, prominence=prominence, width=width
    )
    included = [p for p in peaks if p in include]
    if len(included):
        return included[0]


def find_offset(frame_rms, beat_length):
    n_frames = frame_rms.shape[0]

    def try_offset(offset):
        n_beats = (n_frames - offset) // beat_length
        cut = frame_rms[offset : (n_beats * beat_length) + offset]
        intrabeat_rms = tf.reshape(cut, [-1, beat_length])
        means = tf.reduce_mean(intrabeat_rms, axis=0)
        min_offset = tf.argmin(means).numpy() + 1
        if min_offset != beat_length:
            return try_offset(offset + min_offset)
        return offset

    return try_offset(0)


def remix(beats, indexes):
    return [tf.gather(l, indexes) for l in beats]


def energy_by_beat_and_level(beats):
    energies = [analysis.energy(l, axis=[1, 2, 3]) for l in beats]
    return tf.stack(energies, axis=1)
