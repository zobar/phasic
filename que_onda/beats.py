import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import curve_fit
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


def logarithmic_decay(x, a, b, c):
    return a * np.log(x + c) + b

def find_length(frame_power, hint=(-0.084398925, 0.78620344, 0.07126941), max_beat_length=58, max_lags=188, min_beat_length=11, prominence=0.008, width=2):
    offsets = np.arange(0, max_lags + 1)
    tf_correlation = tfp.stats.auto_correlation(frame_power, max_lags=max_lags)
    correlation = tf_correlation.numpy()

    try:
        popt, _ = curve_fit(logarithmic_decay, offsets, correlation, p0=hint)
    except (RuntimeError, ValueError) as e:
        print(f'!!! Could not fit curve: {e}')
        corrected_correlation = correlation
    else:
        baseline = logarithmic_decay(offsets, *popt)
        corrected_correlation = correlation - baseline

    all_peaks, _ = find_peaks(corrected_correlation, prominence=prominence, width=width)
    in_range = (all_peaks >= min_beat_length) & (all_peaks <= max_beat_length)
    return np.compress(in_range, all_peaks)

def beat_match(beat_lengths):
    expanded_beat_lengths = tf.expand_dims(beat_lengths, axis=-1)
    candidate_beat_lengths = tf.range(tf.reduce_min(beat_lengths), tf.reduce_max(beat_lengths) + 1)
    errors = expanded_beat_lengths - candidate_beat_lengths
    squared_errors = tf.pow(errors, 2)
    min_squared_errors = tf.reduce_min(squared_errors, axis=1)
    sum_squared_errors = tf.reduce_sum(min_squared_errors, axis=0)
    best_beat_length_index = tf.argmin(sum_squared_errors)
    best_beat_length = candidate_beat_lengths[best_beat_length_index]

    min_squared_error_index = tf.map_fn(tf.argmin, squared_errors, dtype=tf.int64)
    closest_beat_length_indexes = tf.gather(min_squared_error_index, best_beat_length_index, axis=1)
    closest_beat_lengths = tf.gather(beat_lengths, closest_beat_length_indexes, batch_dims=1)

    return best_beat_length, closest_beat_lengths

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
