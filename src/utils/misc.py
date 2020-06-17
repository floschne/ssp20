import itertools as it

import numpy as np
import scipy.signal as ss
import sounddevice as sd

from src.audiosignal import AudioSignal
from src.utils import ms_to_idx, hz_to_idx


def compute_residual_signal(lpc: np.array, frame: np.array):
    return ss.lfilter(np.concatenate(([1], lpc)), 1, frame)


def pre_emphasize(s: np.ndarray, alpha: np.float32 = .95):
    return ss.lfilter(np.array([1, -alpha]), 1, s)


def compute_power_in_time_domain(s: np.ndarray):
    """
    Computes the power of a signal in the time domain
    :param s: the discrete time domain signal or an array of discrete time signals
    :return: the power of the signal
    """
    if s.ndim == 2:
        power = np.mean(s ** 2, axis=1)
        assert power.ndim == 1 and power.shape[0] == s.shape[0]
        return power
    else:
        return np.mean(s ** 2)


def compute_energy_in_time_domain(s: np.ndarray):
    """
    Computes the energy of a signal in the time domain
    :param s: the discrete time domain signal or an array of discrete time signals
    :return: the energy of the signal
    """
    return np.sum(s ** 2)


def compute_energy_in_freq_domain(S: np.ndarray):
    """
    Computes the energy of a signal in the spectral domain
    :param S: the spectrum of s
    :return: the energy of the signal
    """
    return np.sum(np.abs(S) ** 2) / len(S)


def __count_zero_crossings(s: np.ndarray):
    def count(x: np.ndarray):
        return len(list(it.groupby(x, lambda x_n: x_n > 0))) / len(x)

    if s.ndim == 2:
        return np.apply_along_axis(count, 1, s)
    else:
        return count(s)


def is_voiced(frames: np.ndarray, threshold=0.3):
    def voiced(cross: np.ndarray):
        return np.where(cross <= threshold, 1, 0)

    crossings = __count_zero_crossings(frames)

    if crossings.ndim == 2:
        return np.apply_along_axis(voiced, 1, crossings)
    else:
        return voiced(crossings)


def play(signal: np.ndarray, fs_hz: int = 16000):
    sd.play(signal, fs_hz)


def create_excitation_signal(length_ms: int = 32, fs_hz: int = 8000, f0_hz: int = 200, voiced: bool = True):
    ex = None
    length_idx = ms_to_idx(length_ms, fs_hz)
    f0_idx = hz_to_idx(f0_hz, fs_hz)
    if voiced:
        ex = np.zeros(length_idx)
        ex[::f0_idx] = 1
    else:
        ex = np.random.randn(length_idx)

    return AudioSignal(path=None, data=ex, sampling_freq_hz=fs_hz)
