import numpy as np


def ms_to_idx(ms: int, fs: np.float) -> int:
    """
    :param ms: milliseconds
    :param fs: sampling frequency in Hertz
    """
    return int(ms * (10 ** -3) * fs)


def s_to_idx(s: np.float, fs: np.float) -> int:
    """
    :param s: seconds
    :param fs: sampling frequency in Hertz
    """
    return int(s * fs)


def idx_to_ms(idx: int, fs: np.float) -> int:
    """
    :param idx: sample index
    :param fs: sampling frequency in Hertz
    """
    return idx / (fs * (10 ** -3))


def idx_to_s(idx: int, fs: np.float) -> np.float:
    """
    :param idx: sample index
    :param fs: sampling frequency in Hertz
    """
    return idx / fs


def hz_to_ms(hz: np.float) -> int:
    """
    :param hz: Frequency in Hertz
    """
    return int(10 ** 3 / hz)


def hz_to_s(hz: np.float) -> np.float:
    """
    :param hz: Frequency in Hertz
    """
    return 1 / hz


def ms_to_hz(ms: int) -> np.float:
    """
    :param ms: milliseconds
    """
    return 1000 / ms


def s_to_hz(s: np.float) -> np.float:
    """
    :param s: seconds
    """
    return 1 / s
