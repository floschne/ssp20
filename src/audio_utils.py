import numpy as np


def ms_to_idx(ms: int, fs: np.float):
    """
    :param ms: milliseconds
    :param fs: sampling frequency in Hertz
    """
    return ms * 1000 * fs


def s_to_idx(s: np.float, fs: np.float):
    """
    :param s: seconds
    :param fs: sampling frequency in Hertz
    """
    return s * fs


def hz_to_ms(hz: np.float):
    """
    :param hz: Frequency in Hertz
    """
    return 1000 / hz


def hz_to_s(hz: np.float):
    """
    :param hz: Frequency in Hertz
    """
    return 1 / hz
