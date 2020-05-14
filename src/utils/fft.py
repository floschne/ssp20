import numpy as np

from src.audiodata import AudioData
from src.utils import ms_to_idx


def compute_istft(stft: np.ndarray, sampling_freq_hz: int, frame_shift_ms: int,
                  synthesis_window: np.ndarray) -> AudioData:
    """
    Compute the inverse short-time Fourier transform.

    :param stft: STFT transformed signal
    :param sampling_freq_hz: the sampling rate of the origin time domain signal
    :param frame_shift_ms: the frame shift used to compute the STFT in milliseconds
    :param synthesis_window: a numpy array containing a synthesis window function (length must match with time domain
    signal segments that were used to compute the STFT)
    :return: a numpy array containing the time domain signal
    """

    # compute inverse rFFT and apply synthesis window
    time_frames = np.fft.irfft(stft)
    num_frames, samples_per_frame = time_frames.shape
    assert samples_per_frame == len(
        synthesis_window), "Synthesis window must match the number of samples per frame."
    time_frames *= synthesis_window

    # compute output size
    samples_per_shift = ms_to_idx(frame_shift_ms, sampling_freq_hz)
    output_len = samples_per_frame + (num_frames - 1) * samples_per_shift
    time_signal = np.zeros(output_len)

    # reconstruct signal by adding overlapping windowed segments
    for i in range(num_frames):
        time_signal[i * samples_per_shift:i * samples_per_shift + samples_per_frame] += time_frames[i]

    return AudioData(data=time_signal, sampling_freq_hz=sampling_freq_hz)
