import matplotlib.pyplot as plt
import numpy as np

from src.audiodata import AudioData
from src.utils import ms_to_idx, idx_to_ms


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


def compute_stft(frames: np.array, synthesis_window: np.ndarray, sampling_freq: int = 16000) -> [np.ndarray,
                                                                                                 np.ndarray]:
    """
    :param frames: MxN matrix of frame data. M := number of frames, N := length of each frame in samples
    :param synthesis_window: a numpy array containing a synthesis window function (length must match with time domain
    signal segments that were used to compute the STFT)
    :param sampling_freq: the sampling frequency of each of the frames
    :return: stft, freq_axis_hz
    """
    assert frames.shape[1] == synthesis_window.shape[0], "Frame length and window length have to be the same!"

    # apply analysis window by multiplying with each frame
    windowed = synthesis_window * frames

    # FFT on each windowed segment
    stft = np.fft.fft(windowed, axis=1)

    # discard lower half including the nyquist bin
    N = stft.shape[1] // 2 + 1
    stft = stft[:, :N]

    # Nyquist frequency
    nyquist_freq_hz = sampling_freq / 2

    # freq axis in Hz
    freq_axis_hz = np.linspace(0, nyquist_freq_hz, N)

    return stft, freq_axis_hz


def plot_stft(
        stft: np.ndarray,
        duration_ms: int,
        freq_axis_hz: np.ndarray,
        sampling_freq: int = 16000,
        return_plot: bool = False):
    fig, axs = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [10, 0.5]})

    # no margins
    axs[0].margins(x=0.0)
    axs[1].margins(x=0.0)

    # plot stft
    axs[0].set_title(f"Frame Length [ms]: {idx_to_ms(stft.shape[1], sampling_freq)}")
    axs[0].set_ylabel('Frequency [Hz]')
    axs[0].set_xlabel('Time [s]')
    tick_size = (duration_ms / 1000) / 80
    ticks = np.arange(0., (duration_ms / 1000) + tick_size, tick_size)
    axs[0].set_xticks(ticks)
    plt.setp(axs[0].get_xticklabels(), rotation=90)

    im = axs[0].imshow(10 * np.log10(np.maximum(np.square(np.abs(stft.T)), 10 ** (-15))),
                       cmap='viridis',
                       origin='lower',
                       extent=[ticks[0], ticks[-1], freq_axis_hz[0], freq_axis_hz[-1]],
                       aspect='auto')

    fig.colorbar(im, cax=axs[1], orientation="horizontal")

    plt.tight_layout()

    plt.show()

    if return_plot:
        return fig, axs
