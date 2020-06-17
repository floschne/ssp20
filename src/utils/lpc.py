import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl
import scipy.signal as ss


def __auto_correlate(frames: np.array):
    def acf(frame: np.ndarray):
        return np.correlate(frame, frame, 'full')

    if frames.ndim == 2:
        return np.apply_along_axis(acf, 1, frames)
    else:
        return acf(frames)


def __compute_phi(frames: np.array, m: int = 12):
    def cp(frame: np.ndarray):
        acf = __auto_correlate(frame)
        phi_start = len(acf) // 2
        phi_end = phi_start + m
        phi = acf[phi_start:phi_end + 1]
        assert len(phi) == m + 1, f"len(phi)={len(phi)} != m+1={m + 1}"
        return phi

    if frames.ndim == 2:
        return np.apply_along_axis(cp, 1, frames)
    else:
        return cp(frames)


def compute_lpc(frames: np.array, m: int = 12):
    def cl(frame: np.array):
        phi = __compute_phi(frame, m)
        c = phi[:-1]
        r = phi[:-1]
        b = phi[1:]
        a = sl.solve_toeplitz((-c, -r), b)
        assert len(a) == m
        return a

    if frames.ndim == 2:
        return np.apply_along_axis(cl, 1, frames)
    else:
        return cl(frames)


def compute_complex_filter_frequency_response(lpc: np.array, n: int = 512, fs: int = 16000) -> (np.ndarray, np.ndarray):
    def freq_resp(c: np.ndarray):
        return ss.freqz(1, np.concatenate(([1], c)), n, whole=True, fs=fs)

    if lpc.ndim == 2:
        return np.apply_along_axis(freq_resp, 1, lpc)
    else:
        return freq_resp(lpc)


def plot_complex_filter_frequency_response(fresp: np.ndarray):
    w, h = fresp
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_title('Complex filter frequency response')
    ax1.plot(w, 10 * np.log10(np.abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [Hz]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid()
    ax2.axis('tight')
