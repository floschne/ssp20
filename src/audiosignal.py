import librosa as rosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
import sounddevice as sd

from src.utils import conversion


class AudioSignal:

    def __init__(self,
                 path: str = None,
                 data: np.ndarray = None,
                 sampling_freq_hz: int = None):

        if path is not None:
            self.path = path
            print(self.path)
            self.data, self.sampling_freq = rosa.load(path, sr=sampling_freq_hz, mono=False)

            self.channels = 1
            if len(self.data.shape) == 2:
                self.channels = self.data.shape[0]

            print('Successfully loaded audio signal with %s channel(s) from file: %s' % (self.channels, self.path))
        else:
            self.path = None
            assert data is not None
            assert sampling_freq_hz is not None
            self.data = data
            self.sampling_freq = sampling_freq_hz
            self.channels = 1
            print('Successfully loaded audio signal with 1 channel!')

        self.duration_s = len(self.__get_channel_data(0)) / self.sampling_freq

    def plot(self,
             start_s: np.float = None,
             stop_s: np.float = None,
             num_ticks: int = 80,
             return_plot=False,
             channel: int = 0):

        assert stop_s is None or stop_s <= start_s + self.duration_s

        if stop_s is None:
            stop_s = self.duration_s

        fig, axs = None, None
        # init plot
        if start_s is None:
            fig, axs = plt.subplots(figsize=(20, 10))
        else:
            fig, axs = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [2, 10]})

        if start_s is not None:
            # crop signal
            cropped_data, cropped_duration = self.crop(start_s, stop_s, channel=channel)
            cropped_t = np.linspace(start_s, start_s + cropped_duration, len(cropped_data))

            # x-axis ticks
            axs[0].set_xlabel('time [s]')
            tick_size = self.duration_s / num_ticks
            ticks = np.arange(0., self.duration_s + tick_size, tick_size)
            axs[0].set_xticks(ticks)
            plt.setp(axs[0].get_xticklabels(), rotation=90)

            # cropped signal x-axis ticks
            axs[1].set_xlabel('time [s]')
            cropped_tick_size = cropped_duration / num_ticks
            cropped_ticks = np.arange(start_s, start_s + cropped_duration + cropped_tick_size, cropped_tick_size)
            axs[1].set_xticks(cropped_ticks)
            plt.setp(axs[1].get_xticklabels(), rotation=90)

            # title with info
            axs[0].set_title("Duration: %s | Tick size: %s" % (self.duration_s, tick_size))
            axs[1].set_title("Duration: %s | Tick size: %s" % (cropped_duration, cropped_tick_size))

            # red dotted vertical lines at x-axis ticks
            axs[0].grid(True, 'major', 'x', color='r', linestyle='dotted')
            axs[1].grid(True, 'major', 'x', color='r', linestyle='dotted')

            # only small margins
            axs[0].margins(x=0.01, y=0.01)
            axs[1].margins(x=0.01, y=0.01)

            # plot complete signal
            t = np.linspace(0, self.duration_s, len(self.__get_channel_data(channel)))
            axs[0].plot(t, self.__get_channel_data(channel), label='s(t)')
            axs[0].legend()

            # highlight cropped area
            axs[0].axvspan(start_s, start_s + cropped_duration, color='orange', alpha=0.5)

            # plot cropped signal
            axs[1].plot(cropped_t, cropped_data, label='s(t)')
            axs[1].legend()

            plt.tight_layout()

        else:
            # x-axis ticks
            axs.set_xlabel('time [s]')
            tick_size = self.duration_s / num_ticks
            ticks = np.arange(0., self.duration_s + tick_size, tick_size)
            axs.set_xticks(ticks)
            plt.setp(axs.get_xticklabels(), rotation=90)

            # title with info
            axs.set_title("Duration: %s | Tick size: %s" % (self.duration_s, tick_size))

            # red dotted vertical lines at x-axis ticks
            axs.grid(True, 'major', 'x', color='r', linestyle='dotted')

            # only small margins
            axs.margins(x=0.01, y=0.01)

            # plot complete signal
            t = np.linspace(0, self.duration_s, len(self.__get_channel_data(channel)))
            axs.plot(t, self.__get_channel_data(channel), label='s(t)')
            axs.legend()

        if return_plot:
            return fig, axs

    def play(self,
             start_s: np.float = 0.,
             stop_s: np.float = None,
             channel: int = 0) -> None:

        assert start_s >= 0
        assert stop_s is None or stop_s <= self.duration_s

        if stop_s is None:
            stop_s = self.duration_s

        cropped_data, cropped_duration = self.crop(start_s, stop_s, channel=channel)

        sd.play(cropped_data, self.sampling_freq)

    def get_frames(self,
                   frame_length_ms: int = 32,
                   frame_shift_ms: int = 16,
                   channel: int = 0) -> [np.ndarray, np.ndarray]:
        """
        :param frame_length_ms: the length of one frame in ms
        :param frame_shift_ms: the shift of each frame in ms
        :return: frame_centers_ms, frames
        """

        # translate from ms to indices
        frame_length_idx = self.ms_to_idx(frame_length_ms)
        frame_shift_idx = self.ms_to_idx(frame_shift_ms)

        # number of possible frames (w/o padding!)
        num_frames = np.floor((len(self.__get_channel_data(channel)) - frame_length_idx) / frame_shift_idx) + 1

        v_time_frame = list()
        frames = list()
        for i in np.arange(num_frames):
            frame_start = int(i * frame_shift_idx)
            frame_end = int(i * frame_shift_idx + frame_length_idx)
            assert (frame_end - frame_start) == frame_length_idx

            # frame center
            v_time_frame.append(self.idx_to_ms(frame_start) + (frame_length_ms / 2))
            # crop frame from signal
            frames.append(self.__get_channel_data(channel)[frame_start: frame_end])

        frame_centers_ms = np.array(v_time_frame)
        frames = np.array(frames)

        assert len(frames) == len(frame_centers_ms)

        return frame_centers_ms, frames

    def estimate_fundamental_freq(self,
                                  frame_length_ms: int = 32,
                                  frame_shift_ms: int = 16,
                                  min_freq_hz: int = 80,
                                  max_freq_hz: int = 400,
                                  plot=False,
                                  fig=None,
                                  axs=None,
                                  channel: int = 0) -> np.ndarray:

        # get frames
        v_time_frame, m_frames = self.get_frames(frame_length_ms, frame_shift_ms, channel=channel)

        # acf
        res = []
        for frame in m_frames:
            # compute auto-correlation
            acf = np.correlate(frame, frame, 'full')
            acf = acf[acf.size // 2:]

            # translate min_freq and max_freq to frame indices
            min_freq_idx = self.ms_to_idx(conversion.hz_to_ms(min_freq_hz))
            max_freq_idx = self.ms_to_idx(conversion.hz_to_ms(max_freq_hz))

            # find the fundamental period between 80Hz and 400Hz
            fundamental_period_idx = max_freq_idx + np.argmax(acf[max_freq_idx: min_freq_idx])

            # compute fundamental freq as reciprocal of the fundamental period
            fundamental_period_s = self.idx_to_s(fundamental_period_idx)
            res.append(1 / fundamental_period_s)

        res = np.array(res)

        if plot:
            # create plots
            if fig is None or axs is None:
                fig, axs = plt.subplots(1, 1, figsize=(20, 10))

            axs2 = axs.twinx()

            # x-axis ticks
            num_ticks = 80
            axs.set_xlabel('time [s]')
            tick_size = self.duration_s / num_ticks
            ticks = np.arange(0., self.duration_s + tick_size, tick_size)
            axs.set_xticks(ticks)
            plt.setp(axs.get_xticklabels(), rotation=90)

            # title with info
            axs.set_title("Duration: %s[s] | Tick size: %s[s]" % (self.duration_s, tick_size))

            # red dotted vertical lines at x-axis ticks
            axs.grid(True, 'major', 'x', color='r', linestyle='dotted')

            # only small margins
            axs.margins(x=0.01, y=0.01)
            axs2.margins(x=0.01, y=0.01)

            # set plot order
            axs.set_zorder(axs2.get_zorder() + 1)
            axs.patch.set_visible(False)

            # y-axis ticks
            axs.set_ylabel('Estimated fundamental frequency [Hz]', color='C3')
            ticks = np.arange(0., max(res) + 10, 10)
            axs.set_yticks(ticks)
            axs2.set_ylabel('Signal s(t)', color='C0')

            # plot estimated fundamental freqs
            t = np.linspace(0, self.duration_s, len(res))
            axs.plot(t, res, color='C3')

            # plot complete signal
            t2 = np.linspace(0, self.duration_s, len(self.data))
            axs2.plot(t2, self.__get_channel_data(channel), color='C0')

        return res

    def compute_stft(self,
                     frame_length_ms: int = 32,
                     frame_shift_ms: int = 16,
                     window_name: str = 'hann',
                     window=None,
                     channel: int = 0) -> [np.ndarray, np.ndarray, np.ndarray]:
        """
        :param frame_length_ms:
        :param frame_shift_ms:
        :param window_name: The analysis window which is applied to each frame.
        By default: Hann Window. Add 'sqrt_' as prefix to apply sqrt.
        :return: stft, freq_axis_hz, frame_centers_ms
        """

        if window is None:
            if 'sqrt_' in window_name:
                window = ss.get_window(window_name[5:], self.ms_to_idx(frame_length_ms), fftbins=True)
                window = np.sqrt(window)
            else:
                window = ss.get_window(window_name, self.ms_to_idx(frame_length_ms), fftbins=True)

        assert len(window) == self.ms_to_idx(frame_length_ms)

        # framing
        frame_centers_ms, frames = self.get_frames(frame_length_ms, frame_shift_ms, channel=channel)

        # apply analysis window by multiplying with each frame
        windowed = window * frames

        # FFT on each windowed segment
        stft = np.fft.fft(windowed, axis=1)

        # discard lower half including the nyquist bin
        N = stft.shape[1] // 2 + 1
        stft = stft[:, :N]

        # Nyquist frequency
        nyquist_freq_hz = self.sampling_freq / 2

        # freq axis in Hz
        freq_axis_hz = np.linspace(0, nyquist_freq_hz, N)

        return stft, freq_axis_hz, frame_centers_ms

    def plot_stft(self,
                  stft: np.ndarray = None,
                  freq_axis: np.ndarray = None,
                  frame_centers_ms: np.ndarray = None,
                  frame_length_ms: int = 32,
                  frame_shift_ms: int = 16,
                  window: str = 'hann',
                  return_plot: bool = False,
                  channel: int = 0):

        if stft is None or freq_axis is None or frame_centers_ms is None:
            stft, freq_axis, frame_centers_ms = self.compute_stft(frame_length_ms, frame_shift_ms, window,
                                                                  channel=channel)

        fig, axs = plt.subplots(3, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [2, 10, 0.5]})

        # x-axis ticks
        axs[0].set_xlabel('Time [s]')
        tick_size = self.duration_s / 80
        ticks = np.arange(0., self.duration_s + tick_size, tick_size)
        axs[0].set_xticks(ticks)
        plt.setp(axs[0].get_xticklabels(), rotation=90)

        # title with info
        axs[0].set_title("Duration: %s | Tick size: %s" % (self.duration_s, tick_size))

        # red dotted vertical lines at x-axis ticks
        axs[0].grid(True, 'major', 'x', color='r', linestyle='dotted')

        # no margins
        axs[0].margins(x=0.0)
        axs[1].margins(x=0.0)
        axs[2].margins(x=0.0)

        # plot complete time-domain signal
        t = np.linspace(0, self.duration_s, len(self.data))
        axs[0].plot(t, self.data)

        # plot complete frequency-domain signal

        axs[1].set_title(f"Frame Length [ms]: {frame_length_ms} | Frame Shift [ms]: {frame_shift_ms}| Window: {window}")
        axs[1].set_ylabel('Frequency [Hz]')
        axs[1].set_xlabel('Time [s]')
        tick_size = self.duration_s / 80
        ticks = np.arange(0., self.duration_s + tick_size, tick_size)
        axs[1].set_xticks(ticks)
        plt.setp(axs[1].get_xticklabels(), rotation=90)

        im = axs[1].imshow(10 * np.log10(np.maximum(np.square(np.abs(stft.T)), 10 ** (-15))),
                           cmap='viridis',
                           origin='lower',
                           extent=[ticks[0], ticks[-1], freq_axis[0], freq_axis[-1]],
                           aspect='auto')

        fig.colorbar(im, cax=axs[2], orientation="horizontal")

        plt.tight_layout()

        if return_plot:
            return fig, axs

    def compute_short_time_cepstrum(self,
                                    frame_length_ms: int = 32,
                                    frame_shift_ms: int = 16,
                                    window_name: str = 'hann',
                                    window=None,
                                    shift=False,
                                    channel: int = 0) -> [np.ndarray, np.ndarray, np.ndarray]:
        """
        :param frame_length_ms:
        :param frame_shift_ms:
        :param window_name: The analysis window which is applied to each frame.
        By default: Hann Window. Add 'sqrt_' as prefix to apply sqrt.
        :return: real cepstrum, freq_axis_hz, frame_centers_ms
        """

        if window is None:
            if 'sqrt_' in window_name:
                window = ss.get_window(window_name[5:], self.ms_to_idx(frame_length_ms), fftbins=True)
                window = np.sqrt(window)
            else:
                window = ss.get_window(window_name, self.ms_to_idx(frame_length_ms), fftbins=True)

        assert len(window) == self.ms_to_idx(frame_length_ms)

        # framing
        frame_centers_ms, frames = self.get_frames(frame_length_ms, frame_shift_ms, channel=channel)

        # apply analysis window by multiplying with each frame
        windowed = window * frames

        # RFFT on each windowed segment
        spectrum = np.fft.rfft(windowed, axis=1)

        # compute log magnitude spectrum
        log_mag = np.log(np.abs(spectrum))

        # compute the cepstrum by applying the IDFT of the log mag spectrum
        cepstrum = np.fft.irfft(log_mag, axis=1)

        if shift:
            cepstrum = np.fft.fftshift(cepstrum)

        # quefrency axis in 1/Hz
        quefrency_axis = np.arange(-cepstrum[0].size // 2, cepstrum[0].size // 2)

        return cepstrum, quefrency_axis, frame_centers_ms

    def crop(self,
             start_s: np.float,
             stop_s: np.float,
             channel: int = 0) -> (np.ndarray, np.float):

        cropped_data = self.__get_channel_data(channel)[self.s_to_idx(start_s): self.s_to_idx(stop_s)]
        cropped_duration = len(cropped_data) / self.sampling_freq

        return cropped_data, cropped_duration

    def ms_to_idx(self, ms: int) -> int:
        return conversion.ms_to_idx(ms, self.sampling_freq)

    def s_to_idx(self, s: np.float) -> int:
        return conversion.s_to_idx(s, self.sampling_freq)

    def idx_to_ms(self, idx: int) -> int:
        return conversion.idx_to_ms(idx, self.sampling_freq)

    def idx_to_s(self, idx: int) -> np.float:
        return conversion.idx_to_s(idx, self.sampling_freq)

    def __get_channel_data(self, channel: int = 0):
        assert channel < self.channels, "Cannot get data from channel %s since only %s channels available!" % (
            channel, self.channels)

        if self.channels == 1:
            return self.data
        else:
            return self.data[channel]
