import librosa as rosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


class AudioData:

    def __init__(self, path, data=None, dur=None, sampling_freq=None):
        if path is not None:
            self.path = path
            self.data, self.fs = rosa.load(path, sr=None)
            self.duration = len(self.data) / self.fs

            print('Successfully loaded audio data from file: %s' % (self.path))
        else:
            self.path = None
            self.data = data
            self.fs = sampling_freq
            self.duration = dur
            print('Successfully loaded audio data!')

    def plot(self, start_s=None, stop_s=None, num_ticks=80):
        assert stop_s is None or stop_s <= start_s + self.duration

        if stop_s is None:
            stop_s = self.duration

        fig, axs = None, None
        # init plot
        if start_s is None:
            fig, axs = plt.subplots(figsize=(20, 10))
        else:
            fig, axs = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [2, 10]})

        if start_s is not None:
            # crop signal
            cropped_data = self.data[int(start_s * self.fs): int(stop_s * self.fs)]
            cropped_duration = len(cropped_data) / self.fs
            cropped_t = np.linspace(start_s, start_s + cropped_duration, len(cropped_data))

            # x-axis ticks
            axs[0].set_xlabel('time [s]')
            tick_size = self.duration / num_ticks
            ticks = np.arange(0., self.duration + tick_size, tick_size)
            axs[0].set_xticks(ticks)
            plt.setp(axs[0].get_xticklabels(), rotation=90)

            # cropped signal x-axis ticks
            axs[1].set_xlabel('time [s]')
            cropped_tick_size = cropped_duration / num_ticks
            cropped_ticks = np.arange(start_s, start_s + cropped_duration + cropped_tick_size, cropped_tick_size)
            axs[1].set_xticks(cropped_ticks)
            plt.setp(axs[1].get_xticklabels(), rotation=90)

            # title with info
            axs[0].set_title("Duration: %s | Tick size: %s" % (self.duration, tick_size))
            axs[1].set_title("Duration: %s | Tick size: %s" % (cropped_duration, cropped_tick_size))

            # red dotted vertical lines at x-axis ticks
            axs[0].grid(True, 'major', 'x', color='r', linestyle='dotted')
            axs[1].grid(True, 'major', 'x', color='r', linestyle='dotted')

            # only small margins
            axs[0].margins(x=0.01, y=0.01)
            axs[1].margins(x=0.01, y=0.01)

            # plot complete signal
            t = np.linspace(0, self.duration, len(self.data))
            axs[0].plot(t, self.data)

            # highlight cropped area
            axs[0].axvspan(start_s, start_s + cropped_duration, color='orange', alpha=0.5)

            # plot cropped signal
            axs[1].plot(cropped_t, cropped_data)

            plt.tight_layout()

            plt.show()

        else:
            # x-axis ticks
            axs.set_xlabel('time [s]')
            tick_size = self.duration / num_ticks
            ticks = np.arange(0., self.duration + tick_size, tick_size)
            axs.set_xticks(ticks)
            plt.setp(axs.get_xticklabels(), rotation=90)

            # title with info
            axs.set_title("Duration: %s | Tick size: %s" % (self.duration, tick_size))

            # red dotted vertical lines at x-axis ticks
            axs.grid(True, 'major', 'x', color='r', linestyle='dotted')

            # only small margins
            axs.margins(x=0.01, y=0.01)

            # plot complete signal
            t = np.linspace(0, self.duration, len(self.data))
            axs.plot(t, self.data)

            plt.show()

    def play(self, start=0, stop=None):
        assert start >= 0
        assert stop is None or stop <= self.duration

        if stop is None:
            stop = self.duration

        data = self.data[int(start * self.fs): int(stop * self.fs)]

        sd.play(data, self.fs)

    def get_frames(self, frame_length_ms: int = 32, frame_shift_ms: int = 16) -> [np.ndarray, np.ndarray]:

        # translate from ms to indices
        frame_length = self.fs * (frame_length_ms / 1000)
        frame_shift = self.fs * (frame_shift_ms / 1000)

        # number of possible frames (w/o padding!)
        num_frames = np.floor((len(self.data) - frame_length) / frame_shift) + 1

        v_time_frame = list()
        m_frames = list()
        for i in np.arange(num_frames):
            # frame center
            v_time_frame.append(((i + 1) * frame_length_ms) / 2)

            # crop frame from signal
            frame_start = int(i * frame_shift)
            frame_end = int(i * frame_shift + frame_length)
            m_frames.append(self.data[frame_start: frame_end])

        v_time_frame = np.array(v_time_frame)
        m_frames = np.array(m_frames)

        return v_time_frame, m_frames

    def estimate_fundamental_freq(self, frame_length_ms: int = 32, frame_shift_ms: int = 16, min_freq: int = 80,
                                  max_freq: int = 400, plot=False):

        # get frames
        v_time_frame, m_frames = self.get_frames(frame_length_ms, frame_shift_ms)

        # acf
        res = []
        for frame in m_frames:
            # compute auto-correlation
            acf = np.correlate(frame, frame, 'full')
            acf = acf[acf.size // 2:]

            # translate index to ms
            step_ms = frame_length_ms / len(frame)
            # translate min_freq and max_freq to frame indices
            min_freq_idx = (1 / min_freq * 1000) / step_ms
            max_freq_idx = (1 / max_freq * 1000) / step_ms

            # find the fundamental period between 80Hz and 400Hz
            fundamental_period_idx = max_freq_idx + np.argmax(acf[int(max_freq_idx): int(min_freq_idx)])

            # compute fundamental freq as reziprocal of the fundamental period
            fundamental_period_ms = (fundamental_period_idx * step_ms) / 1000
            res.append(1 / fundamental_period_ms)

        res = np.array(res)

        if plot:
            # create plots
            fig, axs = plt.subplots(1, 1, figsize=(20, 10))
            axs2 = axs.twinx()

            # x-axis ticks
            num_ticks = 80
            axs.set_xlabel('time [s]')
            tick_size = self.duration / num_ticks
            ticks = np.arange(0., self.duration + tick_size, tick_size)
            axs.set_xticks(ticks)
            plt.setp(axs.get_xticklabels(), rotation=90)

            # title with info
            axs.set_title("Duration: %s[s] | Tick size: %s[s]" % (self.duration, tick_size))

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
            t = np.linspace(0, self.duration, len(res))
            axs.plot(t, res, color='C3')

            # plot complete signal
            t2 = np.linspace(0, self.duration, len(self.data))
            axs2.plot(t2, self.data, color='C0')

            plt.show()

        return res

    def crop(self, start_s: np.float, stop_s: np.float):
        cropped_data = self.data[self.s_to_idx(start_s): self.s_to_idx(stop_s)]
        cropped_duration = len(cropped_data) / self.fs

        return cropped_data, cropped_duration

    def ms_to_idx(self, ms: int):
        return ms * 1000 * self.fs

    def s_to_idx(self, s: np.float):
        return s * self.fs
