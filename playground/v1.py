"""First version of something like a modular synth.

GOALS:
- select one of two input signals: sine or saw tooth.
- have one filter that low-passes with a user selected, live-changing, frequency.
"""
import abc

import numpy as np

import scipy.signal

Signal = np.ndarray


class SumFilter:

    def __init__(self, window=5):
        self.window = window
        self.last_signal = None

    def __call__(self, signal: Signal):
        ...

# def __call__(self, ts, signal: Signal):
# -> Signal

class Parameter:

    def __init__(self, value):
        self.value = value

    def get(self):
        ...


class SawSource:

    def __init__(self, amplitude, frequency):
        self.amplitude = amplitude
        self.frequency = frequency

    def __call__(self, ts: Signal) -> Signal:
        period = 2 * np.pi * self.frequency / 200000
        o = (((ts % period) / (period / 2)) - 1) * self.amplitude
        print(np.min(o), np.max(o))
        return o


class SineSource:

    def __init__(self, amplitude, frequency):
        self.amplitude = amplitude
        self.frequency = frequency

    def __call__(self, ts: Signal) -> Signal:
        return (self.amplitude * np.sin(2 * np.pi * self.frequency * ts) +
                self.amplitude * .5 * np.sin(2 * np.pi * 2/3 * self.frequency * ts) +
                self.amplitude * .25 * np.sin(2 * np.pi * 2/3 * 2/3 * self.frequency * ts)
                )


class LowPassFilter:

    def __init__(self, window=100):
        self.last_signal = np.zeros((512, 1))
        self.f = np.ones((window, 1)) / window

    def __call__(self, ts: Signal) -> Signal:
        # 1024, 1
        full_signal = np.concatenate((self.last_signal, ts), axis=0)
        o = scipy.signal.convolve(full_signal, self.f,
                                  mode='valid')
        self.last_signal = ts
        return o[-512:, :]


class ModulateFilter:

    def __init__(self, freq=400):
        self.sin = SineSource(0.2, freq)

    def __call__(self, signal):
        return signal * self.sin(np.arange(512) )


class OutputGeneratorV1:

    def __init__(self, src):
        self.src = src
        self.lowpass = LowPassFilter()

    def __call__(self, ts: Signal) -> Signal:
        return self.lowpass(self.src(ts))

