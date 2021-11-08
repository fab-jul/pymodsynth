"""First version of something like a modular synth.

GOALS:
- select one of two input signals: sine or saw tooth.
- have one filter that low-passes with a user selected, live-changing, frequency.
"""
import abc

import numpy as np

import scipy.signal

# one array of ts, one array of data (may have multiple channels)
# shape: ((512,)    , (512,1)   )
Signal = (np.ndarray, np.ndarray)


class ConstantSignal:
    def __init__(self, value):
        self.value = value

    def __call__(self, inp: Signal) -> Signal:
        ts, _ = inp
        o = np.repeat(self.value, ts.shape[0])
        return ts, o


class SawSource:

    def __init__(self, amplitude_signal, frequency_signal):
        self.amplitude = amplitude_signal
        self.frequency = frequency_signal

    def __call__(self, inp: Signal) -> Signal:
        ts, _ = inp
        period = 2 * np.pi * self.frequency(inp)[1] / 200000
        o = (((ts % period) / (period / 2)) - 1) * self.amplitude(inp)[1]
        return ts, o.reshape(-1,1)



class SineSource:

    def __init__(self, amplitude_signal, frequency_signal):
        self.amplitude = amplitude_signal
        self.frequency = frequency_signal

    def __call__(self, inp: Signal) -> Signal:
        ts, _ = inp
        print("ts", ts.shape, inp[1])
        print("freq", self.frequency(inp))
        print("freq[1]]", self.frequency(inp)[1])

        print("self.amplitude(inp)[1]", self.amplitude(inp)[1].shape)
        print("self.frequency(inp)[1] * ts", (self.frequency(inp)[1] * ts).shape)
        print("big", (self.amplitude(inp)[1] * np.sin(2 * np.pi * self.frequency(inp)[1] * ts)).shape)


        o = (self.amplitude(inp)[1] * np.sin(2 * np.pi * self.frequency(inp)[1] * ts) +
             self.amplitude(inp)[1] * .5 * np.sin(2 * np.pi * 2 / 3 * self.frequency(inp)[1] * ts) +
             self.amplitude(inp)[1] * .25 * np.sin(2 * np.pi * 2 / 3 * 2 / 3 * self.frequency(inp)[1] * ts))
        print("o", o.shape)
        print("o reshaped", o.reshape(-1, 1).shape)
        return ts, o.reshape(-1, 1)


class LowPassFilter:
# wip rewrite to ts, xs
    def __init__(self, window=100):
        self.last_signal = np.zeros((512, 1))
        self.f = np.ones((window, 1)) / window

    def __call__(self, inp: Signal) -> Signal:
        ts, xs = inp
        # 1024, 1
        full_signal = np.concatenate((self.last_signal, inp), axis=0)
        o = scipy.signal.convolve(full_signal, self.f,
                                  mode='valid')
        self.last_signal = inp
        return o[-512:, :]


class ModulateFilter:

    def __init__(self, freq=0.0033):
        self.sin = SineSource(1.0, freq)

    def __call__(self, signal):
        o = signal[:, 0] * self.sin(np.arange(512)) # hack, because this modulator has no persistent inner clock
        return np.reshape(o, (512,1))


class OutputGeneratorV1:

    def __init__(self):
        self.freq = SineSource(amplitude_signal=ConstantSignal(0.4), frequency_signal=ConstantSignal(10))
        self.src = SineSource(amplitude_signal=ConstantSignal(0.4), frequency_signal=self.freq)
        #self.lowpass = LowPassFilter()
        #self.modulator = ModulateFilter()

    def __call__(self, inp: Signal) -> Signal:
        return self.src(inp)


