from mz import base
from mz import helpers
from mz import sources

import math
import dataclasses
import scipy.signal
import functools
import numpy as np
from numpy.polynomial import Polynomial


@functools.lru_cache(maxsize=128)
def basic_reverb_ir(delay: int, echo: int, p: float):
    print("Making a reverb...")
    # We give it `delay` samples of nothing, then a linspace down.

    _, decayer = poly_fit([0, 0.3, 0.8, 0.9], [0.2, 0.15, 0.01, 0.], num_samples=echo)

    h = np.random.binomial(1, p, delay + echo) * np.concatenate(
        (np.zeros(delay), decayer), axis=0)
    h = h[:, np.newaxis]
    h[0, :] = 1  # Always repeat the signal also!
    return h


def poly_fit(xs, ys, num_samples):
    assert len(xs) == len(ys)
    p = Polynomial.fit(xs, ys, deg=len(xs) - 1)
    xs, ys = p.linspace(num_samples)
    return xs, ys


@helpers.mark_for_testing(src=lambda: base.Constant(10.))
class Reverb(base.Module):

    src: base.Module
    delay: base.SingleValueModule = base.Constant(3000)
    echo: base.SingleValueModule = base.Constant(10000)
    p: base.SingleValueModule = base.Constant(0.05)

    def out_given_inputs(self,
                         clock_signal: base.ClockSignal,
                         src: np.ndarray,
                         delay: float,
                         echo: float,
                         p: float):

        past_context = math.ceil(delay + echo)
        num_frames = int(math.ceil(past_context / clock_signal.num_samples)) + 1
        src = self.prepend_past("src", current=src, num_frames=num_frames)

        h = basic_reverb_ir(delay, echo, p)

        convolved = scipy.signal.convolve(src, h, mode="valid")
        return convolved[-clock_signal.num_samples:, :]


@functools.lru_cache(maxsize=128)
def _get_me_some_butter(order, fs, mode, sampling_rate):
    print("MAKING BUTTER")
    return scipy.signal.butter(order, fs, mode, fs=sampling_rate, output='sos')


@helpers.mark_for_testing(inp=sources.SineSource)
class ButterworthFilter(base.Module):

    inp: base.Module
    f_low: base.SingleValueModule = base.Constant(10.)
    f_high: base.SingleValueModule = base.Constant(100.)
    mode: str = "hp"
    order: int = 10
        
    def setup(self):
        # TODO: Use prepend_past
        self._last_signal = base.Stateful()

    def out_given_inputs(self, clock_signal: base.ClockSignal, 
                         inp, f_low: float, f_high: float):
        if self._last_signal is None:
            self._last_signal = clock_signal.zeros()
        num_samples, num_channels = clock_signal.shape

        f_low = np.clip(f_low, 1e-10, clock_signal.sample_rate/2)
        f_high = np.clip(f_high, 1e-10, clock_signal.sample_rate/2)

        full_signal = np.concatenate((self._last_signal, inp), axis=0)
        self._last_signal = inp

        fs = {"lp": f_low, "hp": f_high, "bp": (f_low, f_high)}[self.mode]
        sos = _get_me_some_butter(self.order, fs, self.mode, int(clock_signal.sample_rate))
        filtered_signal = scipy.signal.sosfilt(sos, full_signal[:,0])[-num_samples:, np.newaxis]
        filtered_signal = filtered_signal * np.ones(num_channels)
        return filtered_signal[-num_samples:, :]


# TODO(dariok,fab-jul): Make DelayElement into a module such that we can interactively
# change stuff like `time`.

# TODO(dariok): do we have access to the sampling frequency of the whole mod synt?
# if yes it's nice cause we could do actual physical values for example milliseconds for delay
# and hz for filter cutoffs...
@dataclasses.dataclass
class DelayElement:
    # number of buffer steps the signal is delayed
    time: int
    # how much of the signal is fed back (0-1)
    feedback: float
    # amplification applied to the delay
    gain: float = 1
    # whether or not the normal signal amplitude is decreased when the feedback increases
    # (requires some gain to keep the volume)
    proportional: bool = False
    # lower cutoff frequency of the filter relative to the nyquist frequency
    lo_cut: float = 0.01
    # upper cutoff frequency of the filter relative to the nyquist frequency
    hi_cut: float = 0.8
    # use a basic limiter to avoid distortion
    limit: bool = False

    def __post_init__(self):
        self.filter_coeff_b, self.filter_coeff_a = scipy.signal.butter(
            4, [self.lo_cut, self.hi_cut], 'band')
        self.delay = np.array([0])

    def __call__(self, signal_buffer: helpers.ArrayBuffer, num_samples: int):
        # TODO: Disabled, as I think it's just always 0,
        # Will this work for signals > num_samples?
        # n = signal_buffer.maxlen - self.time - 1
        full_signal = signal_buffer.get()
        delayed_signal = full_signal[:num_samples]

        if self.proportional:
            self.delay = delayed_signal * (1 - self.feedback) + self.delay * self.feedback
        else:
            self.delay = delayed_signal + self.delay * self.feedback

        self.delay = scipy.signal.lfilter(self.filter_coeff_b, self.filter_coeff_a, self.delay)

        delay = self.delay * self.gain

        if self.limit:
            # TODO: only experimental, need to check the proper dynamic range
            # you can get some drive if you set a high gain (~100) and enable limiting
           delay = scipy.special.expit(delay*6)

        return delay


class SimpleDelay(base.Module):

    signal: base.Module
    delay: DelayElement
    mix: float = 0.7

    def setup(self):
        self.buffer = helpers.ArrayBuffer(self.delay.time + 1)

    def out(self, clock_signal: base.ClockSignal):
        input_signal = self.signal(clock_signal)
        self.buffer.push(input_signal)
        return (input_signal * (1 - self.mix) + 
                self.delay(self.buffer, clock_signal.num_samples) * self.mix)
