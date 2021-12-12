from mz import base
import math
from mz import helpers
import scipy.signal
import functools
import numpy as np
from numpy.polynomial import Polynomial

from mz import sources


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
        self._last_signal = base.State()

    def out_given_inputs(self, clock_signal: base.ClockSignal, inp, f_low: float, f_high: float):
        if not self._last_signal.is_set:
            self._last_signal.set(clock_signal.zeros())
        num_samples, num_channels = clock_signal.shape

        f_low = np.clip(f_low, 1e-10, clock_signal.sample_rate/2)
        f_high = np.clip(f_high, 1e-10, clock_signal.sample_rate/2)

        full_signal = np.concatenate((self._last_signal.get(), inp), axis=0)
        self._last_signal.set(inp)

        fs = {"lp": f_low, "hp": f_high, "bp": (f_low, f_high)}[self.mode]
        sos = _get_me_some_butter(self.order, fs, self.mode, int(clock_signal.sample_rate))
        filtered_signal = scipy.signal.sosfilt(sos, full_signal[:,0])[-num_samples:, np.newaxis]
        filtered_signal = filtered_signal * np.ones(num_channels)
        return filtered_signal[-num_samples:, :]

