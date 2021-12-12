from mz import base
import math
from mz import helpers
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
