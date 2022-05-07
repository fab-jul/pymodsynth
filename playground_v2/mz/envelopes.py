from typing import Sequence

import numpy as np
import sys
from mz import base
from mz import helpers

@helpers.mark_for_testing()
class ADSREnvelopeGenerator(base.BaseModule):

    # These times are relative to `total_length`.
    attack: base.SingleValueModule = base.Constant(1.)
    decay: base.SingleValueModule = base.Constant(1.)
    sustain: base.SingleValueModule = base.Constant(0.5)
    release: base.SingleValueModule = base.Constant(1.)
    hold: base.SingleValueModule = base.Constant(5.)
    total_length: base.SingleValueModule = base.Constant(1000.)

    def out_given_inputs(self, clock_signal: base.ClockSignal, 
                         attack: float, decay: float, sustain: float, release: float, hold: float,
                         total_length: float):
        """Return an envelope of length `total_length`"""
        # TODO: Support int SingleValueModules in the framework?
        total_length: int = round(total_length)
        current_length = attack + decay + hold + release
        scale = total_length / current_length
        attack = np.linspace(0, 1, round(attack * scale))
        decay = np.linspace(1, sustain, round(decay * scale))
        hold = np.ones(round(hold * scale)) * sustain
        release = np.linspace(sustain, 0, round(release * scale))
        result = np.concatenate((attack, decay, hold, release), 0)
        # Fix off-by-one errors due to rounding, this will add/remove 1 frame to fit `total_length`.
        return base.pad_or_truncate(result, total_length)


class PiecewiseLinearEnvelope(base.BaseModule):

    xs: Sequence[float]
    ys: Sequence[float]
    length: base.SingleValueModule = base.Constant(500.)

    def setup(self):
        assert len(self.xs) == len(self.ys)
        assert max(self.xs) <= 1.
        assert min(self.xs) >= 0.
        if self.xs[-1] < 1.:
            self.xs = (*self.xs, 1.)
            self.ys = (*self.ys, self.ys[-1])

    def out_given_inputs(self, clock_signal: base.ClockSignal, length: float):
        length: int = round(length)
        print("#$envelope length", length)
        prev_x_abs = 0
        prev_y = 1.
        pieces = []
        for x, y in zip(self.xs, self.ys):
            x_abs = round(x * length)
            if x_abs - prev_x_abs <= 0:
                prev_y = y
                continue
            pieces.append(np.linspace(prev_y, y, x_abs - prev_x_abs))
            prev_x_abs = x_abs
            prev_y = y
        env = np.concatenate(pieces, 0)
        env = clock_signal.pad_or_truncate(env, pad=env[-1])
        return env


class RectangleEnvGen_other(base.BaseModule):
    length: base.Module

    def setup(self):
        self.out = PiecewiseLinearEnvelope([0.0, 0.1, 0.9], [0.0, 0.8, 0.9], self.length)


class RectangleEnvGen(base.BaseModule):
    length: base.Module

    def out(self, clock_signal: base.ClockSignal):
        len_signal = self.length(clock_signal)
        length = int(np.mean(len_signal))
        #print(length)
        return np.ones((length,))

# see sources.SignalWithEnvelope
# class SignalWithEnvelope(base.BaseModule):
#
#     src: base.BaseModule
#     env: base.BaseModule
#
#     def out(self, clock_signal: base.ClockSignal):
#         # This defines the length!
#         env = self.env(clock_signal)
#         fake_clock_signal = clock_signal.change_length(env.shape[0])
#         src = self.src(fake_clock_signal)
#         return env * src




