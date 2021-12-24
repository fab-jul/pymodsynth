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
        if len(result) > total_length:
            result = result[:total_length]
        else:
            missing = np.zeros((total_length - len(result),), dtype=result.dtype)
            result = np.concatenate((result, missing), 0)
        return clock_signal.add_channel_dim(result)
