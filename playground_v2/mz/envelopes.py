import numpy as np
import sys
from mz import base
from mz import helpers

@helpers.mark_for_testing()
class ADSREnvelopeGenerator(base.BaseModule):

    attack: base.SingleValueModule = base.Constant(100.)
    decay: base.SingleValueModule = base.Constant(100.)
    sustain: base.SingleValueModule = base.Constant(0.5)
    release: base.SingleValueModule = base.Constant(100.)
    hold: base.SingleValueModule = base.Constant(500.)

    def out_given_inputs(self, clock_signal: base.ClockSignal, 
                         attack: float, decay: float, sustain: float, release: float, hold: float):
        attack = np.linspace(0, 1, round(attack))
        decay = np.linspace(1, sustain, round(decay))
        hold = np.ones(round(hold)) * sustain
        release = np.linspace(sustain, 0, round(release))
        result = np.concatenate((attack, decay, hold, release), 0)
        # NOTE/describe: ignores clock_signal
        return clock_signal.add_channel_dim(result)

        if len(result) < clock_signal.num_samples:
            result = np.concatenate((result, np.zeros(clock_signal.num_samples - len(result))), 0)
        elif len(result) > clock_signal.num_samples:
            print("WARN: envelope too big for clock_signal!", file=sys.stderr)
            result = result[:clock_signal.num_samples]
        return np.broadcast_to(result.reshape(-1, 1), clock_signal.shape)
