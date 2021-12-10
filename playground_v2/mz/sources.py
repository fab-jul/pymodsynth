from mz import base  # TODO: private import to not pollute star imports!
import numpy as np


@base.moduleclass
class SineSource(base.Module):
    frequency: base.Module = base.Constant(440.)
    amplitude: base.Module = base.Constant(1.0)
    phase: base.Module = base.Constant(0.0)
    _last_cumsum_value: base.State = base.State(0.)

    def out_given_inputs(self, 
                         clock_signal: base.ClockSignal, 
                         frequency: np.ndarray, amplitude: np.ndarray, phase: np.ndarray):
        dt = np.mean(clock_signal.ts[1:] - clock_signal.ts[:-1])
        cumsum = np.cumsum(frequency * dt, axis=0) + self._last_cumsum_value.get()
        self._last_cumsum_value.set(cumsum[-1, :])
        out = amplitude * np.sin((2 * np.pi * cumsum) + phase)
        return out
