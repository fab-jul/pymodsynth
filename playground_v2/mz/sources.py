
from typing import Sequence
from mz import base
from mz import helpers
import numpy as np


# ------------------------------------------------------------------------------
# Sine


@helpers.mark_for_testing()
class SineSource(base.Module):
    frequency: base.Module = base.Constant(440.)
    amplitude: base.Module = base.Constant(1.0)
    phase: base.Module = base.Constant(0.0)

    def setup(self):
        self._last_cumsum_value = base.Stateful(0.)

    def out_given_inputs(self, 
                         clock_signal: base.ClockSignal, 
                         frequency: np.ndarray, 
                         amplitude: np.ndarray,
                         phase: np.ndarray):
        dt = np.mean(clock_signal.ts[1:] - clock_signal.ts[:-1])
        cumsum = np.cumsum(frequency * dt, axis=0) + self._last_cumsum_value
        self._last_cumsum_value = cumsum[-1, :]
        out = amplitude * np.sin((2 * np.pi * cumsum) + phase)
        return out


@helpers.mark_for_testing()
class TimeIndependentSineSource(base.Module):
    """A sine that always starts at 0, regardless of the current song time."""

    frequency: base.Module = base.Constant(440.)
    amplitude: base.Module = base.Constant(1.0)
    phase: base.Module = base.Constant(0.0)

    def out_given_inputs(self, 
                         clock_signal: base.ClockSignal, 
                         frequency: np.ndarray, 
                         amplitude: np.ndarray,
                         phase: np.ndarray
                         ):
        dt = np.mean(clock_signal.ts[1:] - clock_signal.ts[:-1])
        cumsum = np.cumsum(frequency * dt, axis=0)
        out = amplitude * np.sin((2 * np.pi * cumsum) + phase)
        return out


# ------------------------------------------------------------------------------
# Triangular Shapes

_EPS_ALPHA = 1e-7


class SkewedTriangleSource(base.Module):
    """A triangle that has the peak at alpha * period.
    
    Set to alpha = 1. for a saw, alpha = 0.5 for a symetrical triangle.
    """

    frequency: base.Module = base.Constant(220.)
    alpha: base.Module = base.Constant(0.5)

    def out_given_inputs(self, clock_signal: base.ClockSignal, frequency: np.ndarray,
                         alpha: np.ndarray):
        alpha = np.clip(alpha, _EPS_ALPHA, 1-_EPS_ALPHA)
        period = 1/frequency
        ys = (clock_signal.ts % period)/period * (1/alpha)
        ys_flipped = 1/(1-alpha) -(clock_signal.ts % period)/period * (1/(1-alpha))
        ys = np.minimum(ys, ys_flipped)
        return ys * 2 - 1


# ------------------------------------------------------------------------------
# Others

class PeriodicTriggerSource(base.Module):
    
    bpm: base.SingleValueModule = base.Constant(130)
    note_value: float = 1/4

    def out_given_inputs(self, clock_signal: base.ClockSignal, bpm: float):
        sample_rate = clock_signal.sample_rate
        samples_per_forth = round(sample_rate / (bpm / 60))  # number of samples between 1/4 triggers
        samples_per_pattern_element = round(samples_per_forth * 4 * self.note_value)

        # Elements are in {True, False}.
        triggers = (clock_signal.sample_indices % samples_per_pattern_element == 0)
        # Elements are in {1, 0}.
        triggers = np.ones(clock_signal.shape) * triggers.reshape(-1, 1)
        return triggers


class Hold(base.Module):
    """Takes a sparse signal and holds each element.

    Example:
    
        input:  [0, 0, 10,  0,  0, 5, 0, 0]
        output: [0, 0, 10, 10, 10, 5, 5, 5]

    See docstring of `Cycler` for a usage example.
    """

    src: base.Module

    def setup(self):
        self.prev_value = base.Stateful(0.)

    def out_given_inputs(self, clock_signal: base.ClockSignal, src):
        indices_non_zero = np.nonzero(src)[0].tolist()
        if not indices_non_zero:  # All inputs are zero!
            return np.ones_like(src) * self.prev_value
        if indices_non_zero[0] != 0:
            src[0] = self.prev_value
            indices_non_zero.insert(0, 0)
        indices_non_zero.append(src.shape[0])
        res = np.concatenate(
            [np.ones((end-start, 1)) * src[start, :] 
                              for start, end in zip(indices_non_zero, indices_non_zero[1:])],
                              axis=0)
        self.prev_value = res[-1, 0]
        return res


class Cycler(base.BaseModule):
    """Returns the elements in `seq` one by one whenever called.

    Example:
      cycler = Cycler(seq=[7, 5, 6])
      cycler(clock) -> [7]
      cycler(clock) -> [5]
      cycler(clock) -> [6]
      cycler(clock) -> [7]
      ...

    Useful as a shape_maker for `TriggerModulator`. Combine with `Hold` 
    to hold values:

      cycler = Cycler(seq=[7, 5, 6])
      triggers = ...
      cycler_at_triggers = Hold(
          TriggerModulator(shape_maker=cycler, tirggers=triggers))

      # Now the outputs would look like this:
      triggers = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
      outputs =  [0, 0, 7, 7, 7, 5, 5, 5, 6, 6, 7, 7, 7, 7]


    """

    # TODO: Figure out why this must be hashable!
    seq: Sequence[float]

    def setup(self):
        self.current_i = base.Stateful(0)

    def out(self, clock_signal: base.ClockSignal):
        el = self.seq[self.current_i]
        self.current_i = (self.current_i + 1) % len(self.seq)
        return clock_signal.add_channel_dim(np.array([el]))

 
@helpers.mark_for_testing(shape_maker=lambda: Cycler((1, 2, 3)),
                          triggers=PeriodicTriggerSource)
class TriggerModulator(base.Module):
    """Puts a shape onto an envelope.

    The `shape_maker` can be an arbirary module. It will be called
    for each trigger, and is expected to return a signal of whatever
    length is suitable.
    """

    shape_maker: base.BaseModule
    triggers: base.Module

    # How to combine overlapping shapes.
    combinator: base._Operator = np.add

    def setup(self):
        self.previous = base.Stateful()

    def out(self, clock_signal: base.ClockSignal):
        """Generate one shape per trigger."""

        trigger_indices = np.nonzero(self.triggers(clock_signal))[0]

        envelopes = [self.shape_maker(clock_signal.clock.get_clock_signal_with_start(i)) 
                     for i in trigger_indices]

        current_signal = clock_signal.zeros()
        if self.previous is not None and len(self.previous) > 0:
            previous_signal = self.previous
        else:
            previous_signal = clock_signal.zeros()
        if envelopes:
            # does any envelope go over frame border?
            latest_envelope_end = max(i + len(env) for i, env in zip(trigger_indices, envelopes))
            if latest_envelope_end > clock_signal.num_samples:
                remainder = latest_envelope_end - clock_signal.num_samples
            else:
                remainder = 0
            current_signal = np.pad(current_signal, pad_width=((0, remainder), (0, 0)))
            for i, envelope in zip(trigger_indices, envelopes):
                if len(envelope.shape) != 2:
                    envelope = clock_signal.add_channel_dim(envelope)
                current_signal[i:i + len(envelope), :] = envelope
        # combine the old and new signal using the given method
        max_len = max(len(previous_signal), len(current_signal))
        previous_signal = np.pad(previous_signal, pad_width=((0, max_len - len(previous_signal)), (0, 0)))
        current_signal = np.pad(current_signal, pad_width=((0, max_len - len(current_signal)), (0, 0)))
        result = self.combinator(previous_signal, current_signal)
        self.previous = result[clock_signal.num_samples:]
        res = result[:clock_signal.num_samples]
        return res


# TODO: From here down, code is not working, because of rhythm issues. REVISIT!

## TODO: Rhytm problems!!! when you have multiple things
#@helpers.mark_for_testing()
#class Periodic(base.Module):
#
#    def setup(self):
#        self._signal = base.Stateful()
#        self._future = base.Stateful(base.BlockFutureCache())
#
#    def set_signal(self, signal: np.ndarray):
#        self._signal = signal
#
#    def out(self, clock_signal: base.ClockSignal):
#        if self._signal is None:
#            return clock_signal.zeros()
#        future: base.BlockFutureCache = self._future.get()
#        output = future.get(num_samples=clock_signal.num_samples, future=self._signal)
#        if len(output.shape) != 2:
#            output = np.broadcast_to(output.reshape(-1, 1), clock_signal.shape)
#        return output


       
## @dataclasses.dataclass(unsafe_hash=True)
## class Pattern:
##     """Input to TriggerSource."""
##     # Beginning of notes. Use 0 to indicate pause, and 1... to indicate notes relative
##     # to a base frequency, where 1 == the base frequency, and 13 == the base frequency but one octave higher.
##     # Examples: [1, 0, 0, 0, 1, 0, 0, 0], [1, 4, 7]
##     pattern: Tuple[float, ...]
##     # What each note in `pattern` represents.
##     note_value: float = 1/4 # 1/4, 1/8, etc.
##     # How long to hold each note in `pattern`, in terms of `note_value`
##     note_lengths: Union[int, Tuple[int, ...]] = 1  
## 
##     def __post_init__(self):
##         self.pattern = tuple(self.pattern)
##         # TODO: broadcast note_lengths
## 
##     def to_triggers(self, bpm, sample_rate):
##         # assert self.note_value >= 1/4  # TODO
##         samples_per_forth = round(sample_rate / (bpm / 60))  # number of samples between 1/4 triggers
##         sampler_per_pattern_element = round(samples_per_forth * 4 * self.note_value)
## 
##         broadcaster = np.zeros(sampler_per_pattern_element)
##         broadcaster[0] = 1
## 
##         pattern_as_01 = [int(p > 0.5) for p in self.pattern]
##         triggers = (np.array(pattern_as_01).reshape(-1, 1) * broadcaster).flatten()
##         return triggers
## 
## 
## class MelodySequencer(base.MultiOutputModule):
## 
##     bpm: base.SingleValueModule
##     # Should be list of Parameters
##     pattern: Pattern
## 
##     foo: base.BaseModule
##     bar: Sequence[base.BaseModule]
## 
##     def setup(self):
## 
##         self.bar = {"foo":{"bar": ["bas", M]}}
## 
##         isinstance(bar, Module)
## 
##         self.periodic = Periodic()
##         self.note = base.Constant(220)  # TODO
##         self.hold = base.Constant(512)
## 
##     def out(self, clock_signal: base.ClockSignal):
##         bpm: float = self.bpm.out_single_value(clock_signal)
##         pattern_in_sample_space = self.pattern.to_triggers(bpm, clock_signal.sample_rate)
##         self.periodic.set_signal(pattern_in_sample_space)
##         return {
##             "triggers": self.periodic(clock_signal),
##             "note": self.note(clock_signal),
##             "hold": self.hold(clock_signal),
##             }
## 
## 
## def main():
##     from mz.experimental import subplots
##     import matplotlib.pyplot as plt
## 
##     # TODO: WIP
##     s = StepSequencer(SineSource())
##     a = s.output("foo")
##     b = s.output("bar")
## 
##     c = a + b
## 
##     s = base.ClockSignal.test_signal()
##     print(a(s), b(s))
## 
##     #pattern = Pattern([1, 0, 1, 0], note_values=1/4)
##     #s = subplots.Subplots(nrows=3)
##     #s.next_ax().plot(pattern.to_triggers(160, 44100))
##     #plt.show()
## 
## 
## if __name__ == "__main__":
##     main()
## 
