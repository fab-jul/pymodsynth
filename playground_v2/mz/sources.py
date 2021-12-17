import functools

from typing import Optional, Sequence, Union, Tuple
import collections
import dataclasses
from mz import base
from mz import helpers
import numpy as np


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


# TODO: Rhytm problems!!! when you have multiple things
@helpers.mark_for_testing()
class Periodic(base.Module):

    def setup(self):
        self._signal = base.Stateful()
        self._future = base.Stateful(base.BlockFutureCache())

    def set_signal(self, signal: np.ndarray):
        self._signal = signal

    def out(self, clock_signal: base.ClockSignal):
        if self._signal is None:
            return clock_signal.zeros()
        future: base.BlockFutureCache = self._future.get()
        output = future.get(num_samples=clock_signal.num_samples, future=self._signal)
        if len(output.shape) != 2:
            output = np.broadcast_to(output.reshape(-1, 1), clock_signal.shape)
        return output


class PeriodicTriggerSource(base.Module):
    
    bpm: base.SingleValueModule = base.Constant(130)
    note_value: float = 1/4

    def out_given_inputs(self, clock_signal: base.ClockSignal, bpm: float):
        sample_rate = clock_signal.sample_rate
        samples_per_forth = round(sample_rate / (bpm / 60))  # number of samples between 1/4 triggers
        samples_per_pattern_element = round(samples_per_forth * 4 * self.note_value)

        # Bool array.
        triggers = (clock_signal.sample_indices % samples_per_pattern_element == 0)
        # {0, 1} array.
        triggers = triggers.reshape(-1, 1) * np.ones(clock_signal.shape)
        return triggers


#class Melody(base.BaseModule):
#    melody: Sequence[Note]



# TODO: WIP 
## class PatternMaker(base.BaseModule):
## 
##     p: base.SingleValueModule
##     l: base.SingleValueModule
## 
##     def out_given_inputs(self, clock_signal: ClockSignal, p, l):
##         ...
## 
## 
## class SamplesToTriggers(base.Module):
## 
##     def out(self, clock_signal: base.ClockSignal):
##         sample = self.sampler.sample(clock_signal.get_clock(), 
##                                      num_samples=5000)  # TODO: should be internal somehow I think
## 
## 
## class SineSample(base.BaseModule):
## 
##     frequency: BaseModule
##     length: BaseModule
## 
##     def setup(self):
##         pass
## 
##     def out(self, clock_signal):
##         freq, length = ...
## 
##         self._sincache[[freq, leng]] = ...
## 
##         sin()
##         #fake_clock_signal = ClockSignalStartingAt0(lenght=self.length(clock_signal))
##         return self.src(fake_clock_signal)
 
# TODO: Add to tests
# TODO: Fix, currently broken
class TriggerModulator(base.Module):
    """
    Simplified OldTriggerModulator. Put an envelope on a trigger track.
    Stateful:
    Put an envelope on every trigger. If result is longer than a frame, keep the rest for the next call.
    Combine overlaps with a suitable function: max, fst, snd, add, ...
    """

    shape_maker: base.BaseModule
    triggers: base.Module
    combinator: base._Operator = np.add

    def setup(self):
        self.previous = base.Stateful()

    def out(self, clock_signal: base.ClockSignal):
        """Generate one sample per trigger"""
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
                current_signal[i:i + len(envelope)] = envelope.reshape((-1, 1))
        # combine the old and new signal using the given method
        max_len = max(len(previous_signal), len(current_signal))
        previous_signal = np.pad(previous_signal, pad_width=((0, max_len - len(previous_signal)), (0, 0)))
        current_signal = np.pad(current_signal, pad_width=((0, max_len - len(current_signal)), (0, 0)))
        result = self.combinator(previous_signal, current_signal)
        self.previous = result[clock_signal.num_samples:]
        res = result[:clock_signal.num_samples]
        return res

       
@dataclasses.dataclass(unsafe_hash=True)
class Pattern:
    """Input to TriggerSource."""
    # Beginning of notes. Use 0 to indicate pause, and 1... to indicate notes relative
    # to a base frequency, where 1 == the base frequency, and 13 == the base frequency but one octave higher.
    # Examples: [1, 0, 0, 0, 1, 0, 0, 0], [1, 4, 7]
    pattern: Tuple[float, ...]
    # What each note in `pattern` represents.
    note_value: float = 1/4 # 1/4, 1/8, etc.
    # How long to hold each note in `pattern`, in terms of `note_value`
    note_lengths: Union[int, Tuple[int, ...]] = 1  

    def __post_init__(self):
        self.pattern = tuple(self.pattern)
        # TODO: broadcast note_lengths

    def to_triggers(self, bpm, sample_rate):
        # assert self.note_value >= 1/4  # TODO
        samples_per_forth = round(sample_rate / (bpm / 60))  # number of samples between 1/4 triggers
        sampler_per_pattern_element = round(samples_per_forth * 4 * self.note_value)

        broadcaster = np.zeros(sampler_per_pattern_element)
        broadcaster[0] = 1

        pattern_as_01 = [int(p > 0.5) for p in self.pattern]
        triggers = (np.array(pattern_as_01).reshape(-1, 1) * broadcaster).flatten()
        return triggers


class MelodySequencer(base.MultiOutputModule):

    bpm: base.SingleValueModule
    # Should be list of Parameters
    pattern: Pattern

    foo: base.BaseModule
    bar: Sequence[base.BaseModule]

    def setup(self):

        self.bar = {"foo":{"bar": ["bas", M]}}

        isinstance(bar, Module)

        self.periodic = Periodic()
        self.note = base.Constant(220)  # TODO
        self.hold = base.Constant(512)

    def out(self, clock_signal: base.ClockSignal):
        bpm: float = self.bpm.out_single_value(clock_signal)
        pattern_in_sample_space = self.pattern.to_triggers(bpm, clock_signal.sample_rate)
        self.periodic.set_signal(pattern_in_sample_space)
        return {
            "triggers": self.periodic(clock_signal),
            "note": self.note(clock_signal),
            "hold": self.hold(clock_signal),
            }


def main():
    from mz.experimental import subplots
    import matplotlib.pyplot as plt

    # TODO: WIP
    s = StepSequencer(SineSource())
    a = s.output("foo")
    b = s.output("bar")

    c = a + b

    s = base.ClockSignal.test_signal()
    print(a(s), b(s))

    #pattern = Pattern([1, 0, 1, 0], note_values=1/4)
    #s = subplots.Subplots(nrows=3)
    #s.next_ax().plot(pattern.to_triggers(160, 44100))
    #plt.show()


if __name__ == "__main__":
    main()
