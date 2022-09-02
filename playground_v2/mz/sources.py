import dataclasses
from functools import cache
from typing import Sequence, Callable, List, Tuple, Optional
from mz import base
from mz import helpers
from mz import envelopes
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
        # TODO: Move to clock signal, add `dt` function.
        dt = np.mean(clock_signal.ts[1:] - clock_signal.ts[:-1])
        cumsum = np.cumsum(frequency * dt, axis=0) + self._last_cumsum_value
        self._last_cumsum_value = cumsum[-1]
        out = amplitude * np.sin((2 * np.pi * cumsum) + phase)
        return out


class LFO(base.Module):

    frequency: base.Module = base.Constant(1)
    lo: float = 0.
    hi: float = 1.

    def setup(self):
        self.sine = SineSource(frequency=self.frequency)

    def out(self, clock_signal):
        sine = self.sine(clock_signal)
        sine_between_0_and_1 = (sine + 1) / 2
        return sine_between_0_and_1 * (self.hi - self.lo) + self.lo



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

# TODO: Does not properly handle frequency = LFO!
@helpers.mark_for_testing()
class SkewedTriangleSource(base.Module):
    """A triangle that has the peak at alpha * period.
    Set to alpha = 1. for a saw, alpha = 0.5 for a symmetrical triangle.
    """

    frequency: base.Module = base.Constant(220.)
    alpha: base.Module = base.Constant(0.5)

    def out_given_inputs(self, clock_signal: base.ClockSignal, frequency: np.ndarray,
                         alpha: np.ndarray):
        alpha = np.clip(alpha, _EPS_ALPHA, 1-_EPS_ALPHA)
        period = 1/frequency
        ys = (clock_signal.ts % period)/period * (1/alpha)
        ys_flipped = 1/(1-alpha) - (clock_signal.ts % period)/period * (1/(1-alpha))
        ys = np.minimum(ys, ys_flipped)
        return ys * 2 - 1


# ---
# TODO


class SignalWithEnvelope(base.BaseModule):

    src: base.BaseModule
    env: base.BaseModule

    def out(self, clock_signal: base.ClockSignal):
        # This defines the length!
        env = self.env(clock_signal)
        fake_clock_signal = clock_signal.change_length(env.shape[0])
        src = self.src(fake_clock_signal)
        return env * src


class PiecewiseLinearEnvelope(base.Module):

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




# ------------------------------------------------------------------------------
# Others

# via https://pages.mtu.edu/~suits/notefreqs.html
# lower case means "sharp" for now
FREQUENCY_BY_NOTES = {
    "A": 440,
    "a": 466.16,
    "B": 493.88,
    "C": 523.25,
    "c": 554.37,
    "D": 587.33,
    "d": 622.25,
    "E": 659.25,
    "F": 698.46,
    "f": 739.99,
    "G": 783.99,
    "g": 830.61,
}


class PeriodicTriggerSource(base.Module):
    
    bpm: base.SingleValueModule = base.Constant(130)
    notes_per_beat: int = 1

    def out_given_inputs(self, clock_signal: base.ClockSignal, bpm: float):
        sample_rate = clock_signal.sample_rate
        # Number of samples in a bar, as determined by the BPM.
        samples_per_bar = round(sample_rate / (bpm / 60))
        samples_per_note = round(samples_per_bar / self.notes_per_beat)

        # Elements are in {True, False}.
        triggers = clock_signal.sample_indices % samples_per_note == 0
        # Elements are in {1, 0}.
        return clock_signal.ones() * triggers


class ASCIITriggerSource(base.Module):

    bpm: base.SingleValueModule = base.Constant(130)
    beats: str = "X.."
    notes_per_beat: int = 4

    def setup(self):
        cycler = Cycler([1 if c == "X" else 0 for c in self.beats])
        triggers = PeriodicTriggerSource(self.bpm, self.notes_per_beat)
        self.out = TriggerModulator(cycler, triggers)


class ASCIIMelody(base.Module):

    bpm: base.SingleValueModule = base.Constant(130)
    melody: str = "A..G.."
    # TODO: Should be a module but for this, the Cylcer
    # has to take modules!!!
    octave: int = 0
    notes_per_beat: int = 4

    def setup(self):
        beats = "".join("." if c == "." else "X" for c in self.melody)
        self.triggers = ASCIITriggerSource(self.bpm, beats, self.notes_per_beat)
        octave_offset = 2 ** self.octave
        cycler = Cycler([FREQUENCY_BY_NOTES[c] * octave_offset
                         for c in self.melody if c != "."])
        self.out = Hold(TriggerModulator(cycler, self.triggers))
    

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
            [np.ones((end-start,)) * src[start]
             for start, end in zip(indices_non_zero, indices_non_zero[1:])],
            axis=0)
        self.prev_value = res[-1]
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
    match_clock_shape: bool = False

    def setup(self):
        self.current_i = base.Stateful(0)

    def out(self, clock_signal: base.ClockSignal):
        el = self.seq[min(self.current_i, len(self.seq)-1)]
        self.current_i = (self.current_i + 1) % len(self.seq)
        if self.match_clock_shape:
            return el * clock_signal.ones()
        else:
            return np.array([el], dtype=clock_signal.get_out_dtype())

 
@helpers.mark_for_testing(shape_maker=lambda: Cycler((1, 2, 3)),
                          triggers=PeriodicTriggerSource)
class TriggerModulator(base.Module):
    """Puts a shape onto an trigger.

    The `shape_maker` can be an arbitrary module. It will be called
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
        print("clock signal length", len(clock_signal.ts))
        trigger_indices = np.nonzero(self.triggers(clock_signal))[0]
        # trigger_indices = []
        # for ti in trigger_indices_:
        #     trigger_indices.append(clock_signal.sample_indices[ti])
        # print("#####trigger_indices", trigger_indices)
        # ???
        envelopes = [self.shape_maker(clock_signal.clock.get_clock_signal_with_start(i, length=len(clock_signal.ts)))
                     for i in trigger_indices]
        # import matplotlib.pyplot as plt
        # for x in envelopes:
        #     plt.plot(x)
        # plt.show()

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
            current_signal = np.pad(current_signal, pad_width=((0, remainder)))
            for i, envelope in zip(trigger_indices, envelopes):
                current_signal[i:i + len(envelope)] = envelope
        # combine the old and new signal using the given method
        max_len = max(len(previous_signal), len(current_signal))
        previous_signal = np.pad(previous_signal, pad_width=((0, max_len - len(previous_signal))))
        current_signal = np.pad(current_signal, pad_width=((0, max_len - len(current_signal))))
        result = self.combinator(previous_signal, current_signal)
        self.previous = result[clock_signal.num_samples:]
        res = result[:clock_signal.num_samples]
        return res


# class RealTriggerModulator(base.Module):
#     """
#     Simplified OldTriggerModulator. Put an envelope on a trigger track.
#     Stateful:
#     Put an envelope on every trigger. If result is longer than a frame, keep the rest for the next call.
#     Combine overlaps with a suitable function: max, fst, snd, add, ...
#     """
#
#
#
#     def __init__(self, trigger_signal: sources.TriggerSource, envelope_gen: base.BaseModule, combinator=np.add):
#         super().__init__()
#         self.previous = None
#         self.trigger_signal = trigger_signal
#         self.env_gen = envelope_gen
#         self.combinator = combinator
#
#     def __call__(self, clock_signal: ClockSignal):
#         """Generate one envelope per trigger"""
#         trigger_indices = np.nonzero(self.trigger_signal(clock_signal))[0]
#         envelopes = self.env_gen(clock_signal, desired_indices=trigger_indices)
#         current_signal = clock_signal.zeros()
#         previous_signal = self.previous if self.previous is not None and len(self.previous) > 0 else np.zeros(
#             shape=clock_signal.ts.shape)
#         if envelopes:
#             # does any envelope go over frame border?
#             latest_envelope_end = max(i + len(env) for i, env in zip(trigger_indices, envelopes))
#             if latest_envelope_end > clock_signal.num_samples:
#                 remainder = latest_envelope_end - clock_signal.num_samples
#             else:
#                 remainder = 0
#             current_signal = np.pad(current_signal, pad_width=((0, remainder), (0, 0)))
#             for i, envelope in zip(trigger_indices, envelopes):
#                 current_signal[i:i + len(envelope)] = envelope.reshape((-1, 1))
#                 # plt.plot(envelope)
#                 # plt.show()
#         # combine the old and new signal using the given method
#         max_len = max(len(previous_signal), len(current_signal))
#         previous_signal = np.pad(previous_signal, pad_width=((0, max_len - len(previous_signal)), (0, 0)))
#         current_signal = np.pad(current_signal, pad_width=((0, max_len - len(current_signal)), (0, 0)))
#         result = self.combinator(previous_signal, current_signal)
#         self.previous = result[len(clock_signal.ts):]
#         res = result[:len(clock_signal.ts)]
#         return res

# rhythm modules

@dataclasses.dataclass(eq=True, unsafe_hash=True)
class Pattern:
    """
    Input to TriggerSource. E.g., ([1, 0, 1, 0], 1/4) or ([0, 1, 1], 1/8)

    pattern: Which note (relative pitch) is played in which order?
    note_values: distance between note starts (1/3, 1/4, 1/8 etc). This is independent of note_lengths.
    note_lengths: how long should every note in the pattern sound? Will be used as hold parameter of the envelope.
    So a break (no sound) can be achieved with a 0 in the pattern or a note_length shorter than note_values.

    Drum patterns don't need a note_lengths.
    """
    pattern: Tuple
    note_values: float
    note_lengths: Optional[Tuple]


class NewTriggerSource(base.BaseModule):
    """
    Repeat the given pattern. The effective pattern duration is pattern.note_lengths*len(pattern.pattern). The
    individual notes can be shorter or longer than pattern.note_lengths, as given in pattern.note_values.
    The output for a given frame is a set of index,value pairs, signifying that at time clock_signal[index], the
    output is equal to value, and 0 at all other times.
    A NewTriggerModulator is necessary to turn this output into a signal.
    Hold for an actual signal of values/0 of length clock_signal.
    """
    bpm: base.Module
    pattern: Pattern
    hold: bool = False

    @cache
    def _make_two_bars(self, bpm, hold):  # not just one bar, to simplify wraparound
        samples_per_bar = 4 * base.SAMPLING_FREQUENCY / (bpm / 60)
        # duration of pattern in bars
        pattern, note_values, note_lengths = self.pattern
        pattern_duration_bars = note_lengths * len(pattern)  # e.g. 6 notes á 1/4 = 6/4 = 1.5 bars
        pattern_duration_samples = int(samples_per_bar * pattern_duration_bars)
        bar = np.zeros(shape=(pattern_duration_samples * 2,))
        # at intervals of note_length, set either a single value or all (if hold)
        samples_between_notes = pattern_duration_samples / len(pattern)
        indices = []
        for index, value in enumerate(list(pattern)*2):
            note_length = note_lengths[index % len(note_lengths)]
            indices.append((index * samples_between_notes, value))
            if not hold:  # single value at index
                bar[index * samples_between_notes] = value
            else:
                bar[index * samples_between_notes:index * samples_between_notes + note_length] = value
        return bar, indices

    def out(self, clock_signal: base.ClockSignal):
        bpm = np.mean(self.bpm(clock_signal))
        two_bars, indices = self._make_two_bars(bpm, self.hold)  # indices is list of tuples of index, value
        # where is frame start in the pattern?
        frame_start = clock_signal.sample_indices[0] % len(two_bars)
        frame_end   = clock_signal.sample_indices[-1] % len(two_bars)
        if self.hold:
            return two_bars[frame_start:frame_end]
        # we are interested in all the index,value pairs where the index is between frame_start and frame_end
        # and we need to shift these indices into the frame
        ixs_of_interest = [(frame_start + i % len(clock_signal.ts), v) for (i, v) in indices if frame_start <= i <= frame_end]
        return ixs_of_interest





class TriggerSource(base.Module):
    """Take patterns and bpm and acts as a source of a single trigger track.
    Set use_values to true if you don't just want ones in the output
    """

    bpm: base.Module
    pattern: Pattern
    use_values: bool = False

    @staticmethod
    def pattern_to_trigger_indices(clock_signal, samples_per_beat, pattern, note_value):
        frame_len = clock_signal.num_samples
        samples_per_note = round(samples_per_beat * note_value * 4)
        spaced_trigger_indices = np.nonzero(pattern)[0] * samples_per_note
        trigger_pattern_length = len(pattern) * samples_per_note
        # what pattern-repetition are we at? where (offset) inside the frame is the current pattern-repetition?
        offset = clock_signal.sample_indices[0] % trigger_pattern_length
        reps = int(np.ceil(frame_len / trigger_pattern_length))
        # print("reps", reps, frame_len, "/", trigger_pattern_length)
        trigger_frames = np.concatenate(
            [np.array(spaced_trigger_indices) + (i * trigger_pattern_length) for i in range(reps + 1)])
        trigger_frames = np.array(list(filter(lambda x: offset <= x < offset + frame_len, trigger_frames)))
        trigger_frames = trigger_frames - offset

        # also return rotated pattern:
        pos_in_trigger_frame = clock_signal.sample_indices[0] % trigger_pattern_length
        percentage_in_trigger_frame = pos_in_trigger_frame / trigger_pattern_length
        index = int(np.round(percentage_in_trigger_frame * len(pattern)))
        rot_pat = np.roll(pattern, -index)
        # print("rot_pat", rot_pat)
        return trigger_frames.astype(int), rot_pat

    def out(self, clock_signal: base.ClockSignal) -> np.ndarray:
        bpm = np.mean(self.bpm(clock_signal))
        samples_per_beat = base.SAMPLING_FREQUENCY / (bpm / 60)  # number of samples between 1/4 triggers
        if samples_per_beat < 2.0:
            print("Warning: Cannot deal with samples_per_beat < 2")  # TODO: but should!
            samples_per_beat = 2.0
        trigger_indices, rotated_pattern = TriggerSource.pattern_to_trigger_indices(clock_signal, samples_per_beat,
                                                                                    self.pattern.pattern,
                                                                                    self.pattern.note_values)
        trigger_signal = clock_signal.zeros()
        print("@@ trigger signal", len(trigger_signal))
        if not self.use_values:
            trigger_signal[trigger_indices] = 1.0
        else:
            reps = int(np.ceil(len(trigger_indices) / len(rotated_pattern)))
            repeated = np.tile(rotated_pattern, reps)
            if len(trigger_indices) > 0:  # TODO: there is a shape bug here
                trigger_signal[trigger_indices] = repeated[:len(trigger_indices)]
            #print([x for x in trigger_signal if x > 0])
        #print("trigger signal len", len(trigger_signal), trigger_signal)
        return trigger_signal


class Track(base.Module):
    bpm: base.Module
    pattern: Pattern
    env_gen: base.BaseModule
    combinator: Callable = np.add

    def setup(self):
        trigger_source = TriggerSource(self.bpm, self.pattern, use_values=True) >> base.Collect("ts")
        self.out = TriggerModulator(triggers=trigger_source, shape_maker=self.env_gen, combinator=self.combinator)


class NoteTrack(base.Module):
    bpm: base.Module
    note_pattern: NotePattern
    # a BaseModule constructor with a hole of type length: Module
    env_gen: Callable[[base.Module], base.BaseModule] = lambda t: envelopes.RectangleEnvGen(length=t)
    carrier_waveform: Callable[[base.Module], base.Module] = lambda t: SineSource(frequency=t)
    carrier_base_frequency: base.Module = base.Constant(220)
    combinator: Callable = np.add

    def setup(self):
        samples_per_bar = base.SAMPLING_FREQUENCY / (self.bpm / 60) * 4 >> base.Collect("samples per bar")
        # config.envelope_gen takes a length module and produces an env_gen with the user's params and length
        hold_signal = Hold(TriggerSource(bpm=self.bpm,
                                         pattern=Pattern(pattern=self.note_pattern.note_lengths,
                                                         note_values=1 / len(self.note_pattern.note_lengths)),
                                         use_values=True
                                         ) >> base.Collect("triggers")
                           ) >> base.Collect("hold")
        print(Pattern(pattern=self.note_pattern.note_lengths, note_values=1 / len(self.note_pattern.note_lengths)))
        env_length = samples_per_bar * hold_signal >> base.Collect("env_length")
        env_gen = self.env_gen(env_length) >> base.Collect("env_gen")
        track = Track(bpm=self.bpm,
                      pattern=self.note_pattern,
                      env_gen=env_gen,
                      combinator=self.combinator) >> base.Collect("track")

        notes = tuple([base.FreqFactors.STEP.value ** n for n in self.note_pattern.pattern])
        notes_pattern = Pattern(pattern=notes, note_values=self.note_pattern.note_values)
        carrier = self.carrier_waveform(self.carrier_base_frequency * Hold(TriggerSource(bpm=self.bpm,
                                                                                         pattern=notes_pattern,
                                                                                         use_values=True)))
        modulated = carrier * track
        self.out = modulated







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
