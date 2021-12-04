import dataclasses
import functools
import operator

from modules import ClockSignal, Clock, Module, Parameter, Random, SineSource, SawSource, TriangleSource, \
    SAMPLING_FREQUENCY, NoiseSource, Constant, Id, FreqFactors
import random
import numpy as np
from typing import Dict, List, NamedTuple, Callable, Union

import matplotlib.pyplot as plt

P = Parameter


class EnvelopeGen(Module):  # TODO: This is ONLY a Module to make Parameter keying work!
    def __mul__(self, other):
        return _MathEnvGen(operator.mul, self, other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return _MathEnvGen(operator.truediv, self, other)

    def __rtruediv__(self, other):
        return _MathEnvGen(operator.truediv, other, self)

    def __add__(self, other):
        return _MathEnvGen(operator.add, self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return _MathEnvGen(operator.sub, self, other)

    def __rsub__(self, other):
        return _MathEnvGen(operator.sub, other, self)

    def __or__(self, other):
        return _MathEnvGen(lambda first, second: np.concatenate([first, second]), self, other)

    def __lshift__(self, other):
        # add zeros to the right
        return self | (RectangleEnvelopeGen(length=Constant(other)) * 0.0)

    def __rshift__(self, other):
        # add zeros to the left
        return (RectangleEnvelopeGen(length=Constant(other)) * 0.0) | self


class _MathEnvGen(EnvelopeGen):
    """Borrowed from modules._MathModule"""

    def __init__(self, op, left, right):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

    def __call__(self, clock_signal: ClockSignal, desired_indices):
        left = self._maybe_call(self.left, clock_signal, desired_indices)
        right = self._maybe_call(self.right, clock_signal, desired_indices)
        return [self.op(le, ri) for le, ri in zip(left, right)]

    @staticmethod
    def _maybe_call(env_gen_or_number, clock_signal, desired_indices):
        if isinstance(env_gen_or_number, EnvelopeGen):
            return env_gen_or_number(clock_signal, desired_indices)
        return np.array([env_gen_or_number])  # so we can broadcast the number


class EnvelopeSource(Module):
    def __init__(self, envelope_gen: EnvelopeGen):
        super().__init__()
        self.envelope_gen = envelope_gen
        self.sign_exponent = 0  # TODO: later haha

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        env = self.envelope_gen(clock_signal, [0])[0]  # get a single envelope, and unpack from list TODO: why not [1]?
        start = clock_signal.sample_indices[0] % len(env)
        signal = env
        while len(signal) < len(clock_signal.ts):
            signal = np.concatenate([signal, env * (-1) ** self.sign_exponent])
        signal = np.roll(signal, -start)
        res = np.reshape(signal[:len(clock_signal.ts)], newshape=clock_signal.ts.shape)
        self.collect("dings") << res
        return res


############################################################################################################
# Api for envelope generators: They pass clock_signal to their param-sources, but only generate an envelope
# for desired indices. Those are clear from the trigger signal of the calling function.
# Therefore, __call__ takes a clock signal, a list of desired indices and returns a list of envelopes.


def func_gen(func, num_samples, curvature, start_val=0, end_val=1):
    """Produce num_samples samples of func between [0, curvature], but squished into [0,1]"""
    num_samples = int(num_samples)
    # print(func, num_samples, curvature, start_val, end_val)
    xs = func(np.linspace(0, curvature, num_samples))
    xs = (xs - xs[0]) / (np.max(xs) - xs[0])
    return xs * (end_val - start_val) + start_val


class FuncEnvelopeGen(EnvelopeGen):
    def __init__(self, func: Callable, length, curvature, start_val=Constant(0), end_val=Constant(1)):
        self.func = func
        self.length = length
        self.curvature = curvature
        self.start_val = start_val
        self.end_val = end_val

    def __call__(self, clock_signal: ClockSignal, desired_indices):
        length = self.length(clock_signal)
        curvature = self.curvature(clock_signal)
        start_val = self.start_val(clock_signal)
        end_val = self.end_val(clock_signal)
        res = []
        for i in desired_indices:
            res.append(func_gen(self.func, length[i, 0], curvature[i, 0], start_val[i, 0], end_val[i, 0]))
        return res


class ExpEnvelopeGen(EnvelopeGen):
    """An exp'ly rising edge followed by an exp'ly falling edge.
    Equivalent to FuncEnvelopeGen(np.exp, attack...) | FuncEnvelopeGen(lambda t: np.log(1+t), decay...)"""

    def __init__(self, attack_length, decay_length, attack_curvature, decay_curvature):
        self.attack_length = attack_length
        self.decay_length = decay_length
        self.attack_curvature = attack_curvature
        self.decay_curvature = decay_curvature

    def __call__(self, clock_signal: ClockSignal, desired_indices):
        attack_length = self.attack_length(clock_signal)
        decay_length = self.decay_length(clock_signal)
        attack_curvature = self.attack_curvature(clock_signal)
        decay_curvature = self.decay_curvature(clock_signal)

        res = []
        for i in desired_indices:
            attack = func_gen(np.exp, attack_length[i, 0], attack_curvature[i, 0])
            decay = func_gen(lambda t: np.log(1 + t), decay_length[i, 0], decay_curvature[i, 0], 1, 0)
            envelope = np.concatenate((attack, decay), 0)
            res.append(envelope)
        return res


class RectangleEnvelopeGen(EnvelopeGen):
    """Equivalent to FuncEnvelopeGen(func=lambda t: t, num_samples=length, curvature=1, start_val=1, end_val=1)"""

    def __init__(self, length: Module):
        self.length = length

    def __call__(self, clock_signal: ClockSignal, desired_indices):
        length = self.length(clock_signal)[0, 0]
        return [np.ones((length,)) for i in desired_indices]


class ADSREnvelopeGen(EnvelopeGen):
    """Borrowed from modules.py. Equivalent to a sum of FuncEnvelopeGens."""

    # TODO: rewrite as FuncEnvelopeGen concatenation
    def __init__(self, attack: Module, decay: Module, sustain: Module, release: Module, hold: Module):
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.hold = hold

    def __call__(self, clock_signal: ClockSignal, desired_indices):
        t_attack = self.attack.out_mean_int(clock_signal)
        t_decay = self.decay.out_mean_int(clock_signal)
        sustain_height = self.sustain.out_mean_int(clock_signal)
        t_hold = self.hold.out_mean_int(clock_signal)
        t_release = self.release.out_mean_int(clock_signal)

        res = []
        for i in desired_indices:
            attack = np.linspace(0, 1, t_attack)
            decay = np.linspace(1, sustain_height, t_decay)
            hold = np.ones(t_hold) * sustain_height
            release = np.linspace(sustain_height, 0, t_release)
            envelope = np.concatenate((attack, decay, hold, release), 0)
            res.append(envelope)
        return res


#######################################################################################################


@dataclasses.dataclass
class Pattern:
    """Input to TriggerSource. E.g., ([1, 0, 1, 0], 1/4) or ([0, 1, 1], 1/8)"""
    pattern: List[float]
    note_values: float
    note_lengths: List[float] = None  # not used by percussion instruments


class TrackConfig(NamedTuple):
    """Input to Track"""
    pattern: Pattern
    envelope_gen: EnvelopeGen
    # when an envelope is not enough: modulate to carrier, filter, add noise, ...
    # must have type Module -> Module
    post: Callable[[Module], Module] = lambda m: Id(m)
    combinator: Callable = np.add  # a property of the "instrument": how to combine overlapping notes?


class TriggerSource(Module):
    """Take patterns and bpm and acts as a source of a single trigger track.
    Set use_values to true if you don't just want ones in the output
    """

    def __init__(self, bpm: Module, pattern: Pattern, use_values=False):
        super().__init__()
        self.bpm = bpm
        self.pattern = pattern
        self.use_values = use_values

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

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        bpm = np.mean(self.bpm(clock_signal))
        samples_per_beat = SAMPLING_FREQUENCY / (bpm / 60)  # number of samples between 1/4 triggers
        if samples_per_beat < 2.0:
            print("Warning: Cannot deal with samples_per_beat < 2")  # TODO: but should!
            samples_per_beat = 2.0
        trigger_indices, rotated_pattern = TriggerSource.pattern_to_trigger_indices(clock_signal, samples_per_beat,
                                                                                    self.pattern.pattern,
                                                                                    self.pattern.note_values)
        trigger_signal = clock_signal.zeros()
        if not self.use_values:
            trigger_signal[trigger_indices] = 1.0
        else:
            reps = int(np.ceil(len(trigger_indices) / len(rotated_pattern)))
            repeated = np.tile(rotated_pattern, reps)
            if len(trigger_indices) > 0:
                trigger_signal[trigger_indices] = repeated[:len(trigger_indices)]
        return trigger_signal


class TriggerModulator(Module):
    """
    Simplified OldTriggerModulator. Put an envelope on a trigger track.
    Stateful:
    Put an envelope on every trigger. If result is longer than a frame, keep the rest for the next call.
    Combine overlaps with a suitable function: max, fst, snd, add, ...
    """

    def __init__(self, trigger_signal: TriggerSource, envelope_gen: EnvelopeGen, combinator=np.add):
        super().__init__()
        self.previous = None
        self.trigger_signal = trigger_signal
        self.env_gen = envelope_gen
        self.combinator = combinator

    def __call__(self, clock_signal: ClockSignal):
        """Generate one envelope per trigger"""
        trigger_indices = np.nonzero(self.trigger_signal(clock_signal))[0]
        envelopes = self.env_gen(clock_signal, desired_indices=trigger_indices)
        current_signal = clock_signal.zeros()
        previous_signal = self.previous if self.previous is not None and len(self.previous) > 0 else np.zeros(
            shape=clock_signal.ts.shape)
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
                # plt.plot(envelope)
                # plt.show()
        # combine the old and new signal using the given method
        max_len = max(len(previous_signal), len(current_signal))
        previous_signal = np.pad(previous_signal, pad_width=((0, max_len - len(previous_signal)), (0, 0)))
        current_signal = np.pad(current_signal, pad_width=((0, max_len - len(current_signal)), (0, 0)))
        result = self.combinator(previous_signal, current_signal)
        self.previous = result[len(clock_signal.ts):]
        res = result[:len(clock_signal.ts)]
        return res


# now, we compose like this:
# Pattern -> TriggerSource() -> triggerIndices -> TriggerModulator(env_gen)
# for example, package Pattern and env_gen together in TrackConfig

class Track(Module):
    """A single repeating pattern with its own envelope gen"""

    def __init__(self, bpm: Module, config: TrackConfig):
        super().__init__()
        self.bpm = bpm
        self.pattern = config.pattern
        self.env_gen = config.envelope_gen
        self.combinator = config.combinator
        self.post = config.post  # by default, the identity Module -> no effect

        self.trigger_source = TriggerSource(self.bpm, self.pattern)
        self.trigger_modulator = TriggerModulator(trigger_signal=self.trigger_source, envelope_gen=self.env_gen,
                                                  combinator=self.combinator)
        # TODO: this self.post stuff is a bit questionable.. having lambda m: X(m) as args...
        self.out = self.post(self.trigger_modulator)


# need: source that produces tone until trigger. the note will switch to the given frequency.
# single notes possible, but also chords
# pattern notation: 0: no tone, 1: base tone, and then rt(2,12)**higher, so that 12 is an octave higher than 1.
# note pattern: [1, 3, 1, 4, 1, 4, 3, 1], 1/8
# note lengths: by default [1/8]*8, but multiplied by parameter, and can pass so that not all are equal
#
# so: make an envelope_gen that 1. makes correct length, and 2. the correct tones.
#

# need step signal which takes indices and values and when index is reached, it takes value=values[index]
# as input to frequency of a SineSource
#

class Hold(Module):
    """A trigger has a value, and the output is a step signal where after after trigger1, the value of the signal is
    the value of trigger1 and so on.."""

    def __init__(self, inp: Module):
        self.inp = inp
        self.previous_value = 0.0

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        inp = self.inp(clock_signal)
        # add first trigger if not already present
        if inp[0] == 0.0:
            first_val = np.ones((1, clock_signal.shape[1])) * self.previous_value
            values = np.concatenate((first_val, inp))
            first_slice_index = np.ones((1, clock_signal.shape[1]))
            slice_indices = np.nonzero((np.concatenate((first_slice_index, inp))))[0]
        else:
            values = inp
            slice_indices = np.nonzero(inp)[0]
        if len(slice_indices) > 0:
            self.previous_value = values[slice_indices[-1]][0]
            # add last index if not already present
            if slice_indices[-1] != clock_signal.num_samples:
                slice_indices = np.append(slice_indices, clock_signal.num_samples)
            # create chunks and concat
            chunks = []
            for i, index in enumerate(slice_indices[:-1]):
                chunks.append(np.ones((slice_indices[i + 1] - index)) * values[index, :])
            out = np.concatenate(chunks)
        else:
            out = np.zeros_like(clock_signal.ts)
        return out.reshape(clock_signal.shape)


class HoldTest(Module):
    def __init__(self):
        pattern = Pattern(pattern=[1, 2, 3, 0], note_values=1 / 4, note_lengths=[1 / 8, 1 / 8, 1 / 4, 1 / 8])
        self.trigger_src = TriggerSource(Parameter(120, key="b"), pattern, use_values=True)
        self.hold = Hold(self.trigger_src)
        self.out = SineSource(frequency=self.hold * 110)


class InstrumentTrack(Module):
    """If envelope_gen in config is None, create a window env gen with length note_lengths. Otherwise, use the given"""

    def __init__(self, bpm: Module, config: TrackConfig):
        # split config.Pattern into pattern and note values, and get the note lengths too. the pattern will be passed
        # to Track(...), but the note values and lengths go to the envelope generator, which is injected to Track too

        samples_per_bar = SAMPLING_FREQUENCY / (bpm / 60) * 4

        if config.envelope_gen is None:
            note_lengths = config.pattern.note_lengths if config.pattern.note_lengths is not None \
                else [config.pattern.note_values] * len(config.pattern.pattern)
            note_len_vals = 1.0 / len(note_lengths)
            env_gen = FuncEnvelopeGen(func=lambda t: t,
                                      length=samples_per_bar * Hold(TriggerSource(bpm=bpm,
                                                                                  pattern=Pattern(note_lengths,
                                                                                                  note_len_vals),
                                                                                  use_values=True)),
                                      curvature=Constant(1),
                                      start_val=Constant(1),
                                      end_val=Constant(1))
        else:
            env_gen = config.envelope_gen
        # envelope gens need a certain length, but are otherwise rectangular windows... for now. later: attack TODO
        # posts
        # need note values to make carrier
        notes = [FreqFactors.STEP.value ** n for n in config.pattern.pattern]
        notes_pattern = Pattern(pattern=notes, note_values=config.pattern.note_values)
        carrier = TriangleSource(frequency=220 * Hold(TriggerSource(bpm=bpm, pattern=notes_pattern, use_values=True)))
        post = lambda m: m * carrier
        self.out = Track(bpm=bpm, config=TrackConfig(pattern=config.pattern, envelope_gen=env_gen, post=post))


"""
    Parametrize with trigger patterns for different tracks (kick, snare, hihat, ...).
    Trigger patterns go over any number of bars and repeat forever.
    The trigger patterns live in bar-time, not sample-time:
    The trigger patterns will be spaced out in time according to the OldDrumMachine's bpm.
    A beat (as in bpm) is 1/4 of a bar.
    A trigger has no length! The length of an envelope is the envelope generator's concern.
    Combining overlapping envelopes is the OldTriggerModulator's concern.
    Ways to write down a trigger pattern:
        Direct:     [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0], (1 bar) or (1/16) or (2 bars) etc
                    Needs additional information: Either how many bars this is, or the note length.
        Inverse:    Each entry is the inverse of the length of the pause after the trigger.
                    [4,4,4,4] -> [1,1,1,1], (1 bar)
                    [2,4,4] -> [1,0,1,1]
                    [2,2] -> [1,0,1,0]
                    Downside: How to do offbeats, or other patterns that don't start at 1?
        ...?
    Currently supported: Direct with note_value, see OldTrack class.
    Every track has its own envelope generator and a postprocessor wavefunction to control the pitch.
"""


class DrumMachine(Module):
    def __init__(self, bpm: Module, track_cfg_dict: Dict[str, TrackConfig]):
        self.bpm = bpm
        self.out_dict = {name: Track(self.bpm, track_cfg) for name, track_cfg in track_cfg_dict.items()}
        self.out = np.sum([out for out in self.out_dict.values()])


class NewDrumTest(Module):
    def __init__(self):
        bpm = Parameter(120, key='b')

        kick_env = FuncEnvelopeGen(func=np.exp, length=P(100), curvature=P(3)) | \
                   FuncEnvelopeGen(func=np.exp, length=P(1000), curvature=P(2), start_val=Constant(1),
                                   end_val=Constant(0))
        snare_env = ExpEnvelopeGen(attack_length=P(200), attack_curvature=P(10), decay_length=P(30),
                                   decay_curvature=P(3)) | \
                    ExpEnvelopeGen(attack_length=P(50), attack_curvature=P(5), decay_length=P(500),
                                   decay_curvature=P(1000))
        hihat_env = ExpEnvelopeGen(attack_length=P(400), attack_curvature=P(3), decay_length=P(800),
                                   decay_curvature=P(200)) * 0.6

        track_dict = {
            "kick": TrackConfig(pattern=Pattern([1, 0, 1, 0, 1, 0, 1, 0] * 4 + [1, 0, 1, 0, 1, 1, 1, 1], 1 / 8),
                                envelope_gen=kick_env,
                                post=lambda m: m * (TriangleSource(frequency=P(60)) + NoiseSource() * 0.05)
                                ),
            "snare": TrackConfig(pattern=Pattern([0, 0, 1, 0, 0, 0, 1, 1] * 4 + [1, 0, 1, 0, 1, 0, 1, 1], 1 / 8),
                                 envelope_gen=snare_env,
                                 post=lambda m: m * (TriangleSource(frequency=P(1000)) + NoiseSource() * 0.6)
                                 ),
            "hihat": TrackConfig(pattern=Pattern([0, 1, 0, 1, 0, 1, 0, 1] * 3 + [1, 1, 1, 1, 1, 1, 1, 1] * 2, 1 / 8),
                                 envelope_gen=hihat_env,
                                 post=lambda m: m * NoiseSource()
                                 ),
        }

        percussion = DrumMachine(bpm=bpm, track_cfg_dict=track_dict)

        note_track = TrackConfig(pattern=Pattern(pattern=random.choices([0, 1, 3, 7, 12, 14, 18, 24], k=8),
                                                 note_values=random.choice([1 / 4, 1 / 8, 3 / 8, 1 / 16, 3 / 16]),
                                                 note_lengths=random.choices(
                                                     [1 / 4, 1 / 8, 3 / 8, 1 / 16, 3 / 16, 1 / 32, 3 / 32, 1 / 64,
                                                      3 / 64, 1 / 128, 3 / 128], k=8)),
                                 envelope_gen=None,
                                 post=None
                                 )

        instruments1 = InstrumentTrack(bpm=bpm, config=note_track)

        note_track2 = TrackConfig(pattern=Pattern(pattern=random.choices([0, 0.1, 0.3, 0.7, 1.2, 1.4, 1.8, 2.4], k=8),
                                                 note_values=random.choice([1/2, 1 / 4, 3 / 4, 3 / 8]),
                                                 note_lengths=random.choices(
                                                     [1 / 4, 1 / 8, 3 / 8, 1 / 16, 3 / 16], k=8)),
                                 envelope_gen=None,
                                 post=None
                                 )

        instruments2 = InstrumentTrack(bpm=bpm, config=note_track2)

        self.out = percussion + instruments1 * 0.1 + instruments2 * 0.1
