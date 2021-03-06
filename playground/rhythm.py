import dataclasses
import functools
import scipy.signal
import operator

from numpy.polynomial import Polynomial

from playground.modules import ClockSignal, Clock, Module, Parameter, Random, SineSource, SawSource, TriangleSource, \
    SAMPLING_FREQUENCY, NoiseSource, Constant, Id, FreqFactors, FrameBuffer, ButterworthFilter
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
        return self | (RectangleEnvGen(length=Constant(other)) * 0.0)

    def __rshift__(self, other):
        # add zeros to the left
        return (RectangleEnvGen(length=Constant(other)) * 0.0) | self


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
    num_samples = max(1, int(num_samples))
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


class ConstEnvGen(EnvelopeGen):
    """Pass a vector which will be returned every time."""

    def __init__(self, vector):
        self.vector = vector

    def __call__(self, clock_signal, desired_indices):
        return [self.vector for i in desired_indices]


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


class RectangleEnvGen(EnvelopeGen):
    """Equivalent to FuncEnvelopeGen(func=lambda t: t, num_samples=length, curvature=1, start_val=1, end_val=1)"""

    def __init__(self, length: Module):
        self.length = length

    def __call__(self, clock_signal: ClockSignal, desired_indices):
        length = int(self.length(clock_signal)[0, 0])
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


class MultiSource(Module):
    def __init__(self, base_frequency: Module, source: Callable[[Module], Module], num_overtones: int, randomize_phases=True):
        """The source parameter is a module constructor"""
        super().__init__()
        self.freqs = [i * base_frequency for i in range(1, num_overtones + 1)]
        self.amps = [1.0/i for i in range(1, num_overtones + 1)]

        def _phase():
            return Constant(random.random()*2*np.pi) if randomize_phases else Constant(0)
        self.out = sum(source(frequency=freq, phase=_phase()) * amp for freq, amp in zip(self.freqs, self.amps)) / sum(self.amps)

#######################################################################################################


@dataclasses.dataclass
class Pattern:
    """Input to TriggerSource. E.g., ([1, 0, 1, 0], 1/4) or ([0, 1, 1], 1/8)"""
    pattern: List[float]
    note_values: float


@dataclasses.dataclass
class TrackConfig:
    """Input to Track"""
    pattern: Pattern
    envelope_gen: EnvelopeGen
    post: Callable[[Module], Module] = Id  # modulate to carrier, filter, add noise, ...
    combinator: Callable = np.add  # a property of the "instrument": how to combine overlapping notes?


@dataclasses.dataclass
class NotePattern(Pattern):
    """How long should every note in the pattern sound? Will be used as hold parameter of the envelope."""
    note_lengths: List[float]


# TODO: this child of TrackConfig needs default params even though they don't necessarily make sense, bc dataclass...
@dataclasses.dataclass
class NoteTrackConfig(TrackConfig):
    """The envelope_gen takes the lengths from the NotePattern as length inputs."""
    pattern: NotePattern = None
    # takes a length Module and gives an envelope with desired specs
    envelope_gen: Callable[[Module], EnvelopeGen] = lambda t: RectangleEnvGen(length=t)
    carrier_waveform: Callable[[Module], Module] = lambda t: SineSource(frequency=t)
    carrier_base_frequency: Module = Constant(440)


p1 = Pattern(pattern=[1, 2, 3], note_values=1 / 4)
p2 = NotePattern(pattern=[1, 2, 3, 4], note_lengths=[1 / 2, 1 / 4, 1 / 2, 1 / 4], note_values=1 / 4)


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
            if len(trigger_indices) > 0: # TODO: there is a shape bug here
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


class NoteTrack(Module):
    """If envelope_gen in config is None, create a window env gen with length note_lengths. Otherwise, use the given"""

    def __init__(self, bpm: Module, config: NoteTrackConfig):
        samples_per_bar = SAMPLING_FREQUENCY / (bpm / 60) * 4

        # config.envelope_gen takes a length module and produces an env_gen with the user's params and length
        hold_signal = Hold(TriggerSource(bpm=bpm,
                                         pattern=Pattern(pattern=config.pattern.note_lengths,
                                                         note_values=1 / len(config.pattern.note_lengths)),
                                         use_values=True
                                         )
                           )
        env_gen = config.envelope_gen(samples_per_bar * hold_signal)

        # do not pass config.post on, because we will post _after_ lifting this track to carrier
        track_cfg = TrackConfig(pattern=config.pattern, envelope_gen=env_gen, combinator=config.combinator)
        track = Track(bpm=bpm, config=track_cfg)

        notes = [FreqFactors.STEP.value ** n for n in config.pattern.pattern]
        notes_pattern = Pattern(pattern=notes, note_values=config.pattern.note_values)
        carrier = config.carrier_waveform(config.carrier_base_frequency * Hold(TriggerSource(bpm=bpm,
                                                                                             pattern=notes_pattern,
                                                                                             use_values=True)))
        modulated = carrier * track
        post = config.post(modulated)
        self.out = post


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

        note_env = lambda length: ExpEnvelopeGen(
            attack_length=length * 0.05,
            attack_curvature=P(10),
            decay_length=length * 0.95,
            decay_curvature=P(3)
        )
        #note_env = lambda length: RectangleEnvGen(length=length)
        note_track = NoteTrackConfig(
            pattern=NotePattern(pattern=random.choices([0, 1, 3, 7, 12, 14, 18, 24], k=8),
                                note_values=random.choice([1 / 4, 1 / 8, 3 / 8, 1 / 16, 3 / 16]),
                                note_lengths=random.choices(
                                    [1 / 4, 1 / 8, 3 / 8, 1 / 16, 3 / 16, 1 / 32, 3 / 32, 1 / 64, 3 / 64, 1 / 128,
                                     3 / 128], k=8)
                                ),
            envelope_gen=note_env,
            carrier_waveform=lambda t: TriangleSource(frequency=t),
            carrier_base_frequency=Parameter(220, key='f'),
            post=lambda t: ButterworthFilter(t, f_low=Parameter(1, key='o'), f_high=Parameter(5000, key='p'), mode='bp'),
        )

        instruments1 = NoteTrack(bpm=bpm, config=note_track)

        self.out = percussion + instruments1 * 0.3


class MultiNote(Module):
    """
    WIP
    Actually envelopes of high freqs should decay quicker, so maybe we should add envs, not signals...
    """
    def __init__(self, bpm: Module, source_waveform: Callable[[Module], Module], num_overtones: int):
        waves = []
        pattern = NotePattern(pattern=random.choices([0, 1, 3, 7, 12, 14, 18, 24], k=8),
                                    note_values=random.choice([1 / 4, 1 / 8, 3 / 8, 1 / 16, 3 / 16]),
                                    note_lengths=random.choices(
                                        [1 / 4, 1 / 8, 3 / 8, 1 / 16, 3 / 16, 1 / 32, 3 / 32, 1 / 64, 3 / 64], k=8)
                              )
        base_freq = Parameter(220, key='f')
        base_phase = Parameter(2 * np.pi, key="p")
        for i in range(num_overtones):
            phase_factor = base_phase * random.random()
            amp_factor = 1/(1+i)
            length_factor = (Parameter(1, key="l", lo=0, hi=10, clip=True)/(i+1))
            freq_factor = base_freq * (i+1.0)
            note_env = lambda length: ExpEnvelopeGen(
                attack_length=length * length_factor * 0.05,
                attack_curvature=P(10),
                decay_length=length * length_factor * 0.95,
                decay_curvature=P(3)
            )
            note_track = NoteTrackConfig(
                pattern=pattern,
                envelope_gen=note_env,
                carrier_waveform=lambda t: source_waveform(frequency=t, phase=phase_factor) * amp_factor,
                carrier_base_frequency=freq_factor,
                post=Id,
            )
            wave = NoteTrack(bpm=bpm, config=note_track)
            waves.append(wave)
        self.out = sum(waves)/num_overtones


class MultiNoteTest(Module):
    def __init__(self):
        bpm = Parameter(120, key="b")
        self.out = MultiNote(bpm=bpm, source_waveform=SineSource, num_overtones=3)


class MultiSourceTest(Module):
    def __init__(self):
        self.src = MultiSource(base_frequency=Parameter(220, key='f'), source=SineSource, num_overtones=100)
        self.out = ButterworthFilter(self.src, f_low=P(10, key="o"), f_high=P(5000, key="p"), mode="bp")


class HoldTest(Module):
    def __init__(self):
        pattern = Pattern(pattern=[1, 2, 3, 0], note_values=1 / 4, note_lengths=[1 / 8, 1 / 8, 1 / 4, 1 / 8])
        self.trigger_src = TriggerSource(Parameter(120, key="b"), pattern, use_values=True)
        self.hold = Hold(self.trigger_src)
        self.out = SineSource(frequency=self.hold * 110)


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


class Reverb(Module):

    def __init__(self, src: Module,
                 delay: Module = Constant(3000),
                 echo: Module = Constant(10000),
                 p: Module = Constant(0.05)):
        super().__init__()
        self.delay = delay
        self.echo = echo
        self.p = p
        self.b = FrameBuffer()
        self.src = src

    def out(self, clock_signal: ClockSignal):
        o = self.src(clock_signal)
        num_samples, num_c = clock_signal.shape
        self.b.push(o, max_frames_to_buffer=10)
        signal = self.b.get()

        h = basic_reverb_ir(self.delay.out_mean_int(clock_signal),
                            self.echo.out_mean_int(clock_signal),
                            self.p.out_mean_float(clock_signal))

        convolved = scipy.signal.convolve(signal, h, mode="valid")
        return convolved[-num_samples:, :]


class Ufgregt(Module):

    def __init__(self):
        # kick_sample = KickSampler().make()

        kick_env = FuncEnvelopeGen(func=lambda t: np.exp(-((t - 0.2) / 1.9) ** 2) * np.sin(t * 2 * np.pi * 1.2),
                                   length=Constant(2000), curvature=P(3))
        kick_track = TrackConfig(Pattern(pattern=[1, 0, 1, 0], note_values=1 / 8), kick_env)
        self.out = Track(Parameter(120), kick_track)
        self.out = Reverb(self.out,
                          delay=P(1000, 0, 10000, knob="fx2_1"),
                          echo=P(5000, 0, 10000, knob="fx2_2"),
                          p=P(0.05, 0, 1, knob="fx2_3"))
