import dataclasses
import functools
import operator

from modules import ClockSignal, Clock, Module, Parameter, Random, SineSource, SawSource, TriangleSource, SAMPLING_FREQUENCY, plot_module, StepSequencing, NoiseSource
import random
import numpy as np
from typing import Dict, List, NamedTuple, Callable, Union

import matplotlib.pyplot as plt

P = Parameter


class EnvelopeGen:
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
        return np.array(env_gen_or_number)  # so we can broadcast the number


############################################################################################################


# TODO: generalize envelope generators:
# concatenate arbitrary number of function-segments.
# a segment's end equals the next segment's start.
# segments like: linear, exponential, logarithmic, polynomial?, sinusoidal?
# https://theproaudiofiles.com/synthesis-101-envelope-parameters-uses/
# TODO: loop envelopes to get new sources (EnvelopeSource)
# TODO: white noise generator
# TODO: should envelope generators have a parent class and implement the operations +-*/ and concat ?


def func_gen(func, num_samples, curvature, start_val=0, end_val=1):
    """Produce num_samples samples of func between [0, curvature], but squished into [0,1]"""
    xs = func(np.linspace(0, curvature, num_samples))
    xs = (xs - xs[0]) / (np.max(xs) - xs[0])
    return xs * (end_val - start_val) + start_val




# import matplotlib.pyplot as plt
# plt.plot(func_gen(lambda t: np.random.rand(len(t))*2-1, 100, 3))
# plt.show()




class ExpEnvelopeGen(EnvelopeGen):
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
            decay = func_gen(lambda t: np.log(1+t), decay_length[i, 0], decay_curvature[i, 0], 1, 0)
            envelope = np.concatenate((attack, decay), 0)
            res.append(envelope)
        return res


class RectangleEnvelopeGen(EnvelopeGen):
    def __init__(self, length: Module):
        self.length = length

    def __call__(self, clock_signal: ClockSignal, desired_indices):
        length = self.length(clock_signal)[0, 0]
        return [np.ones((length,)) for i in desired_indices]


class ADSREnvelopeGen(EnvelopeGen):
    """
    Borrowed from modules.py
    New api for envelope generators: They pass clock_signal to their param-sources, but only generate an envelope
    for desired indices. Those are clear from the trigger signal of the calling function.
    Therefore, __call__ takes a clock signal, a list of desired indices and returns a list of envelopes.
    TODO: consider if we should enhance the Module.call signature with optional desired indices.
    """

    def __init__(self, attack: Module, decay: Module, sustain: Module, release: Module, hold: Module):
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.hold = hold

    def __call__(self, clock_signal: ClockSignal, desired_indices):
        t_attack = self.attack(clock_signal)
        t_decay = self.decay(clock_signal)
        sustain_height = self.sustain(clock_signal)
        t_hold = self.hold(clock_signal)
        t_release = self.release(clock_signal)

        res = []
        for i in desired_indices:
            attack = np.linspace(0, 1, t_attack[i, 0])
            decay = np.linspace(1, sustain_height[i, 0], t_decay[i, 0])
            hold = np.ones(t_hold[i, 0]) * sustain_height[i, 0]
            release = np.linspace(sustain_height[i, 0], 0, t_release[i, 0])
            envelope = np.concatenate((attack, decay, hold, release), 0)
            res.append(envelope)
        return res


class TriggerModulator:
    """
        Put an envelope on every trigger. If result is longer than a frame, keep the rest for the next call.
        Combine overlaps with a suitable function: max, fst, snd, add, ...
        """
    def __init__(self):
        self.previous = None

    # def get_previous(self, num_samples):
    #     if self.previous is not None:
    #         chunk = self.previous[:num_samples, :]
    #         self.previous = self.previous[num_samples:, :]
    #     if len(chunk) > num_samples:
    #         chunk = np.pad(chunk, pad_width=(0, num_samples - len(chunk)))
    #     return chunk

    def __call__(self, clock_signal: ClockSignal, triggers, envelope_gen, combinator=np.add):
        """Generate one envelope per trigger"""

        trigger_indices = np.nonzero(triggers)[0]
        envelopes = envelope_gen(clock_signal, desired_indices=trigger_indices)
        current_signal = np.zeros(shape=clock_signal.ts.shape)
        previous_signal = self.previous if self.previous is not None and len(self.previous) > 0 else np.zeros(shape=clock_signal.ts.shape)
        if envelopes:
            # does any envelope go over frame border?
            latest_envelope_end = max([i + len(env) for i, env in zip(trigger_indices, envelopes)])
            if latest_envelope_end > clock_signal.num_samples:
                remainder = latest_envelope_end - clock_signal.num_samples
            else:
                remainder = 0
            current_signal = np.pad(current_signal, pad_width=((0, remainder), (0, 0)))
            for i, envelope in zip(trigger_indices, envelopes):
                current_signal[i:i+len(envelope)] = envelope.reshape((-1, 1))
                # plt.plot(envelope)
                # plt.show()
        # combine the old and new signal using the given method
        max_len = max(len(previous_signal), len(current_signal))
        previous_signal = np.pad(previous_signal, pad_width=((0, max_len - len(previous_signal)), (0, 0)))
        current_signal = np.pad(current_signal, pad_width=((0, max_len - len(current_signal)), (0, 0)))
        result = combinator(previous_signal, current_signal)
        self.previous = result[len(clock_signal.ts):]
        res = result[:len(clock_signal.ts)]
        return res


class Track(NamedTuple):
    name: str
    pattern: List[int]
    note_values: float
    envelope_gen: Callable  # regular function or Module
    carrier: Module
    trigger_modulator: TriggerModulator



class DrumMachine(Module):
    """
    Parametrize with trigger patterns for different tracks (kick, snare, hihat, ...).
    Trigger patterns go over any number of bars and repeat forever.
    The trigger patterns live in bar-space, not time:
    The trigger patterns will be spaced out in time according to the DrumMachine's bpm.
    A trigger has no length! The length of an envelope is the envelope generator's concern.
    Combining overlapping envelopes is the TriggerModulator's concern.
    Ways to write down a trigger pattern:
        Direct:     [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0], (1 bar) or (1/16) or (2 bars) etc
                    Needs additional information: Either how many bars this is, or the note length.
        Inverse:    Each entry is the inverse of the length of the pause after the trigger.
                    [4,4,4,4] -> [1,1,1,1], (1 bar)
                    [2,4,4] -> [1,0,1,1]
                    [2,2] -> [1,0,1,0]
                    Downside: How to do offbeats, or other patterns that don't start at 1?
        ...?
    Currently supported: Direct with note_value, see Track class
    Every track also has a envelope generator.
    """

    def __init__(self, bpm: Module, tracks: List[Track]):
        super().__init__()
        self.bpm = bpm
        self.tracks = tracks
        self.dummy = SineSource(frequency=P(220, key='w'))

    @staticmethod
    def extend_tracks(clock_signal, track, samples_per_beat):
        triggers = DrumMachine._track_to_triggers(track, samples_per_beat)
        # these have different lengths depending on the number of bars given. loop until end of frame.
        orig_trig = triggers[:]
        while len(triggers) < len(clock_signal.ts):
            triggers = np.append(triggers, orig_trig, axis=0)
        # we started all triggers from time 0. apply the offset
        offset = clock_signal.sample_indices[0] % len(triggers)
        triggers = np.roll(triggers, -offset)
        res = triggers[:len(clock_signal.ts)]
        return res

    def out(self, clock_signal: ClockSignal):
        # a beat is 1/4 bar. bps = bpm/60. 1/bps = seconds / beat. sampling_freq = samples / second.
        # -> samples / beat = sampling_freq / bps
        bpm = np.mean(self.bpm(clock_signal))
        samples_per_beat = SAMPLING_FREQUENCY / (bpm / 60)  # number of samples between 1/4 triggers
        if samples_per_beat < 2:
            print("Warning: Cannot deal with samples_per_beat < 2")
            samples_per_beat = 2
        #print("samples_per_beat", samples_per_beat)
        # generate all trigger tracks.
        trigger_tracks = [DrumMachine.extend_tracks(clock_signal, track, samples_per_beat) for track in self.tracks]
        # now these trigger_tracks must be given envelopes. this is an operation with time-context, and should be
        # handled by a pro - that is a module which deals with things like last_generated_signal or future_cache etc.

        #for name, tri in zip(["kick trigger", "snare trigger", "hihat trigger"], trigger_tracks):
        #    self.collect(name) << tri

        signals = []
        for track, trigger_track in zip(self.tracks, trigger_tracks):
            signal = track.trigger_modulator(clock_signal, trigger_track, track.envelope_gen)

            # modulate envelope with its carrier
            carrier = track.carrier(clock_signal)[:len(signal)]
            signal = signal * carrier

            self.collect(track.name) << signal

            signals.append(signal)
        sum_of_tracks = functools.reduce(np.add, signals)
        #plt.plot(sum_of_tracks)
        #plt.show()
        return sum_of_tracks

    @staticmethod
    def _track_to_triggers(track: Track, samples_per_beat):
        """Create a time signal of triggers. The length corresponds to the number of bars in the given track pattern"""
        _, pattern, note_value, _, _, _ = track
        samples_per_note = round(samples_per_beat * note_value * 4)  # e.g., 1/16*4 = (1/4 * samples_per_beat) per note
        indices = np.nonzero(pattern)[0] * samples_per_note
        triggers = np.zeros(len(pattern) * samples_per_note)
        #print("indices", indices)
        triggers[indices] = 1
        #plt.plot(triggers)
        #plt.show()
        return triggers



class Drummin(Module):

    def __init__(self):
        kick = Track(name="kick",
                     pattern=[1, 0, 0, 0, 1, 1, 0, 0],
                     note_values=1 / 8,
                     #envelope_gen=ADSREnvelopeGen(attack=P(10), decay=P(5), sustain=P(1), release=P(100), hold=P(2000)),
                     envelope_gen=ExpEnvelopeGen(attack_length=P(100), attack_curvature=P(3), decay_length=P(1000), decay_curvature=P(2)),
                     carrier=TriangleSource(frequency=P(60)) + NoiseSource() * 0.05,
                     trigger_modulator=TriggerModulator(),
                     )
        snare = Track(name="snare",
                      pattern=[0, 0, 1, 0, 0, 0, 1, 0],
                      note_values=1 / 8,
                      #envelope_gen=ADSREnvelopeGen(attack=P(10), decay=P(5), sustain=P(1), release=P(100), hold=P(400)),
                      envelope_gen=ExpEnvelopeGen(attack_length=P(200), attack_curvature=P(10), decay_length=P(30),
                                                  decay_curvature=P(3)) |
                                   ExpEnvelopeGen(attack_length=P(50), attack_curvature=P(5), decay_length=P(500),
                                                  decay_curvature=P(1000)),
                      carrier=TriangleSource(frequency=P(1000)) + NoiseSource() * 0.6,
                      trigger_modulator=TriggerModulator(),
                      )
        hihat = Track(name="hihat",
                      pattern=[0, 1, 0, 1, 0, 1, 0, 1],
                      note_values=1 / 8,
                      #envelope_gen=ADSREnvelopeGen(attack=P(10), decay=P(2), sustain=P(1), release=P(100), hold=P(100)),
                      envelope_gen=ExpEnvelopeGen(attack_length=P(400), attack_curvature=P(3), decay_length=P(1200),
                                                  decay_curvature=P(200)),
                      carrier=NoiseSource(),
                      trigger_modulator=TriggerModulator(),
                      )
        # notes = Track(name="notes",
        #               pattern=[1, 1, 1, 1, 1, 1, 1, 1],
        #               note_values=1 / 8,
        #               envelope_gen=RectangleEnvelopeGen(length=P(4000)),
        #               trigger_modulator=TriggerModulator(),
        #               carrier=SineSource(frequency=Random(max_amplitude=880, change_chance=0.00005)),
        #               )

        self.output = DrumMachine(bpm=Parameter(120, key='q'), tracks=[kick, snare])

        #self.synthie = StepSequencing()

        self.out = self.output #+ self.synthie


if __name__ == "__main__":
    plot_module(Drummin, plot=(".*",), num_steps=50)

