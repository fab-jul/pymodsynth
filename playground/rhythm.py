import dataclasses
import functools

from modules import ClockSignal, Clock, Module, Parameter, Random, SineSource, SawSource, TriangleSource, SAMPLING_FREQUENCY
import random
import numpy as np
from typing import Dict, List, NamedTuple, Callable

import matplotlib.pyplot as plt

P = Parameter


class ADSREnvelopeGen:
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
        envelopes = envelope_gen(clock_signal, trigger_indices)
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
        self.bpm = bpm
        self.tracks = tracks
        self.dummy = SineSource(frequency=P(220, key='w'))

    def out(self, clock_signal: ClockSignal):
        # a beat is 1/4 bar. bps = bpm/60. 1/bps = seconds / beat. sampling_freq = samples / second.
        # -> samples / beat = sampling_freq / bps
        bpm = np.mean(self.bpm(clock_signal))
        samples_per_beat = SAMPLING_FREQUENCY / (bpm / 60)  # number of samples between 1/4 triggers
        assert(samples_per_beat >= 2)
        print("samples_per_beat", samples_per_beat)
        # generate all trigger tracks.
        trigger_tracks = []
        for track in self.tracks:
            triggers = DrumMachine._track_to_triggers(track, samples_per_beat)
            # these have different lengths depending on the number of bars given. loop until end of frame.
            while len(triggers) < len(clock_signal.ts):
                triggers = np.tile(triggers, 2)
            # we started all triggers from time 0. apply the offset
            offset = clock_signal.sample_indices[0] % clock_signal.num_samples
            triggers = np.roll(triggers, offset)
            trigger_tracks.append(triggers[:len(clock_signal.ts)])
        # for x in trigger_tracks:
        #     plt.plot(x)
        #     plt.show()

        # now these trigger_tracks must be given envelopes. this is an operation with time-context, and should be
        # handled by a pro - that is a module which deals with things like last_generated_signal or future_cache etc.
        signals = []
        for track, trigger_track in zip(self.tracks, trigger_tracks):
            signal = track.trigger_modulator(clock_signal, trigger_track, track.envelope_gen)
            signals.append(signal)
        return functools.reduce(np.add, signals)

    @staticmethod
    def _track_to_triggers(track: Track, samples_per_beat):
        """Create a time signal of triggers. The length corresponds to the number of bars in the given track pattern"""
        _, pattern, note_value, _, _ = track
        samples_per_note = round(samples_per_beat * note_value * 4)  # e.g., 1/16*4 = (1/4 * samples_per_beat) per note
        indices = np.nonzero(pattern)[0] * samples_per_note
        triggers = np.zeros(len(pattern) * samples_per_note)
        #print("indices", indices)
        triggers[indices] = 1
        #plt.plot(triggers)
        #plt.show()
        return triggers








kick = Track(name="kick",
             pattern=[1, 0, 0, 1, 1, 0, 0, 1],
             note_values=1 / 8,
             envelope_gen=ADSREnvelopeGen(attack=P(200), decay=P(10), sustain=P(0.5), release=P(200), hold=P(200)),
             trigger_modulator=TriggerModulator(),
             )
snare = Track(name="snare",
              pattern=[0, 1, 0, 1],
              note_values=1 / 4,
              envelope_gen=ADSREnvelopeGen(attack=P(100), decay=P(10), sustain=P(0.2), release=P(100), hold=P(50)),
              trigger_modulator=TriggerModulator(),
              )

# t1 = DrumMachine._track_to_triggers(kick, 4)
# t2 = DrumMachine._track_to_triggers(snare, 4)
# print(t1)
# print(t2)


class Drummin(Module):

    def __init__(self):
        kick = Track(name="kick",
                     pattern=[1, 0, 0, 1, 1, 0, 0, 0],
                     note_values=1 / 8,
                     envelope_gen=ADSREnvelopeGen(attack=P(200), decay=P(10), sustain=P(0.5), release=P(200), hold=P(200)),
                     trigger_modulator=TriggerModulator(),
                     )
        snare = Track(name="snare",
                      pattern=[0, 1, 0, 1],
                      note_values=1 / 4,
                      envelope_gen=ADSREnvelopeGen(attack=P(100), decay=P(10), sustain=P(0.2), release=P(100), hold=P(50)),
                      trigger_modulator=TriggerModulator(),
                      )
        hihat = Track(name="hihat",
                      pattern=[1, 1, 1, 1, 1, 1, 1, 1],
                      note_values=1 / 8,
                      envelope_gen=ADSREnvelopeGen(attack=P(50), decay=P(10), sustain=P(0.1), release=P(100), hold=P(30)),
                      trigger_modulator=TriggerModulator(),
                      )
        self.output = DrumMachine(bpm=Parameter(100, key='q'), tracks=[kick, snare, hihat])
        self.out = self.output




