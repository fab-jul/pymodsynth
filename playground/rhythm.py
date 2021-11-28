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
            attack = np.linspace(0, 1, t_attack[i,:])
            decay = np.linspace(1, sustain_height, t_decay[i,:])
            hold = np.ones(t_hold[i,:]) * sustain_height[i,:]
            release = np.linspace(sustain_height[i,:], 0, t_release[i,:])
            envelope = np.concatenate((attack, decay, hold, release), 0)
            res.append(envelope)
        return res


class Track(NamedTuple):
    name: str
    pattern: List[int]
    note_values: float
    envelope_gen: Callable  # regular function or Module


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

    def _get_old_signal(self):

        return None

    def out(self, clock_signal: ClockSignal):
        if old := self._get_old_signal():
            return old

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


        return None

    @staticmethod
    def _track_to_triggers(track: Track, samples_per_beat):
        """Create a time signal of triggers. The length corresponds to the number of bars in the given track pattern"""
        _, pattern, note_value, _ = track
        samples_per_note = round(samples_per_beat * note_value * 4)  # e.g., 1/16*4 = (1/4 * samples_per_beat) per note
        indices = np.nonzero(pattern)[0] * samples_per_note
        triggers = np.zeros(len(pattern) * samples_per_note)
        #print("indices", indices)
        triggers[indices] = 1
        #plt.plot(triggers)
        #plt.show()
        return triggers


class TriggerModulator:
    """
        Put an envelope on every trigger. If result is longer than a frame, keep the rest for the next call.
        Combine overlaps with a suitable function: max, fst, snd, add, ...
        """
    def __init__(self):

    def __call__(self, clock_signal: ClockSignal, triggers, envelope_gen):
        """Generate only one envelope per trigger"""
        trigger_indices = np.nonzero(triggers)
        trigger_ts = clock_signal.ts[trigger_indices]
        trigger_clock_indices = clock_signal.sample_indices[trigger_indices]
        # the envelope_gen should depend on trigger_ts and trigger_clock_indices only
        envelopes = []
        for ts, clk_ind in zip (trigger_ts, trigger_clock_indices):
            envelopes.append(envelope_gen(ts, clk_ind))


kick = Track(name="kick",
             pattern=[1, 0, 0, 1, 1, 0, 0, 1],
             note_values=1 / 8,
             envelope_gen=ADSREnvelopeGen(attack=P(200), decay=P(10), sustain=P(0.5), release=P(200), hold=P(200)),
             )
snare = Track(name="snare",
              pattern=[0, 1, 0, 1],
              note_values=1 / 4,
              envelope_gen=ADSREnvelopeGen(attack=P(100), decay=P(10), sustain=P(0.2), release=P(100), hold=P(50)),
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
                     )
        snare = Track(name="snare",
                      pattern=[0, 1, 0, 1],
                      note_values=1 / 4,
                      envelope_gen=ADSREnvelopeGen(attack=P(100), decay=P(10), sustain=P(0.2), release=P(100), hold=P(50)),
                      )
        hihat = Track(name="hihat",
                      pattern=[1, 1, 1, 1, 1, 1, 1, 1],
                      note_values=1 / 8,
                      envelope_gen=ADSREnvelopeGen(attack=P(50), decay=P(10), sustain=P(0.1), release=P(100), hold=P(30)),
                      )
        self.output = DrumMachine(bpm=Parameter(44000, key='q'), tracks=[kick, snare, hihat])
        self.out = self.output




