from modules import ClockSignal, Clock, Module, Parameter, Random, SineSource, SawSource, TriangleSource
import random
import numpy as np
from typing import Dict, List, NamedTuple, Callable

P = Parameter


class ADSRShapeGen:
    """Borrowed from modules.py"""

    def __init__(self, attack, decay, sustain, release, hold):
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.hold = hold

    def __call__(self):
        t_attack = round(self.attack.get())
        t_decay = round(self.decay.get())
        sustain_height = self.sustain.get()
        t_hold = round(self.hold.get())
        t_release = round(self.release.get())

        attack = np.linspace(0, 1, t_attack)
        decay = np.linspace(1, sustain_height, t_decay)
        hold = np.ones(t_hold) * sustain_height
        release = np.linspace(sustain_height, 0, t_release)
        return np.concatenate((attack, decay, hold, release), 0)


class Track(NamedTuple):
    name: str
    pattern: List[int]
    note_values: float
    shape_gen: Callable  # regular function or Module


class DrumMachine(Module):
    """
    Parametrize with trigger patterns for different tracks (kick, snare, hihat, ...).
    Trigger patterns go over any number of bars and repeat forever.
    The trigger patterns live in bar-space, not time:
    The trigger patterns will be spaced out in time according to the DrumMachine's bpm.
    A trigger has no length! The length of a shape is the shape generator's concern.
    Ways to write down a trigger pattern:
        Direct:     [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0], (1 bar) or (1/16) or (2 bars) etc
                    Needs additional information: Either how many bars this is, or the note length.
        Inverse:    Each entry is the inverse of the length of the pause after the trigger.
                    [4,4,4,4] -> [1,1,1,1], (1 bar)
                    [2,4,4] -> [1,0,1,1]
                    [2,2] -> [1,0,1,0]
                    Downside: How to do offbeats, or other patterns that don't start at 1?
        ...?
    Every track needs a shape generator
    """

    def __init__(self, bpm: Module, tracks: List[Track]):
        self.bpm = bpm
        self.tracks = tracks
        self.dummy = SineSource(frequency=P(220, key='w'))

    def out(self, clock_signal: ClockSignal):
        bpm = np.mean(self.bpm(clock_signal))
        return self.dummy(clock_signal)


class Drummin(Module):

    def __init__(self):
        kick = Track(name="kick",
                     pattern=[1, 0, 0, 1, 1, 0, 0, 0],
                     note_values=1 / 8,
                     shape_gen=ADSRShapeGen(attack=P(200), decay=P(10), sustain=P(0.5), release=P(200), hold=P(200)),
                     )
        snare = Track(name="snare",
                      pattern=[0, 1, 0, 1],
                      note_values=1 / 4,
                      shape_gen=ADSRShapeGen(attack=P(100), decay=P(10), sustain=P(0.2), release=P(100), hold=P(50)),
                      )
        hihat = Track(name="hihat",
                      pattern=[1, 1, 1, 1, 1, 1, 1, 1],
                      note_values=1 / 8,
                      shape_gen=ADSRShapeGen(attack=P(50), decay=P(10), sustain=P(0.1), release=P(100), hold=P(30)),
                      )
        self.output = DrumMachine(bpm=Parameter(80, key='q'), tracks=[kick, snare, hihat])
        self.out = self.output




