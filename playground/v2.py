import dataclasses
import time

import numpy as np

Signal = np.ndarray


class SampleAndHold:

    def __init__(self):
        pass

    def __call__(self,
                 signal: Signal,
                 num_steps: int):
        pass


class SignalGenerator:

    def __init__(self):
        pass

    def __call__(self,
                 ts: Signal,
                 signal: str = "sin",
                 amplitude: float = 0.2,
                 freq: float = 224):
        return amplitude * np.sin(2 * np.pi * freq * ts)


def __call__(self, signal, carrier=0):
    pass


def program(clock, parameters):
    lfo = SinusGenerator(freq=1)
    sin = SinusGenerator(freq=lfo, amplitude=param
    return sin(clock)


def sample_generor()
    while 1:
        parameters = update_params(...)
        o = program(clock, parameters)
        ...


