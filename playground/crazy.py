import dataclasses
import time
from typing import Optional

import numpy as np


class LFO:

    def __init__(self, freq, amplitude):
        self.i = 4
        self.freq = freq
        self.amplitude = amplitude

    def __call__(self, x):
        return self.amplitude


@dataclasses.dataclass
class Parameter:
    name: Optional[str] = None
    x: float = None

    def set_value(self, x):
        self.x = x

    def get(self):
        return self.x


class Module:

    def __call__(self, ts: np.ndarray):
        pass




class Modulator:

    def __init__(self, freq: Module, sig: Module):
        self.sig = sig
        self.sin = SineMaker(freq)

    def __call__(self, ts: np.ndarray):
        modulor_sin = self.sin(ts)
        return self.sig(ts) * modulor_sin


class Constant:


    def __call__(self, _):
        return constant


class LowPass:

    def __init__(self, cut_off: Module, sig: Module):
        self.sig = sig
        self.cut_off = cut_off

        self.previous_sig = None

        self.low_pass_filter = [...]

    def __call__(self, ts: np.ndarray):
        sig = self.sig(ts)
        full_sig = np.concatenate((sig, self.previous_sig))
        self.previous_sig = sig
        return convolve(full_sig, low_pass_filter)


class Program:


    def find_params(self):
        parameters = {}
        for cls_name, cls_instance in vars(self).items():
            for k, v in vars(cls_instance).items():
                if isinstance(v, Parameter):
                    param_name = f"{cls_name}.{k}"
                    v.name = param_name
                    print(f"Found param: {param_name}")
                    parameters[param_name] = v
        return parameters

    def __init__(self):
        self.lfo = LFO(10, amplitude=Parameter())
        self.lfo2 = LFO(Parameter(), amplitude=Parameter())

    def __call__(self, clock):
        return self.lfo2(clock)
