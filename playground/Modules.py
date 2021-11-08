
import numpy as np
import dataclasses

import scipy

NUM_CHANNELS = 1
BLOCKSIZE = 512
SHAPE = (BLOCKSIZE, NUM_CHANNELS)

class Module:
    """
    Module :: Signal -> Signal, in particular:
    Module :: [Sampling_Times] -> [Samples]
    Modules can be called on a nparray of sampling times, and calculate an output of the same size according to
    the module graph defined in its constructor.
    A subclass should overwrite self.out, respecting its signature.
    """

    def out(self, ts: np.ndarray) -> np.ndarray:
        raise Exception("not implemented")

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        return self.out(ts)

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


class Constant(Module):
    def __init__(self, value):
        self.value = value
    def out(self, ts):
        # TODO: consider: output a scalar or a vector?
        return np.repeat(self.value, ts.shape[0]).reshape(SHAPE)


# for now,
Parameter = Constant


class SineSource(Module):
    def __init__(self, frequency: Module, amplitude=Parameter(0.9), phase=Parameter(0.0)):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def out(self, ts):
        amp = self.amplitude(ts)
        freq = self.frequency(ts)
        phase = self.phase(ts)
        out = amp * np.sin(2 * np.pi * freq * ts.reshape(SHAPE) + phase)
        out = out.reshape(SHAPE)
        return out


class SawSource(Module):
    def __init__(self, frequency: Module, amplitude=Parameter(0.9), phase=Parameter(0.0)):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def out(self, ts):
        amp = self.amplitude(ts)
        freq = self.frequency(ts)
        phase = self.phase(ts)
        # now, we need ts reshaped so that it does not broadcast into channels
        ts = ts.reshape(SHAPE)
        period = 1 / freq
        #return (((ts + phase) % period) / (period / 2) - 1) * amp
        out = 2 * (ts/period - np.floor(1/2 + ts/period)) * amp
        out = out.reshape(SHAPE)
        return out


class Modulator(Module):
    def __init__(self, inp: Module, carrier_frequency: Module):
        self.carrier = SineSource(carrier_frequency, amplitude=Parameter(1.0))
        self.inp = inp
        # self.out = MultiplierModule(self.carrier, inp) # TODO: consider multiplier module for nice composition
    def out(self, ts):
        out = self.carrier(ts) * self.inp(ts)
        return out


class SimpleLowPass(Module):
    """Simplest lowpass: average over previous <window> values"""
    def __init__(self, inp: Module, window_size):
        self.window_size = window_size # module, not int
        self.last_signal = np.zeros(SHAPE)
        self.inp = inp

    def out(self, ts):
        # every step we could have a different window size. so I am using a for loop for now. TODO: pls advise
        input = self.inp(ts)
        full_signal = np.concatenate((self.last_signal, input), axis=0)
        full_len = full_signal.shape[0]
        inp_len = input.shape[0]
        window_sizes = self.window_size(ts)
        res = []
        for i, ws in enumerate(window_sizes):
            ws = ws[0]  # TODO: will cause problems when we have multiple channels
            current_val = np.mean(full_signal[full_len-inp_len + i - ws:full_len-inp_len + i, :], axis=0)
            res.append(current_val)
        out = np.array(res).reshape(SHAPE)
        self.last_signal = input
        return out



############################################
# ======== Test composite modules ======== #

class BabiesFirstSynthie(Module):
    def __init__(self):
        self.src = SawSource(Parameter(220))
        self.modulator = Modulator(self.src, SineSource(Parameter(300)))
        self.lp = SimpleLowPass(self.modulator, window_size=Parameter(16))
        self.out = self.lp



