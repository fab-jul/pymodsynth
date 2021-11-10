from functools import reduce

import numpy as np
import dataclasses

import scipy
import scipy.signal


class Monitor:
    def __init__(self):
        self.data = np.zeros((1,1))
        self.mod_name = "No name"

    def write(self, data_in, mod_name):
        self.data = data_in
        self.mod_name = mod_name

    def get_data(self):
        return self.data


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
        out = self.out(ts)
        if hasattr(self, "monitor") and self.monitor is not None:
            self.monitor.write(out, self.__repr__())
        return out

    def attach_monitor(self, monitor: Monitor):
        self.monitor = monitor

    def detach_monitor(self):
        self.monitor = None

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
        return np.ones_like(ts) * self.value


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
        out = amp * np.sin(2 * np.pi * freq * ts + phase)
        return out


class SawSource(Module):
    def __init__(self, frequency: Module, amplitude=Parameter(0.9), phase=Parameter(0.0)):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def out(self, ts):
        amp = self.amplitude(ts)
        freq = self.frequency(ts)
        phase = self.phase(ts)  # TODO: unused!
        period = 1 / freq
        out = 2 * (ts/period - np.floor(1/2 + ts/period)) * amp
        return out


class SineModulator(Module):
    def __init__(self, inp: Module, carrier_frequency: Module, inner_amplitude=Parameter(1.0)):
        self.carrier = SineSource(carrier_frequency, amplitude=inner_amplitude)
        self.inp = inp
        # self.out = MultiplierModule(self.carrier, inp) # TODO: consider multiplier module for nice composition

    def out(self, ts):
        out = self.carrier(ts) * self.inp(ts)
        return out


class SimpleLowPass(Module):
    """Simplest lowpass: average over previous <window> values"""
    def __init__(self, inp: Module, window_size: Module):
        self.window_size = window_size
        self.last_signal = None
        self.inp = inp

    def out(self, ts):
        if self.last_signal is None:
            self.last_signal = np.zeros_like(ts)
        num_samples, num_channels = ts.shape
        input = self.inp(ts)
        # Shape: (2*num_frames, num_channels)
        full_signal = np.concatenate((self.last_signal, input), axis=0)
        window_sizes = self.window_size(ts)
        # TODO: Now we have one window size per frame. Seems reasonable?
        # Maybe we want to have a "MapsToSingleValueModule".
        window_size: int = round(float(np.mean(window_sizes)))
        mean_filter = np.ones((window_size,), dtype=ts.dtype)
        result_per_channel = []
        start_time_index = num_samples - window_size + 1
        # Note that this for loop si over at most 2 elements!
        for channel_i in range(num_channels):
            result_per_channel.append(
                # TODO: Check out `oaconvolve`?
                scipy.signal.convolve(full_signal[start_time_index:, channel_i],
                                      mean_filter, "valid"))
        self.last_signal = input
        output = np.stack(result_per_channel, axis=-1)  # Back to correct shape
        assert output.shape == ts.shape, (output.shape, ts.shape, window_size, start_time_index)
        return output


class Lift(Module):
    """Lifts a signal from [-1,1] to [0,1]"""
    def __init__(self, inp: Module):
        self.inp = inp

    def out(self, ts):
        res = self.inp(ts)
        return res / 2 + 0.5


class ScalarMultiplier(Module):
    """Needed for inner generators, so that we can have a changing frequency that is, e.g., not just between [0,1] but
    between [0,440]"""
    # we could still pass a module as the value, instead of a constant float..
    def __init__(self, inp: Module, value: float):
        self.inp = inp
        self.value = value

    def __call__(self, ts):
        return self.inp(ts) * self.value


class Multiplier(Module): # TODO: variadic input
    def __init__(self, inp1: Module, inp2: Module):
        self.inp1 = inp1
        self.inp2 = inp2

    def out(self, ts):
        return self.inp1(ts) * self.inp2(ts)


class ClickSource(Module):
    """
    Creates a click track [...,0,1,0,...]
    One 1 per num_samples
    """
    def __init__(self, num_samples: Module):
        self.num_samples = num_samples
        self.counter = 0


    def out(self, ts: np.ndarray) -> np.ndarray:
        num_samples = int(np.mean(self.num_samples(ts))) # hack to have same blocksize per frame, like Lowpass...
        out = np.zeros_like(ts)
        print("counter", self.counter)
        print("num 1ones:", int(np.ceil((ts.shape[0]-self.counter) / num_samples)))
        for i in range(int(np.ceil((ts.shape[0]-self.counter) / num_samples))):
            out[self.counter + i * num_samples, :] = 1
        self.counter += num_samples - (ts.shape[0] % num_samples)
        self.counter = self.counter % ts.shape[0]
        print("click out", out)
        return out



class PlainMixer(Module):
    """Adds all input signals without changing their amplitudes"""
    def __init__(self, *args):
        self.out = lambda ts: reduce(np.add, [inp(ts) for inp in args]) / (len(args))


class MultiScaler(Module):
    """
    Takes n input modules and n input amplitudes and produces n amplified output modules.
    Combine with PlainMixer to create a Mixer.
    """
    pass #TODO

############################################
# ======== Test composite modules ======== #


def test_module(module: Module, num_frames=5, frame_length=512, num_channels=1, sampling_frequency=44100):
    import matplotlib.pyplot as plt
    res = []
    for i in range(num_frames):
        ts = (i*frame_length + np.arange(frame_length)) / sampling_frequency
        ts = ts[..., np.newaxis] * np.ones((1,))
        out = module(ts)
        res.append(out)
    res = np.concatenate(res)
    plt.plot(res)
    plt.vlines([i * frame_length for i in range(0, num_frames+1)], ymin=np.min(res)*1.1, ymax=np.max(res)*1.1, linewidth=0.8, colors='r')
    plt.hlines(0, -len(res)*0.1, len(res)*1.1, linewidth=0.8, colors='r')
    plt.show()

#test_module(ClickSource(num_samples=Parameter(19)))
#test_module(SimpleLowPass(SineSource(frequency=Parameter(440)), window_size=Parameter(513)))
#test_module(SawSource(frequency=Parameter(440)))


class BabiesFirstSynthie(Module):
    def __init__(self):
        self.lfo = SineSource(Parameter(1))
        self.sin0 = SineSource(frequency=Parameter(440*(2/3)*(2/3)))
        self.sin1 = SineSource(frequency=Parameter(440))
        self.sin2 = SineSource(frequency=Parameter(220))


        self.out = PlainMixer(self.sin0, self.sin1, self.sin2)


        #self.changingsine0 = Multiplier(self.src, self.lfo)
        #self.changingsine1 = SineModulator(self.src, Parameter(1))
        # above 2 should be equal
        #self.lowpass = SimpleLowPass(self.changingsine0, window_size=Parameter(2))

        #self.src = SineSource(ScalarMultiplier(Lift(SineSource(Parameter(10))), 22))
        #self.modulator = SineModulator(self.src, Parameter(10))
        #self.lp = SimpleLowPass(self.modulator, window_size=Parameter(16))
        #self.out = self.lowpass




