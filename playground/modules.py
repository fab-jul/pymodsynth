from functools import reduce, lru_cache
from typing import Mapping, Union, MutableMapping

import numpy as np
import time

import scipy
import scipy.signal

# some modules need access to sampling_frequency
SAMPLING_FREQUENCY = 44100


class State:

    def __init__(self, *, initial_value=None):
        self._value = initial_value

    @property
    def is_set(self) -> bool:
        return self._value is not None

    def get(self):
        return self._value

    def set(self, value):
        self._value = value




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
    measure_time = False

    def out(self, ts: np.ndarray) -> np.ndarray:
        raise Exception("not implemented")


    def __call__(self, ts: np.ndarray) -> np.ndarray:
        if Module.measure_time:
            t0 = time.time()

        out = self.out(ts)

        if Module.measure_time:
            t1 = time.time()
            print("Time to call out(ts) for Module", self.__repr__(), ":", t1 - t0)

        if hasattr(self, "monitor") and self.monitor is not None:
            self.monitor.write(out, self.__repr__())
        return out

    def attach_monitor(self, monitor: Monitor):
        self.monitor = monitor

    def detach_monitor(self):
        self.monitor = None

    def find_params(self) -> MutableMapping[str, "Parameter"]:
        return self._find(Parameter)

    def find_state(self) -> MutableMapping[str, "State"]:
        return self._find(State)

    def _find(self, cls, prefix=""):
        result = {}
        for var_name, var_instance in vars(self).items():
            if isinstance(var_instance, cls):  # Top-level.
                var_instance.name = f"{prefix}{var_name}"
                result[var_name] = var_instance
                continue
            # TODO: We are not recursive, because we only
            #  want top level stuff, but maybe we should reconsider.
#            if isinstance(var_instance, Module):
#
#                # Recursion!
#                result.update(var_instance._find(cls, prefix=f"{var_name}."))
            for k, v in vars(var_instance).items():
                if isinstance(v, cls):
                    param_name = f"{var_name}.{k}"
                    v.name = param_name
                    result[param_name] = v
        return result

    # NOTE: We need to take the params and state, as we cannot
    # find it anymore, since we have new classes when we call this!
    def copy_params_and_state_from(self, src_params, src_state):
        _copy(src=src_params, target=self.find_params())
        _copy(src=src_state, target=self.find_state())


class Constant(Module):
    def __init__(self, value):
        self.value = value
        self.previous_value = value

    def out(self, ts):
        num_samples, num_channels = ts.shape
        # TODO: is cool
#        if abs(self.previous_value - self.value) > 1e-4:
#            out = (np.linspace(self.previous_value, self.value, num_samples).reshape(-1, 1) *
#                   np.ones((num_channels,)))
#            print(self.previous_value, self.value, out[:10])
#        else:
#            out = np.ones_like(ts) * self.value
        out = np.ones_like(ts) * self.value
        self.previous_value = self.value
        return out



    def __repr__(self):
        return f'Constant(value={self.value})'

    def set(self, value):
        self.value = value

    def get(self):
        return self.value

# for now,
Parameter = Constant



class SineSource(Module):
    def __init__(self, frequency: Module, amplitude=Parameter(1.0), phase=Parameter(0.0)):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.prev = None

    def out(self, ts):
        amp = self.amplitude(ts)
        freq = self.frequency(ts)
        phase = self.phase(ts)
        if self.prev is not None:
            last_value = self.prev[-1]
            shift = np.arcsin(last_value) / (2 * np.pi * freq[0])
        else:
            shift = 0
        out = amp * np.sin(2 * np.pi * freq * (ts - shift) + phase)
        self.prev = out
        return out


class SawSource(Module):
    def __init__(self, frequency: Module, amplitude=Parameter(1.0), phase=Parameter(0.0)):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def out(self, ts):
        amp = self.amplitude(ts)
        freq = self.frequency(ts)
        phase = self.phase(ts)
        period = 1 / freq
        out = 2 * (ts/period + phase - np.floor(1/2 + ts/period + phase)) * amp
        return out


class TriangleSource(Module):
    def __init__(self, frequency: Module, amplitude=Parameter(1.0), phase=Parameter(0.0)):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def out(self, ts: np.ndarray) -> np.ndarray:
        amp = self.amplitude(ts)
        freq = self.frequency(ts)
        phase = self.phase(ts)
        period = 1 / freq
        out = (2 * np.abs(2 * (ts/period + phase - np.floor(1/2 + ts/period + phase))) - 1) * amp
        return out


class SineModulator(Module):
    def __init__(self, inp: Module, carrier_frequency: Module, inner_amplitude=Parameter(1.0)):
        self.carrier = SineSource(carrier_frequency, amplitude=inner_amplitude)
        self.inp = inp
        # self.out = MultiplierModule(self.carrier, inp) # TODO: consider multiplier module for nice composition

    def out(self, ts):
        out = self.carrier(ts) * self.inp(ts)
        return out


class KernelGenerator(Module):
    """
    Takes a value generating function func and a length parameter.
    Produces a kernel of length length for every ts, whose values are generated by func.
    TODO: consider: could make func dependent on ts, or even have a func-generating Module
    Returns a shape (frame_length, max_kernel_size, num_channels).
    """
    def __init__(self, inp: Module, func, length: Module):
        self.inp = inp  # ts
        self.func = func  # function from t to [-1, 1]
        self.length = length  # kernel length, a module

    def out(self, ts: np.ndarray) -> np.ndarray:
        # we dont need the inp
        lengths = self.length(ts)  # shape (512, 1)
        print("lengths", lengths)
        max_len = int(np.max(lengths))
        frame_length, num_channels = lengths.shape
        out = np.array([[self.func(i) if i < int(l) else 0 for i in range(max_len)] for l in lengths])
        return out.reshape((frame_length, max_len, num_channels))


class LowPass(Module):
    """
    Takes a kernel-generator to smooth the input.
    TODO: some python loops seem unavoidable, but maybe there are numpy formulations to save effort.
    """
    def __init__(self, inp: Module, kernel_generator: Module):
        self.inp = inp
        self.kernel_generator = kernel_generator
        self.last_signal = None

    def out(self, ts: np.ndarray) -> np.ndarray:
        if self.last_signal is None:
            self.last_signal = np.zeros_like(ts)
        num_samples, num_channels = ts.shape
        inp = self.inp(ts)
        full_signal = np.concatenate((self.last_signal, input), axis=0)
        kernels = self.kernel_generator(ts)
        # shape must be (frame_length, max_kernel_size, num_channels)
        frame_length, max_kernel_size, num_channels = kernels.shape
        slices = np.array([full_signal[i:i+max_kernel_size] for i in range(frame_length - max_kernel_size, frame_length * 2)])
        print("kernels.shape", kernels.shape)
        print("slices.shape", slices.shape)
        # WIP



class SimpleLowPass(Module):
    """Simplest lowpass: average over previous <window> values"""
    def __init__(self, inp: Module, window_size: Module):
        self.window_size = window_size
        self.last_signal = State()
        self.inp = inp

    def out(self, ts):
        if not self.last_signal.is_set:
            self.last_signal.set(np.zeros_like(ts))
        num_samples, num_channels = ts.shape
        input = self.inp(ts)
        # Shape: (2*num_frames, num_channels)
        full_signal = np.concatenate((self.last_signal.get(), input), axis=0)
        window_sizes = self.window_size(ts)
        # TODO: Now we have one window size per frame. Seems reasonable?
        # Maybe we want to have a "MapsToSingleValueModule".
        window_size: int = max(1, round(float(np.mean(window_sizes))))
        mean_filter = np.ones((window_size,), dtype=ts.dtype) / window_size
        result_per_channel = []
        start_time_index = num_samples - window_size + 1
        # Note that this for loop si over at most 2 elements!
        for channel_i in range(num_channels):
            result_per_channel.append(
                # TODO: Check out `oaconvolve`?
                scipy.signal.convolve(full_signal[start_time_index:, channel_i],
                                      mean_filter, "valid"))
        self.last_signal.set(input)
        output = np.stack(result_per_channel, axis=-1)  # Back to correct shape
        assert output.shape == ts.shape, (output.shape, ts.shape, window_size, start_time_index)
        return output


class ShapeModulator(Module):
    """
    Modulate a given shape onto clicks in time domain. Nearby clicks will both get the shape, so they may overlap.
    """
    def __init__(self, inp: Module, shape: Module):
        self.last_signal = None
        self.inp = inp
        self.shape = shape

    def out(self, ts):
        if self.last_signal is None:
            self.last_signal = np.zeros_like(ts)
        click_signal = self.inp(ts)
        # like in SimpleLowpass, we really want a different window for every click. but for the moment,
        # we just use a single window for the whole frame
        shape = self.shape(ts)
        full_click_signal = np.concatenate((self.last_signal, click_signal), axis=0)
        out = scipy.signal.convolve(full_click_signal, shape, mode="valid")
        self.last_signal = click_signal
        return out[-ts.shape[0]:, :]


class ShapeExp(Module):
    """Gives a signal of length shape_length"""
    def __init__(self, shape_length: int, decay=2.0, amplitude=1.0):
        self.shape_length = shape_length
        self.decay = decay
        self.amplitude = amplitude

    def out(self, ts: np.ndarray) -> np.ndarray:
        shape = np.array([self.amplitude / pow(self.decay, i) for i in range(self.shape_length)]).reshape(-1,1)
        return shape


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


class Multiplier(Module):  # TODO: variadic input
    def __init__(self, inp1: Module, inp2: Module):
        self.inp1 = inp1
        self.inp2 = inp2

    def out(self, ts):
        return self.inp1(ts) * self.inp2(ts)


class PlainMixer(Module):
    """Adds all input signals without changing their amplitudes"""
    def __init__(self, *args):
        self.out = lambda ts: reduce(np.add, [inp(ts) for inp in args]) # / (len(args))


class MultiScaler(Module):
    """
    Takes n input modules and n input amplitudes and produces n amplified output modules.
    Combine with PlainMixer to create a Mixer.
    """
    pass #TODO


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
        #print("counter", self.counter)
        #print("num 1ones:", int(np.ceil((ts.shape[0]-self.counter) / num_samples)))
        for i in range(int(np.ceil((ts.shape[0]-self.counter) / num_samples))):
            out[self.counter + i * num_samples, :] = 1
        self.counter += num_samples - (ts.shape[0] % num_samples)
        self.counter = self.counter % ts.shape[0]
        self.counter = self.counter % num_samples
        #print("click out", out)
        return out
        # TODO: buggy for num_samples greater than frame_length!




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
    #plt.vlines([i * frame_length for i in range(0, num_frames+1)], ymin=np.min(res)*1.1, ymax=np.max(res)*1.1, linewidth=0.8, colors='r')
    plt.hlines(0, -len(res)*0.1, len(res)*1.1, linewidth=0.8, colors='r')
    plt.show()


def kernel_test():
    ts = np.arange(512) / 44100
    ts = ts[..., np.newaxis] * np.ones((1,))
    length = PlainMixer(Parameter(1), ScalarMultiplier(Lift(SawSource(frequency=Parameter(10000))), 5))
    k = KernelGenerator(Parameter(1), lambda x: x*x, length=length)
    print("k", k.out(ts[:10, :]))
    print(k.out(ts[:10, :]).shape)
    print("---")
    lp = LowPass(src, kernel_generator=k)

#kernel_test()


#test_module(ZigSource(Parameter(100)))


#test_module(Lift(SawSource(Parameter(100))), num_frames=100)
#test_module(SineSource(Lift(SawSource(Parameter(100)))), num_frames=100)

#test_module(ClickSource(Parameter(100)))
#test_module(SimpleLowPass(SineSource(frequency=Parameter(440)), window_size=Parameter(513)))
#test_module(SawSource(frequency=Parameter(440)))

#test_module(ShapeModulator(ClickSource(Parameter(500)), ShapeExp(100, decay=1.1)))

#test_module(ScalarMultiplier(Lift(SawSource(frequency=Parameter(1000))), 10))

class ClickModulation(Module):
    def __init__(self):
        #self.out = SineModulator(ShapeModulator(ClickSource(Parameter(400)), ShapeExp(200, decay=1.01)), carrier_frequency=#Parameter(220))
        self.sin = SineSource(Parameter(440))
        self.out = self.sin
        #self.out = TriangleSource(Parameter(220))

class TestModule(Module):
    def __init__(self):
        self.lfo = SineSource(Parameter(1))
        self.sin0 = SineSource(frequency=Parameter(440*(2/3)*(2/3)))
        self.sin1 = SineSource(frequency=Parameter(440))
        self.sin2 = SineSource(frequency=Parameter(220))

        self.changingsine0 = Multiplier(self.sin0, self.lfo)
        self.changingsine1 = SineModulator(self.sin0, Parameter(1))
        self.lowpass = SimpleLowPass(self.changingsine0, window_size=Parameter(2))

        self.src = SineSource(ScalarMultiplier(Lift(SineSource(Parameter(10))), 22))
        self.modulator = SineModulator(self.src, Parameter(10))
        self.lp = SimpleLowPass(self.modulator, window_size=Parameter(16))
        self.out = self.lowpass


class StepSequencing(Module):
    def __init__(self):
        self.sin0 = SineSource(frequency=Parameter(440))
        #self.lowpass = SimpleLowPass(self.sin0, window_size=Parameter(2))
        self.out = self.sin0


class TestModule(Module):
    def __init__(self):
        for i in range(100):
            setattr(self, f"freq{i}", Parameter(i))
            setattr(self, f"sin{i}", SawSource(frequency=getattr(self, f"freq{i}")))
        self.lp = SimpleLowPass(self.sin0, window_size=Parameter(2))


# TODO: Make type, it needs set() and get()
ParamOrState = Union[Parameter, State]


def _copy(src: Mapping[str, ParamOrState],
          target: MutableMapping[str, ParamOrState]):
    for k, param in src.items():
        try:
            target[k].set(param.get())
        except KeyError as e:  # Some param disappeared -> ignore.
            print(f"Caught: {e}")
        else:
            target.pop(k)
    if target:  # Some params were not set -> ignore.
        pass


