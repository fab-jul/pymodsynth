import operator
from functools import reduce, lru_cache
from typing import Mapping, Union, MutableMapping, Optional, NamedTuple

import numpy as np
from playground import midi_lib
from playground.tests import helper as tests_helper
import time
import random

import scipy
import scipy.signal

# some modules need access to sampling_frequency
SAMPLING_FREQUENCY = 44100


OUT_DTYPE = np.float32  # TODO


class ClockSignal(NamedTuple):
    ts: np.ndarray
    sample_indices: np.ndarray
    sample_rate: int

    def zeros(self):
        return np.zeros_like(self.ts)

    @property
    def shape(self):
        return self.ts.shape

    @property
    def num_samples(self):
        return self.ts.shape[0]

    def add_channel_dim(self, signal):
        assert len(signal.shape) == 1
        return np.broadcast_to(signal.reshape(-1, 1), self.shape)


class Clock:

    def __init__(self, num_samples=2048, num_channels=2, sample_rate=44100.):
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.i = 0

        self.arange = np.arange(self.num_samples, dtype=int)  # Cache it.

    def __call__(self) -> ClockSignal:
        sample_indices = self.i + self.arange
        ts = sample_indices / self.sample_rate
        # Broadcast `ts` into (num_samples, num_channels)
        ts = ts[..., np.newaxis] * np.ones((self.num_channels,))
        self.i += self.num_samples
        return ClockSignal(ts, sample_indices, self.sample_rate)


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


class Module:
    """
    Module :: Signal -> Signal, in particular:
    Module :: [Sampling_Times] -> [Samples]
    Modules can be called on a nparray of sampling times, and calculate an output of the same size according to
    the module graph defined in its constructor.
    A subclass should overwrite self.out, respecting its signature.
    """
    measure_time = False

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        raise Exception("not implemented")

    def __mul__(self, other):
        """Implement module * scalar and module * module."""
        return _MathModule(operator.mul, self, other)

    def __rmul__(self, other):
        """Implement scalar * module."""
        return self * other

    def __truediv__(self, other):
        """Implement module / scalar and module / module"""
        return _MathModule(operator.truediv, self, other)

    def __rtruediv__(self, other):
        return _MathModule(operator.truediv, other, self)

    def __add__(self, other):
        """Implement module + scalar and module + module"""
        return _MathModule(operator.add, self, other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        """Implement module - scalar and module - module"""
        return _MathModule(operator.sub, self, other)

    def __rsub__(self, other):
        return _MathModule(operator.sub, other, self)

    # TODO: Describe why we have call and out (in short: for convenicence)
    def __call__(self, clock_signal: ClockSignal) -> np.ndarray:
        return self.out(clock_signal)

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
            if isinstance(var_instance, Module):
                result.update(var_instance._find(cls, prefix=f"{var_name}."))
        return result

    # NOTE: We need to take the params and state, as we cannot
    # find it anymore, since we have new classes when we call this!
    def copy_params_and_state_from(self, src_params, src_state):
        _copy(src=src_params, target=self.find_params())
        _copy(src=src_state, target=self.find_state())


class _MathModule(Module):
    """Implement various mathematical operations on modules, see base class."""

    def __init__(self, op, left, right):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

    def out(self, clock_signal: ClockSignal):
        left = self._maybe_call(self.left, clock_signal)
        right = self._maybe_call(self.right, clock_signal)
        return self.op(left, right)

    @staticmethod
    def _maybe_call(module_or_number, clock_signal: ClockSignal):
        if isinstance(module_or_number, Module):
            return module_or_number(clock_signal)
        return module_or_number


@tests_helper.mark_for_testing(value=lambda: 1)
class Constant(Module):
    def __init__(self, value):
        self.value = value

    def out(self, clock_signal: ClockSignal):
        # TODO: sounds cool
        # num_samples, num_channels = ts.shape
        # if abs(self.previous_value - self.value) > 1e-4:
        #     out = (np.linspace(self.previous_value, self.value, num_samples).reshape(-1, 1) *
        #            np.ones((num_channels,)))
        #     print(self.previous_value, self.value, out[:10])
        # else:
        #     out = np.ones_like(ts) * self.value
        out = np.broadcast_to(self.value, clock_signal.shape)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}(value={self.value})'

    def set(self, value):
        self.value = value

    def inc(self, diff):
        """Increment value by `diff`."""
        self.set(self.value + diff)

    def get(self):
        return self.value


@tests_helper.mark_for_testing(value=lambda: 1)
class Parameter(Constant):

    def __init__(self,
                 value: float,
                 lo: Optional[float] = None,
                 hi: Optional[float] = None,
                 key: Optional[str] = None,
                 knob: Optional[midi_lib.KnobConvertible] = None,
                 shift_multiplier: float = 10,
                 clip: bool = False):
        """Create Parameter.

        NOTES:
        - `lo`, `hi` are always used for knobs, but only used for `key` if `clip=True`.
          This is because knobs have a range of 0-127, and we use `lo`, `hi` to map to that range.

        Args:
            value: Initial value
            lo: Lowest sane value. Defaults to 0.1 * value.
            hi: Highest sane value. Defaults to 1.9 * value.
            key: If given, a key on the keyboard that controls this parameter. Example: "f".
            knob: If given, a knob on a Midi controller that controls this parameter.
            shift_multiplier: Only used if `key` is set, in which case this sets how much
              more we change the parameter if SHIFT is pressed on the keyboard.
            clip: If True, clip to [lo, hi] in `set`.
        """
        super().__init__(value)
        if lo is None:
            lo = 0.1 * value
        if hi is None:
            hi = 1.9 * value
        if hi < lo:
            raise ValueError
        self.lo, self.hi = lo, hi
        self.span = self.hi - self.lo
        self.key = key
        self.knob = knob
        self.shift_multiplier = shift_multiplier
        self.clip = clip

    def set(self, value):
        if self.clip:
            self.value = np.clip(value, self.lo, self.hi)
        else:
            self.value = value

    def set_relative(self, rel_value: float):
        """Set with value in [0, 1], and we map to [lo, hi]."""
        self.set(self.lo + self.span * rel_value)


@tests_helper.mark_for_testing()
class Random(Module):
    """Output a constant random amplitude until a random change event changes the amplitude. Best explanation ever."""
    def __init__(self, max_amplitude: float = 1.0, change_chance: float = 0.5):
        self.max_amplitude = max_amplitude
        self.p = change_chance
        self.amp = random.random() * self.max_amplitude

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        # random experiment
        frame_length, num_channels = clock_signal.shape
        res = np.empty((0, num_channels))
        while len(res) < frame_length:
            chance_in_frame = 1 - pow(1 - self.p, frame_length - len(res))
            # to make it independent of frame length, this goes on for a random amount of time
            block_len = random.randint(1, frame_length - len(res))
            res = np.concatenate((res, np.ones((block_len, num_channels)) * self.amp))
            if random.random() < chance_in_frame:
                self.amp = random.random() * self.max_amplitude  # could discretize amps to guarantee pleasant ratios
        return res


@tests_helper.mark_for_testing()
class SineSource(Module):
    def __init__(self,
                 frequency: Module = Constant(440.),
                 amplitude=Parameter(1.0),
                 phase=Parameter(0.0)):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.last_cumsum_value = State(initial_value=0)

    def out(self, clock_signal: ClockSignal):
        dt = np.mean(clock_signal.ts[1:] - clock_signal.ts[:-1])
        amp = self.amplitude(clock_signal)
        freq = self.frequency(clock_signal)
        phase = self.phase(clock_signal)
        cumsum = np.cumsum(freq * dt, axis=0) + self.last_cumsum_value.get()
        # Cache last cumsum for next step.
        self.last_cumsum_value.set(cumsum[-1, :])
        out = amp * np.sin((2 * np.pi * cumsum) + phase)
        return out


@tests_helper.mark_for_testing()
class SawSource(Module):
    def __init__(self, frequency: Module = Constant(440.), amplitude=Parameter(1.0), phase=Parameter(0.0)):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def out(self, clock_signal: ClockSignal):
        amp = self.amplitude(clock_signal)
        freq = self.frequency(clock_signal)
        phase = self.phase(clock_signal)
        period = 1 / freq
        # TODO: Use cumsum
        ts = clock_signal.ts
        out = 2 * (ts/period + phase - np.floor(1/2 + ts/period + phase)) * amp
        return out


@tests_helper.mark_for_testing()
class TriangleSource(Module):
    def __init__(self, frequency: Module = Constant(440.), amplitude=Parameter(1.0), phase=Parameter(0.0)):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        amp = self.amplitude(clock_signal)
        freq = self.frequency(clock_signal)
        phase = self.phase(clock_signal)
        period = 1 / freq
        ts = clock_signal.ts
        out = (2 * np.abs(2 * (ts/period + phase - np.floor(1/2 + ts/period + phase))) - 1) * amp
        return out


class NoiseSource(Module):
    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        return np.random.random(clock_signal.ts.shape)


@tests_helper.mark_for_testing(inp=SineSource, carrier_frequency=SineSource)
class SineModulator(Module):
    def __init__(self, inp: Module, carrier_frequency: Module, inner_amplitude=Parameter(1.0)):
        self.carrier = SineSource(carrier_frequency, amplitude=inner_amplitude)
        self.inp = inp
        # self.out = MultiplierModule(self.carrier, inp) # TODO: consider multiplier module for nice composition

    def out(self, clock_signal: ClockSignal):
        out = self.carrier(clock_signal) * self.inp(clock_signal)
        return out


@tests_helper.mark_for_testing(inp=SineSource)
class SimpleLowPass(Module):
    """Simplest lowpass: average over previous <window> values"""
    def __init__(self, inp: Module, window_size: Module = Constant(3.)):
        self.window_size = window_size
        self.last_signal = State()
        self.inp = inp

    def out(self, clock_signal: ClockSignal):
        if not self.last_signal.is_set:
            self.last_signal.set(clock_signal.zeros())
        num_samples, num_channels = clock_signal.shape
        inp = self.inp(clock_signal)
        # Shape: (2*num_frames, num_channels)
        full_signal = np.concatenate((self.last_signal.get(), inp), axis=0)
        window_sizes = self.window_size(clock_signal)
        # TODO: Now we have one window size per frame. Seems reasonable?
        # Maybe we want to have a "MapsToSingleValueModule".
        window_size: int = max(1, round(float(np.mean(window_sizes))))
        mean_filter = np.ones((window_size,), dtype=OUT_DTYPE) / window_size
        result_per_channel = []
        start_time_index = num_samples - window_size + 1
        # Note that this for loop si over at most 2 elements!
        for channel_i in range(num_channels):
            result_per_channel.append(
                # TODO: Check out `oaconvolve`?
                scipy.signal.convolve(full_signal[start_time_index:, channel_i],
                                      mean_filter, "valid"))
        self.last_signal.set(inp)
        output = np.stack(result_per_channel, axis=-1)  # Back to correct shape
        assert output.shape == clock_signal.shape, (output.shape, clock_signal.shape, window_size, start_time_index)
        return output


# TODO: Add test, currently does not pass
# @tests_helper.mark_for_testing(inp=SineSource, shape=SineSource)
class ShapeModulator(Module):
    """
    Modulate a given shape onto clicks in time domain. Nearby clicks will both get the shape, so they may overlap.
    """
    def __init__(self, inp: Module, shape: Module):
        self.last_signal = None
        self.inp = inp
        self.shape = shape

    def out(self, clock_signal: ClockSignal):
        if self.last_signal is None:
            self.last_signal = clock_signal.zeros()
        click_signal = self.inp(clock_signal)
        # like in SimpleLowpass, we really want a different window for every click. but for the moment,
        # we just use a single window for the whole frame
        shape = self.shape(clock_signal)
        full_click_signal = np.concatenate((self.last_signal, click_signal), axis=0)
        out = scipy.signal.convolve(full_click_signal, shape, mode="valid")
        self.last_signal = click_signal
        return out[-clock_signal.shape[0]:, :]


class ShapeExp(Module):
    """Gives a signal of length shape_length"""
    def __init__(self, shape_length: int, decay=2.0, amplitude=1.0):
        self.shape_length = shape_length
        self.decay = decay
        self.amplitude = amplitude

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        shape = np.array([self.amplitude / pow(self.decay, i) for i in range(self.shape_length)]).reshape(-1,1)
        return shape


@tests_helper.mark_for_testing()
class ClickSource(Module):
    """
    Creates a click track [...,0,1,0,...]
    One 1 per num_samples
    """
    def __init__(self, num_samples: Module = Constant(10)):
        self.num_samples = num_samples
        self.counter = 0

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        num_samples = int(np.mean(self.num_samples(clock_signal))) # hack to have same blocksize per frame, like Lowpass...
        out = clock_signal.zeros()
        #print("counter", self.counter)
        #print("num 1ones:", int(np.ceil((ts.shape[0]-self.counter) / num_samples)))
        for i in range(int(np.ceil((clock_signal.shape[0]-self.counter) / num_samples))):
            out[self.counter + i * num_samples, :] = 1
        self.counter += num_samples - (clock_signal.shape[0] % num_samples)
        self.counter = self.counter % clock_signal.shape[0]
        self.counter = self.counter % num_samples
        #print("click out", out)
        return out
        # TODO: buggy for num_samples greater than frame_length!




############################################
# ======== Test composite modules ======== #


# TODO: Import collecting mechanism
def test_module(module: Module, num_frames=5, frame_length=2048, num_channels=1, sampling_frequency=44100, show=True):
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
    if show:
        plt.show()


def lift(a):
    """Lifts a signal from [-1,1] to [0,1]"""
    return a / 2 + 0.5


# TODO: Currently does not test atm.
# @tests_helper.mark_for_testing()
class ClickModulation(Module):
    def __init__(self):
        self.wild_triangles = sum(TriangleSource(frequency=Random(220 * i, 0.000015)) for i in range(1, 3))
        self.out = KernelConvolver(self.wild_triangles, KernelGenerator(lambda x: 1, length=Parameter(100)))
        # test_module(self.out, num_frames=10)


@tests_helper.mark_for_testing()
class BabiesFirstSynthie(Module):
    def __init__(self):
        self.lfo = SineSource(Parameter(1))
        self.sin0 = SineSource(frequency=Parameter(440*(2/3)*(2/3)))
        self.sin1 = SineSource(frequency=Parameter(440))
        self.sin2 = SineSource(frequency=Parameter(220))

        #self.out = PlainMixer(self.sin0, self.sin1, self.sin2)

        self.changingsine0 = self.sin0 * self.lfo
        self.changingsine1 = SineModulator(self.sin0, Parameter(1))
        self.lowpass = SimpleLowPass(self.changingsine0, window_size=Parameter(2))

        self.src = SineSource(lift(SineSource(Parameter(10))) * 22)
        self.modulator = SineModulator(self.src, Parameter(10))
        self.lp = SimpleLowPass(self.modulator, window_size=Parameter(16))
        self.out = self.lowpass


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
