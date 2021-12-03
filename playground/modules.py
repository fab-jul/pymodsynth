import collections
import functools
import operator
import random
import pprint
import re
import warnings

import plot_lib

import matplotlib.pyplot as plt

import collector_lib
import dataclasses
import enum
import itertools
from functools import reduce, lru_cache
from typing import Mapping, Union, MutableMapping, Optional, TypeVar, Callable, Sequence, Tuple, Any, NamedTuple

import numpy as np
import typing

import midi_lib
import time

import scipy
import scipy.signal

# some modules need access to sampling_frequency
SAMPLING_FREQUENCY = 44100


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

    def __init__(self, num_samples, num_channels, sample_rate):
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


class Monitor:
    def __init__(self):
        self.data = np.zeros((1,1))
        self.mod_name = "No name"

    def write(self, data_in, mod_name):
        self.data = data_in
        self.mod_name = mod_name

    def get_data(self):
        return self.data


T = TypeVar("T")


class Module:
    """
    Module :: Signal -> Signal, in particular:
    Module :: [Sampling_Times] -> [Samples]
    Modules can be called on a nparray of sampling times, and calculate an output of the same size according to
    the module graph defined in its constructor.
    A subclass should overwrite self.out, respecting its signature.
    """
    measure_time = False

    def __mul__(self, other):
        return _MathModule(operator.mul, self, other)

    def __add__(self, other):
        return _MathModule(operator.add, self, other)

    def __truediv__(self, other):
        return _MathModule(operator.truediv, self, other)

    def __init__(self):
        self.collect = collector_lib.FakeCollector()

    def collect_data(self, num_steps, clock: Clock,
                     warmup_num_steps=100) -> Tuple[np.ndarray, Sequence[Tuple[str, Sequence[Any]]]]:
        # Warmup.
        for _ in range(warmup_num_steps):
            clock_signal = clock()
            self(clock_signal)

        # Set collect to a useful instance.
        collectors = self._set("collect", factory=collector_lib.Collector)

        # Loop with `ts`
        all_ts = []
        for _ in range(num_steps):
            clock_signal = clock()
            self(clock_signal)
            all_ts.append(clock_signal.ts)

        # Reset back to fake collector
        self._set("collect", factory=collector_lib.FakeCollector)

        non_empty_collectors = {k: collector for k, collector in collectors.items() if collector}
        if not non_empty_collectors:
            raise ValueError("No module collected data!")

        output = []
        for k, collector in non_empty_collectors.items():
            for shift_collector_name, shift_collector_values in collector:
                full_k = (k + "." + shift_collector_name).strip(".")
                output.append((full_k, shift_collector_values))
        return np.concatenate(all_ts, axis=0), output

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        raise Exception("not implemented")

    def __call__(self, clock_signal: ClockSignal) -> np.ndarray:
        if Module.measure_time:
            t0 = time.time()

        out = self.out(clock_signal)
        if out.shape != clock_signal.ts.shape and "TriggerSource" not in str(type(self)): # hack without isinstance because TriggerSource cannot be imported atm.
            print(type(self))
            raise ValueError(f"Failure at {self.__class__.__name__}")

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

    def _find(self, cls, prefix="", include_root=False):
        result = {}
        if prefix == "" and include_root and isinstance(self, cls):
            result[""] = self
        for var_name, var_instance in vars(self).items():
            if isinstance(var_instance, cls):  # Top-level.
                var_instance.name = f"{prefix}{var_name}"
                result[var_name] = var_instance
                continue
            if isinstance(var_instance, Module):
                result.update(var_instance._find(cls, prefix=f"{var_name}."))
        return result

    def _set(self, var_name: str, factory: Callable[[], T]) -> Mapping[str, T]:
        """Recursively set `var_name` to `factory()` for all submodules."""
        modules = self._find(Module, include_root=True)
        outputs = {}
        for k, m in modules.items():
            # print(f"Setting: {k}.{var_name}")
            value = factory()
            setattr(m, var_name, value)
            outputs[k] = value
        return outputs

    # NOTE: We need to take the params and state, as we cannot
    # find it anymore, since we have new classes when we call this!
    def copy_params_and_state_from(self, src_params, src_state):
        _copy(src=src_params, target=self.find_params())
        _copy(src=src_state, target=self.find_state())


class _MathModule(Module):

    def __init__(self, op, left, right):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

    def out(self, ts):
        left = self._maybe_call(self.left, ts)
        right = self._maybe_call(self.right, ts)
        return self.op(left, right)

    @staticmethod
    def _maybe_call(module_or_number, ts):
        if isinstance(module_or_number, Module):
            return module_or_number(ts)
        return module_or_number


# TODO: Revisit the whole API.
#  The idea of this class is to support modules that do not rely on a clock_signal.
#  E.g. Envelope Generators.
class InputLessModule(Module):

    def get_output(self):
        pass


class GaussEnvelopeGenerator(InputLessModule):

    def __init__(self, elen):
        super().__init__()
        self.cache = None
        self.elen: Parameter = elen

    def get_output(self):
        elen = round(self.elen.get())
        attack = np.linspace(0, 2, elen//4)
        peak = np.linspace(2, 1, elen//10)
        hold = np.ones(elen*2)
        decay = np.linspace(1, 0, elen)#//8)
        zeros = np.zeros(elen*4)
        #return np.array(10*[np.exp(-(x-(elen/2))**2 * 0.001) for x in range(elen)])
        #x = np.arange(elen)
        #return 10*np.exp(-(x-(elen/2))**2 * 0.001)
        return np.concatenate((attack,
                               peak,
                               hold,
                               decay,
                               zeros), 0)


class EnvelopeGenerator(InputLessModule):

    def __init__(self, elen):
        super().__init__()
        self.cache = None
        self.elen: Parameter = elen

    def get_output(self):
        elen = round(self.elen.get())
        attack = np.linspace(0, 1, elen)
        peak = np.linspace(1, 1.1, elen//4)
        hold = np.ones(elen)
        decay = np.linspace(1, 0, elen*4)
        return np.concatenate((attack, peak, hold, decay), 0)


class ADSREnvelopeGenerator(InputLessModule):

    def __init__(self, attack, decay, sustain, release, hold):
        super().__init__()
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.hold = hold or Parameter(100)

    def get_output(self):
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



class Constant(Module):

    def __init__(self, value):
        super().__init__()
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
        out = np.broadcast_to(self.value, clock_signal.ts.shape)
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
        # TODO: Support selecting from a predefined set (e.g. all nice notes).
        if lo is None:
            lo = 0.1 * value
        if not hi:
            hi = max(1, 1.9 * value)
        if hi < lo:
            raise ValueError
        self.lo, self.hi = lo, hi
        self.span = self.hi - self.lo
        self.key = key
        self.knob = knob
        self.shift_multiplier = shift_multiplier
        self.clip = clip

    def __repr__(self):
        return f"Parameter({self.value},{self.knob})"

    def set(self, value):
        if self.clip:
            self.value = np.clip(value, self.lo, self.hi)
        else:
            self.value = value

    def set_relative(self, rel_value: float):
        """Set with value in [0, 1], and we map to [lo, hi]."""
        self.set(self.lo + self.span * rel_value)


class Random(Module):
    """Output a constant random amplitude until a random change event changes the amplitude. Best explanation ever."""
    def __init__(self, max_amplitude, change_chance):
        super().__init__()
        self.max_amplitude = max_amplitude
        self.p = change_chance
        self.amp = random.random() * self.max_amplitude

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        # random experiment
        frame_length, num_channels = clock_signal.ts.shape
        res = np.empty((0, num_channels))
        while len(res) < frame_length:
            chance_in_frame = 1 - pow(1 - self.p, frame_length - len(res))
            # to make it independent of frame length, this goes on for a random amount of time
            block_len = random.randint(1, frame_length - len(res))
            res = np.concatenate((res, np.ones((block_len, num_channels)) * self.amp))
            if random.random() < chance_in_frame:
                self.amp = random.random() * self.max_amplitude  # could discretize amps to guarantee pleasant ratios
        return res


class SineSource(Module):
    def __init__(self, frequency: Module, amplitude=Parameter(1.0), phase=Parameter(0.0)):
        super().__init__()
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


class SawSource(Module):
    def __init__(self, frequency: Module, amplitude=Parameter(1.0), phase=Parameter(0.0)):
        super().__init__()
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def out(self, clock_signal: ClockSignal):
        amp = self.amplitude(clock_signal)
        freq = self.frequency(clock_signal)
        phase = self.phase(clock_signal)
        period = 1 / freq
        # TODO: Do we need cumsum here?
        out = 2 * (clock_signal.ts/period + phase
                   - np.floor(1/2 + clock_signal.ts/period + phase)) * amp
        return out


class TriangleSource(Module):
    def __init__(self, frequency: Module, amplitude=Parameter(1.0), phase=Parameter(0.0)):
        super().__init__()
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        amp = self.amplitude(clock_signal)
        freq = self.frequency(clock_signal)
        phase = self.phase(clock_signal)
        period = 1 / freq
        # TODO: Do we need cumsum here?
        out = (2 * np.abs(2 * (clock_signal.ts/period + phase
                               - np.floor(1/2 + clock_signal.ts/period + phase))) - 1) * amp
        return out


class NoiseSource(Module):
    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        return np.random.random(clock_signal.ts.shape)



class SineModulator(Module):
    def __init__(self, inp: Module, carrier_frequency: Module, inner_amplitude=Parameter(1.0)):
        super().__init__()
        self.carrier = SineSource(carrier_frequency, amplitude=inner_amplitude)
        self.inp = inp
        self.out = self.carrier * self.inp


class KernelGenerator(Module):
    """
    Takes a value generating function func and a length parameter.
    Produces a kernel of length length for every ts, whose values are generated by func.
    TODO: consider: could make func dependent on ts, or even have a func-generating Module
    Returns a shape (frame_length, max_kernel_size, num_channels).
    """

    def __init__(self, func, length: Module):
        super().__init__()
        self.func = func  # function from t to [-1, 1]
        self.length = length  # kernel length, a module

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        norm = lambda v: v / np.linalg.norm(v)
        # we dont need the inp
        lengths = self.length(clock_signal)  # shape (512, 1)
        max_len = int(np.max(lengths))
        frame_length, num_channels = lengths.shape
        out = np.array([norm([self.func(i) if i < int(l) else 0 for i in range(max_len)]) for l in lengths])
        return out.reshape((frame_length, max_len, num_channels))


# TODO: WIP
class FrameBuffer:
    """Store multiple frames."""

    def __init__(self):
        self._buffer: Optional[collections.deque] = None

    def push(self, frame: np.ndarray, max_frames_to_buffer: int):
        buffer = self._update_buffer(frame.shape, max_frames_to_buffer)
        buffer.append(frame)  # We append on the right.

    def get(self) -> np.ndarray:
        return np.concatenate(list(self.iter_buffered()), axis=0)

    def iter_buffered(self) -> typing.Iterable[np.ndarray]:
        assert self._buffer
        return iter(self._buffer)

    def _update_buffer(self, frame_shape: typing.Tuple[int, int], max_frames_to_buffer: int) -> collections.deque:
        if self._buffer is None:
            # Start out with just 0s
            initial_buffer = [np.broadcast_to(0., frame_shape) for _ in range(max_frames_to_buffer)]
            self._buffer = collections.deque(initial_buffer, maxlen=max_frames_to_buffer)
            return self._buffer
        if self._buffer.maxlen == max_frames_to_buffer:
            return self._buffer
        # If we land here, our buffer has a different `maxlen` than `max_frames_to_buffer`, and
        # we need to either shrink or grow.
        if self._buffer.maxlen > max_frames_to_buffer:  # Need to shrink
            self._buffer = collections.deque(self._buffer[-max_frames_to_buffer:],
                                             maxlen=max_frames_to_buffer)
            return self._buffer
        else:  # Need to grow
            self._buffer = collections.deque(self._buffer, maxlen=max_frames_to_buffer)
            while len(self._buffer) < max_frames_to_buffer:
                self._buffer.appendleft(np.broadcast_to(0., frame_shape))
            return self._buffer


class KernelConvolver(Module):
    """
    Takes a kernel-generator to smooth the input.
    TODO: some python loops seem unavoidable, but maybe there are numpy formulations to save effort.
    """
    def __init__(self, inp: Module, kernel_generator: Module):
        super().__init__()
        self.inp = inp
        self.kernel_generator = kernel_generator
        self.last_signal = None

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        if self.last_signal is None:
            self.last_signal = clock_signal.zeros()
        num_samples, num_channels = clock_signal.ts.shape
        inp = self.inp(clock_signal)
        full_signal = np.concatenate((self.last_signal, inp), axis=0)
        self.last_signal = inp
        kernels = self.kernel_generator(clock_signal)
        # shape must be (frame_length, max_kernel_size, num_channels)
        frame_length, max_kernel_size, num_channels = kernels.shape
        slices = np.concatenate([full_signal[i:i+max_kernel_size, :]
                                 for i in range(frame_length - max_kernel_size + 1,
                                                frame_length * 2 - max_kernel_size + 1)])
        slices = slices.reshape(kernels.shape)
        res_block = np.tensordot(slices, kernels, axes=[[1, 2], [1, 2]])
        res = np.diag(res_block).reshape(clock_signal.shape)
        return res



class SimpleLowPass(Module):
    """Simplest lowpass: average over previous <window> values"""
    def __init__(self, inp: Module, window_size: Module):
        super().__init__()
        self.window_size = window_size
        self.last_signal = State()
        self.inp = inp

    def out(self, clock_signal: ClockSignal):
        if not self.last_signal.is_set:
            self.last_signal.set(clock_signal.zeros())
        num_samples, num_channels = clock_signal.ts.shape
        input = self.inp(clock_signal)
        # Shape: (2*num_frames, num_channels)
        full_signal = np.concatenate((self.last_signal.get(), input), axis=0)
        window_sizes = self.window_size(clock_signal)
        # TODO: Now we have one window size per frame. Seems reasonable?
        # Maybe we want to have a "MapsToSingleValueModule".
        window_size: int = max(1, round(float(np.mean(window_sizes))))
        mean_filter = np.ones((window_size,), dtype=clock_signal.ts.dtype) / window_size
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
        assert output.shape == clock_signal.ts.shape, (
            output.shape, clock_signal.ts.shape, window_size, start_time_index)
        return output


class ButterworthFilter(Module):
    """Simplest lowpass: average over previous <window> values"""
    def __init__(self, inp: Module, f_low: Module, f_high: Module, mode: str = "hp", order: int = 10):
        super().__init__()
        self.last_signal = State()
        self.inp = inp
        self.f_low = f_low
        self.f_high = f_high
        self.mode = mode
        self.order = order

    def out(self, clock_signal: ClockSignal):
        if not self.last_signal.is_set:
            self.last_signal.set(clock_signal.zeros())
        num_samples, num_channels = clock_signal.ts.shape
        inp = self.collect("inp") <<  self.inp(clock_signal)
        f_low = self.f_low(clock_signal)[0, 0]
        f_high = self.f_high(clock_signal)[0, 0] + f_low

        f_low = np.clip(f_low, 1e-10, SAMPLING_FREQUENCY/2)
        f_high = np.clip(f_high, 1e-10, SAMPLING_FREQUENCY/2)

        full_signal = np.concatenate((self.last_signal.get(), inp), axis=0)
        self.last_signal.set(inp)
        self.last_signal.set(inp)

        fs = {"lp": f_low, "hp": f_high, "bp": (f_low, f_high)}[self.mode]
        sos = get_me_some_butter(self.order, fs, self.mode)
        filtered_signal = self.collect("filtered") << signal.sosfilt(sos, full_signal[:,0])[-num_samples:, np.newaxis]

        #print(max(filtered_signal[-num_samples:, :]))
        return filtered_signal[-num_samples:, :]


@functools.lru_cache(maxsize=128)
def get_me_some_butter(order, fs, mode):
    print("MAKING BUTTER")
    return signal.butter(order, fs, mode, fs=SAMPLING_FREQUENCY, output='sos')


class ShapeModulator(Module):
    """
    Modulate a given shape onto clicks in time domain. Nearby clicks will both get the shape, so they may overlap.
    """
    def __init__(self, inp: Module, shape: Module):
        super().__init__()
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
        return out[-clock_signal.ts.shape[0]:, :]


class ShapeExp(Module):
    """Gives a signal of length shape_length"""
    def __init__(self, shape_length: int, decay=2.0, amplitude=1.0):
        super().__init__()
        self.shape_length = shape_length
        self.decay = decay
        self.amplitude = amplitude

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        shape = np.array([self.amplitude / pow(self.decay, i) for i in range(self.shape_length)]).reshape(-1,1)
        return shape


class Lift(Module):
    """Lifts a signal from [-1,1] to [0,1]"""
    def __init__(self, inp: Module):
        super().__init__()
        self.inp = inp

    def out(self, clock_signal: ClockSignal):
        res = self.inp(clock_signal)
        return res / 2 + 0.5


class ScalarMultiplier(Module):
    """Needed for inner generators, so that we can have a changing frequency that is, e.g., not just between [0,1] but
    between [0,440]"""
    # we could still pass a module as the value, instead of a constant float..
    def __init__(self, inp: Module, value: float):
        super().__init__()
        self.inp = inp
        self.value = value

    def __call__(self, clock_signal: ClockSignal):
        return self.inp(clock_signal) * self.value


class Multiplier(Module):  # TODO: variadic input
    def __init__(self, inp1: Module, inp2: Module):
        super().__init__()
        self.inp1 = inp1
        self.inp2 = inp2

    def out(self, clock_signal: ClockSignal):
        return self.inp1(clock_signal) * self.inp2(clock_signal)


class PlainMixer(Module):
    """Adds all input signals without changing their amplitudes"""
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        inputs = iter(self.args)
        output = next(inputs)(clock_signal)
        for inp in inputs:
            output += inp(clock_signal)
        return output / len(self.args)


# TODO: Not needed with suport for mul and add I think?
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
        super().__init__()
        self.num_samples = num_samples
        self.counter = 0

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        num_samples = int(np.mean(self.num_samples(clock_signal))) # hack to have same blocksize per frame, like Lowpass...
        out = clock_signal.zeros()
        for i in range(int(np.ceil((clock_signal.ts.shape[0]-self.counter) / num_samples))):
            out[self.counter + i * num_samples, :] = 1
        self.counter += num_samples - (clock_signal.ts.shape[0] % num_samples)
        self.counter = self.counter % clock_signal.ts.shape[0]
        self.counter = self.counter % num_samples

        self.counter = self.collect("counter") << self.counter

        #print("click out", out)
        return out
        # TODO: buggy for num_samples greater than frame_length!


############################################
# ======== Test composite modules ======== #

def test_module(module: Module, num_frames=5, frame_length=2048, num_channels=1, sampling_frequency=44100, show=True):
    res = []
    for i in range(num_frames):
        ts = (i*frame_length + np.arange(frame_length)) / sampling_frequency
        ts = ts[..., np.newaxis] * np.ones((1,))
        out = module(ts)  # TODO: USe clocksignal
        res.append(out)
    res = np.concatenate(res)
    plt.plot(res)
    plt.vlines([i * frame_length for i in range(0, num_frames+1)], ymin=np.min(res)*1.1, ymax=np.max(res)*1.1, linewidth=0.8, colors='r')
    plt.hlines(0, -len(res)*0.1, len(res)*1.1, linewidth=0.8, colors='r')
    if show:
        plt.show()


def kernel_test():
    ts = np.arange(512) / 44100
    ts = ts[..., np.newaxis] * np.ones((1,))
    length = PlainMixer(Parameter(1), ScalarMultiplier(Lift(SawSource(frequency=Parameter(10000))), 10))
    k = KernelGenerator(Parameter(1), lambda x: x*x, length=length)
    print("k", k.out(ts[:10, :]))
    print(k.out(ts[:10, :]).shape)
    print("---")
    #lp = KernelConvolver(src, kernel_generator=k)

#kernel_test()



class ClickModulation(Module):
    def __init__(self):
        super().__init__()
        #self.out = SineModulator(ShapeModulator(ClickSource(Parameter(400)), ShapeExp(200, decay=1.01)), carrier_frequency=Parameter(220))
        self.wild_triangles = PlainMixer(*[TriangleSource(frequency=Random(220 * i, 0.000015)) for i in range(1, 3)])
        self.out = KernelConvolver(self.wild_triangles, KernelGenerator(lambda x: 1, length=Parameter(100)))
        test_module(self.out, num_frames=10)

        #self.out = TriangleSource(Parameter(220))
        #self.one = TriangleSource(frequency=Random(110, 0.00003))
        #self.two = TriangleSource(frequency=Random(440, 0.00006))
        #self.out = PlainMixer(self.one, self.two)

        #self.out = PlainMixer(*[TriangleSource(frequency=Random(110 * i, 0.000015 )) for i in range(1, 4)])
        #self.out = SineSource(frequency=PlainMixer(Parameter(220), Multiplier(Lift(TriangleSource(frequency=Parameter(1))), Parameter(1))))


class BabiesFirstSynthie(Module):
    def __init__(self):
        super().__init__()
        self.lfo = SineSource(Parameter(1))
        self.sin0 = SineSource(frequency=Parameter(440*(2/3)*(2/3)))
        self.sin1 = SineSource(frequency=Parameter(440))
        self.sin2 = SineSource(frequency=Parameter(220))

        #self.out = PlainMixer(self.sin0, self.sin1, self.sin2)

        self.changingsine0 = Multiplier(self.sin0, self.lfo)
        self.changingsine1 = SineModulator(self.sin0, Parameter(1))
        self.lowpass = SimpleLowPass(self.changingsine0, window_size=Parameter(2))

        self.src = SineSource(ScalarMultiplier(Lift(SineSource(Parameter(10))), 22))
        self.modulator = SineModulator(self.src, Parameter(10))
        self.lp = SimpleLowPass(self.modulator, window_size=Parameter(16))
        self.out = self.lowpass


class FreqFactors(enum.Enum):
    OCTAVE = 2.
    STEP = 1.059463





@dataclasses.dataclass
class StepSequencer(Module):

    wave_generator_cls: typing.Type[Module]
    base_frequency: Parameter
    bpm: Parameter
    melody: typing.Sequence[int]
    steps: typing.Sequence[int]
    # Gate mode:
    # - H: hold for the step count,    XXXXXX
    # - E: Sound for each step count   X X X
    # - F: Sound for first step count  X
    # - S: Skip, no sound
    # gate='HSHSHS'
    gate: str
    melody_randomizer: Optional[Parameter] = None
    env_param: Parameter = None
    envelope: Module = None

    def __post_init__(self):
        self.collect = collector_lib.FakeCollector()
        #assert all(1 <= m <= 12 for m in self.melody), self.melody
        self.seq_len = len(self.melody)
        self.frequency = Parameter(1)
        self.waveform_generator = self.wave_generator_cls(frequency=self.frequency)

        self.gate = "".join(itertools.islice(itertools.cycle(self.gate),
                                             len(self.melody)))

        self.melody_cycler = _Cycler(self.melody)

        self.trigger = TriggerMaker(self.bpm)
        self.env_param = self.env_param or Parameter(200)
        self.envelope_generator = self.envelope or EnvelopeGenerator(self.env_param)
        self.envelope_future_cache = FutureCache()

        if self.melody_randomizer:
            # TODO
            class BinaryTrigger(Module):

                def __init__(self, m: Module):
                    super().__init__()
                    self.m = m
                    self.prev_t = False

                def out(self, clock_signal: ClockSignal):
                    t = self.m(clock_signal).max() > 0
                    # TODO: BUG
                    if t and t != self.prev_t:
                        signal = 1.
                    else:
                        signal = 0.
                    self.prev_t = t
                    return clock_signal.add_channel_dim(np.array([signal]))

            self.melody_randomizer_trigger = BinaryTrigger(self.melody_randomizer)
        else:
            self.melody_randomizer_trigger = None

    def copy(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def out(self, clock_signal: ClockSignal) -> np.ndarray:

        if self.melody_randomizer_trigger:
            trigger = self.melody_randomizer_trigger(clock_signal).max() > 0
            if trigger:
                # num_steps = len(self.melody_cycler.xs)
                base = random.choices([(3*i)%12 for i in range(12)], k=random.randint(3, 32))
                random.shuffle(base)
                print("New melody", base)
                self.melody_cycler.xs = base
                self.melody_cycler.i = 0


        triggers = self.collect("triggers") << self.trigger(clock_signal)
        # (E,), E >>> num_samples
        envelope = self.collect("raw") << self.envelope_generator.get_output()
        # (E+T,)
        future = self.collect("future") << _scatter_into_mask(triggers, envelope)

        current_i = clock_signal.sample_indices[0]
        self.envelope_future_cache.prepare(current_i, num_samples=clock_signal.num_samples)
        self.envelope_future_cache.add(future)

        current_envelope = self.collect("envelope") << self.envelope_future_cache.get(
            clock_signal.num_samples)


        # Convert to array
        melody = to_melody(triggers, self.melody_cycler)
        melody = clock_signal.add_channel_dim(melody)
        base_freq = self.base_frequency(clock_signal)[0]
        melody = self.collect("melody") << melody
        gate = "H"  # TODO...
        if gate == "H":
            self.frequency.set(base_freq * FreqFactors.STEP.value ** melody)
            return self.waveform_generator(clock_signal) * current_envelope.reshape(-1, 1)
        elif gate == "S":
            return clock_signal.zeros()


class _Cycler:
    """Cycles between `xs` with a nice interface."""

    def __init__(self, xs):
        self.xs = xs
        self.i = 0

    @property
    def current(self):
        return self.xs[self.i]

    def inc(self):
        self.i = (self.i + 1) % len(self.xs)


def _scatter_into_mask(mask, arr):
    """Put `arr` into output wherever `mask` is 1.

    E.g:
        arr =    np.array([5, 4, 3, 2, 1])
        mask =   np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,...])
        output = np.array([0, 0, 5, 4, 3, 2, 1, 0, 5, 4, 3, 2, ...])

        arr =    np.array([5, 4, 3, 2, 1])
        mask  =   [1, 0]
        output =  [5, 4, 3, 2, 1]

        arr =    np.array([5, 4, 3, 2, 1])
        mask =   np.array([0, 0, 1, 0, 0, 1, 0, 0, 0,...])
        output = np.array([0, 0, 5, 4, 3, 5, 4, 3, 2, ...])

    Revisit after https://stackoverflow.com/questions/70047986/replace-a-single-value-with-multiple-values/70048906#70048906
    """
    mask_len, arr_len = mask.shape[0], arr.shape[0]
    output = np.zeros((mask_len + arr_len, ), dtype=arr.dtype)
    last_index = 0
    for i, x in enumerate(mask):
        if x > 0.5:
            output[i:i + arr_len] = arr
            last_index = i + arr_len
    return output[:last_index]


def to_melody(trigger, melody: _Cycler):
    trigger = trigger[:, 0]  # TODO: first channel.
    output = []
    i, = np.where(trigger > 0.5)
    for split in np.split(trigger, i):
        if len(split) == 0:  # First element of trigger is a 1.
            continue
        if split[0] > 0.5:
            melody.inc()
        output.append(np.ones_like(split) * melody.current)
    return np.concatenate(output, 0)


class Reverb(Module):
    def __init__(self, m, alpha: Module, max_decay: Module):
        super().__init__()
        self.m = m
        self.max_decay = max_decay
        self.alpha = alpha
        self.num_decays = 50
        self.state = State(initial_value=None)

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        if self.state.get() is None or self.state.get().maxlen != self.num_decays:
            self.state.set(collections.deque((clock_signal.zeros() for _ in range(self.num_decays)),
                                             maxlen=self.num_decays))
        d = self.state.get()

        max_decay = self.max_decay(clock_signal)[0, 0]
        alpha = self.alpha(clock_signal)[0, 0]
        decays = np.array([1/max(1e-4, (2**(alpha*i))) for i in range(self.num_decays)]) * max_decay
        o = self.m(clock_signal) + sum(decay * s for decay, s in zip(reversed(decays), d))
        d.append(o)
        self.state.set(d)
        return o



class BaseDrum(Module):
    def __init__(self):
        super().__init__()


class FooBar(Module):
    def __init__(self):
        super().__init__()

        self.base_freq = Parameter(220, lo=220 / 4, hi=440, knob="r_mixer_hi", key="u", clip=True)
        self.bpm = Parameter(160, lo=10, hi=300, knob="r_tempo")

        self.env_param_bass = Parameter(200, lo=100, hi=20000, knob="r_mixer_mi", key="b", clip=True)
        self.env_param_foo = Parameter(200, lo=100, hi=20000, knob="r_mixer_lo", key="c", clip=True)

        self.melody1 = StepSequencer(
            wave_generator_cls=TriangleSource,
            base_frequency=self.base_freq,
            bpm=self.bpm,
            melody=[0, 0, 0, 4, 0, 0, 0, 12, 0, 0, 0, 4],
            steps=[1],
            gate='H',
            env_param=self.env_param_foo,
            #melody_randomizer=Parameter(0, knob="r_4"),
        )

        self.melody2 = self.melody1.copy(
            base_frequency=self.base_freq * (FreqFactors.STEP.value ** 7))
        self.melody3 = self.melody1.copy(
            base_frequency=self.base_freq * (FreqFactors.STEP.value ** 4))

        self.melody = (self.melody1 +
                       self.melody2 +
                       self.melody3)

        self.attack_t = Parameter(200, lo=1, hi=2000, knob="fx1_0")
        self.decay_t = Parameter(10, lo=1, hi=20000, knob="fx1_1")
        self.sustain_t = Parameter(0.5, lo=0.01, hi=1., knob="fx1_2")
        self.release_t = Parameter(200, lo=1, hi=10000, knob="fx1_3")
        self.hold_t = Parameter(200)

        self.bass = self.melody1.copy(
            wave_generator_cls=SineSource,
            bpm=self.bpm/2,
            melody=[1, 4],#random.choices([1,2,4,7,9], k=3),
            env_param=self.env_param_bass,
            envelope=ADSREnvelopeGenerator(
                self.attack_t,
                self.decay_t,
                self.sustain_t,
                self.release_t,
                self.hold_t,
            ),
            base_frequency=self.base_freq / 4)

        self.mixer = Parameter(0.5, lo=0., hi=.8, knob="fx2_1", key="q", clip=True)
        self.mixer_m = Parameter(0.5, lo=0., hi=.8, knob="fx2_2", key="w", clip=True)

        self.out = (self.mixer * self.melody +
                    self.mixer_m * self.bass * 2
        )


        return



        self.out = (self.mixer_m * 1/3*(self.melody1 + self.melody2 + self.melody3)
                    + self.mixer * self.bass)


P = Parameter
from scipy import signal


class Buttering(Module):
    def __init__(self):
        super().__init__()

        self.base_freq = Parameter(220, lo=220/4, hi=440, key="q")
        self.base_freq2 = Parameter(220/4, lo=220/16, hi=440, key="w")
        self.bpm = Parameter(100, lo=10, hi=300, key="e", clip=True)
        bpm_melody = self.bpm * 2
        self.e =Parameter(2000, key="x")
        self.melody_highs = StepSequencer(
            SawSource,
            self.base_freq,
            bpm_melody,
            melody=[1, 0, 12, 11, 8, 1],
            #melody=[1, 2, 3, 4],
            steps=[1],
            gate='SSSH',
            envelope=EnvelopeGenerator(self.e),
            melody_randomizer=Parameter(0, knob="z")
        )
        self.melody_lows = self.melody_highs.copy(base_frequency=self.base_freq/2,
                                                  wave_generator_cls=SawSource,)

        self.mixer_c1 = Parameter(0.5, lo=0, hi=1.5, knob="fx2_1")
        self.mixer_c2 = Parameter(0.5, lo=0, hi=1.5, knob="fx2_2")
        self.melody = (self.mixer_c1 * self.melody_highs +
                       self.mixer_c2 * self.melody_lows)

        self.step_bass = StepSequencer(
            wave_generator_cls=SineSource,
            base_frequency=self.base_freq2,
            bpm=self.bpm,
            melody=[1, 5, 3, 5],
            steps=[1],
            gate="SSSH")

        self.filtered = ButterworthFilter(inp=self.melody+self.step_bass*2, f_low=P(0.01, key="g"), f_high=P(10000, key="h"), mode="bp")
        self.out = self.filtered




class StepSequencing(Module):
    def __init__(self):
        super().__init__()
        self.base_freq = Parameter(220, lo=220/4, hi=440, knob="r_mixer_hi")
        self.base_freq2 = Parameter(220/4, lo=220/16, hi=440, knob="r_mixer_mi")
        self.bpm = Parameter(100, lo=10, hi=300, knob="r_tempo")
        bpm_melody = self.bpm * 2
        self.melody_highs = StepSequencer(
            SawSource,
            self.base_freq,
            bpm_melody,
            melody=[1,2,3,4,5,6,7,8],
            #melody=[1, 2, 3, 4],
            steps=[1],
            gate='SSSH',
            melody_randomizer=Parameter(0, knob="r_4")
        )
        self.melody_lows = self.melody_highs.copy(base_frequency=self.base_freq/2,
                                                  wave_generator_cls=SawSource,)

        self.mixer_c1 = Parameter(0.5, lo=0, hi=1.5, knob="fx2_1")
        self.mixer_c2 = Parameter(0.5, lo=0, hi=1.5, knob="fx2_2")
        self.melody = (self.mixer_c1 * self.melody_highs +
                       self.mixer_c2 * self.melody_lows)

        self.lp_melody = SimpleLowPass(
            self.melody, window_size=Parameter(10, hi=2000, key="w",
                                               knob="fx2_3"))

        self.step_bass = StepSequencer(
            wave_generator_cls=SineSource,
            base_frequency=self.base_freq2,
            bpm=self.bpm,
            melody=[1, 5, 3, 5],
            steps=[1],
            gate="SSSH")


#        self.step_highs = StepSequencer(
#            wave_generator_cls=SineSource,
#            base_frequency=Parameter(220),
#            bpm=self.bpm,
#            melody=[1, 1, 1, 1],
#            steps=[1],
#            gate="SHSH")


        self.out = self.lp_melody  # + self.step_bass  # + self.step_highs


class TestModule(Module):
    def __init__(self):
        super().__init__()
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


class FutureCache:

    def __init__(self):
        self.curr_arr = np.zeros((10,), dtype=np.float32)
        self.curr_i = 0

    def prepare(self, i, num_samples):
        if i > self.curr_i:
            self.curr_arr = self.curr_arr[num_samples:]
            self.curr_i = i

    def _adapt_arr_len(self, arr_len):
        if arr_len <= self.curr_arr.shape[0]:
            return
        missing_zeros = np.zeros((arr_len - self.curr_arr.shape[0],), dtype=self.curr_arr.dtype)
        self.curr_arr = np.concatenate((self.curr_arr, missing_zeros), axis=0)

    def get(self, num_samples):
        self._adapt_arr_len(num_samples)
        return self.curr_arr[:num_samples]

    def add(self, arr):
        assert len(arr.shape) == 1, arr.shape  # TODO: GENERALIZE
        self._adapt_arr_len(arr.shape[0])
        self.curr_arr[:arr.shape[0]] = np.add(self.curr_arr[:arr.shape[0]], arr)


class TriggerMaker(Module):

    def __init__(self, bpm: Module):
        self.bpm = bpm

    def out(self, clock_signal: ClockSignal):
        bpm = self.bpm(clock_signal).max().round()  # TODO
        bps = bpm / 60  # Per second.
        period_in_sec = 1/bps
        period_in_sample_space = round(period_in_sec * clock_signal.sample_rate)
        return clock_signal.add_channel_dim(
            (clock_signal.sample_indices % period_in_sample_space) == 0)


def _intersperse_gaps(vs, gap_len=100):
    o = []
    _, *shape = vs[0].shape
    gap = np.zeros((gap_len, *shape))
    for v in vs:
        o.append(v)
        o.append(gap)
    o.pop()
    return o


def plot_module(synthie_cls, plot=("(melody1|bass).*",), num_steps=4):
    m = synthie_cls()
    #m.env_param_foo.set(10000)
    #m.bpm.set(1000)

    clock = Clock(num_samples=2048, num_channels=1, sample_rate=44100)
    ts, output = m.collect_data(num_steps=num_steps, clock=clock)
    output_to_plot = [(k, vs) for k, vs in output
                      if any(re.fullmatch(r, k) for r in plot)]

    s = plot_lib.Subplots(nrows=len(output_to_plot), width=10)
    for k, vs in output_to_plot:
        ax = s.next_ax()
        vs_arr = np.concatenate(vs, axis=0)
        if vs_arr.shape[0] == ts.shape[0]:
            ax.plot(ts, vs_arr, label=k)
            for i in range(1, num_steps):
                ax.axvline(ts[round(i/num_steps*ts.shape[0])], ls=":")
        else:
            i = 0
            for v in vs:
                ax.plot(range(i, i+len(v)), v, label=k)
                i += len(v) + 100
        ax.legend()
    plt.show()


if __name__ == '__main__':
    plot_module(Buttering, plot=[".*"])

