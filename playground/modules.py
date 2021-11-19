import collections
import dataclasses
import enum
from functools import reduce, lru_cache
from typing import Mapping, Union, MutableMapping, Optional

import numpy as np
import typing

import midi_lib
import time
import random

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
            if isinstance(var_instance, Module):
                result.update(var_instance._find(cls, prefix=f"{var_name}."))
        return result

    # NOTE: We need to take the params and state, as we cannot
    # find it anymore, since we have new classes when we call this!
    def copy_params_and_state_from(self, src_params, src_state):
        _copy(src=src_params, target=self.find_params())
        _copy(src=src_state, target=self.find_state())


class Constant(Module):
    def __init__(self, value):
        self.value = value

    def out(self, ts):
        # TODO: sounds cool
        # num_samples, num_channels = ts.shape
        # if abs(self.previous_value - self.value) > 1e-4:
        #     out = (np.linspace(self.previous_value, self.value, num_samples).reshape(-1, 1) *
        #            np.ones((num_channels,)))
        #     print(self.previous_value, self.value, out[:10])
        # else:
        #     out = np.ones_like(ts) * self.value
        out = np.broadcast_to(self.value, ts.shape)
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
        if not lo:
            lo = 0.1 * value
        if not hi:
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


class Random(Module):
    """Output a constant random amplitude until a random change event changes the amplitude. Best explanation ever."""
    def __init__(self, max_amplitude, change_chance):
        self.max_amplitude = max_amplitude
        self.p = change_chance
        self.amp = random.random() * self.max_amplitude

    def out(self, ts: np.ndarray) -> np.ndarray:
        # random experiment
        frame_length, num_channels = ts.shape
        res = np.empty((0, num_channels))
        while len(res) < frame_length:
            chance_in_frame = 1 - pow(1 - self.p, frame_length - len(res))
            # to make it independent of frame length, this goes on for a random amount of time
            block_len = random.randint(1, frame_length - len(res))
            res = np.concatenate((res, np.ones((block_len, num_channels)) * self.amp))
            if random.random() < chance_in_frame:
                self.amp = random.random() * self.max_amplitude  # could discretize amps to guarantee pleasant ratios
        return res


def get_res(query_in, raising, f):
    res = np.arcsin(query_in) / f
    if not raising:
        return np.pi / f - res
    return res


class SineSource(Module):
    def __init__(self, frequency: Module, amplitude=Parameter(1.0), phase=Parameter(0.0)):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.last_cumsum_value = State(initial_value=0)

    def out(self, ts):
        dt = np.mean(ts[1:] - ts[:-1])
        amp = self.amplitude(ts)
        freq = self.frequency(ts)
        phase = self.phase(ts)
        cumsum = np.cumsum(freq * dt, axis=0) + self.last_cumsum_value.get()
        # Cache last cumsum for next step.
        self.last_cumsum_value.set(cumsum[-1, :])
        out = amp * np.sin((2 * np.pi * cumsum) + phase)
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

    def __init__(self, func, length: Module):
        self.func = func  # function from t to [-1, 1]
        self.length = length  # kernel length, a module

    def out(self, ts: np.ndarray) -> np.ndarray:
        norm = lambda v: v / np.linalg.norm(v)
        # we dont need the inp
        lengths = self.length(ts)  # shape (512, 1)
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
        self.inp = inp
        self.kernel_generator = kernel_generator
        self.last_signal = None

    def out(self, ts: np.ndarray) -> np.ndarray:
        if self.last_signal is None:
            self.last_signal = np.zeros_like(ts)
        num_samples, num_channels = ts.shape
        inp = self.inp(ts)
        full_signal = np.concatenate((self.last_signal, inp), axis=0)
        self.last_signal = inp
        kernels = self.kernel_generator(ts)
        # shape must be (frame_length, max_kernel_size, num_channels)
        frame_length, max_kernel_size, num_channels = kernels.shape
        slices = np.concatenate([full_signal[i:i+max_kernel_size, :] for i in range(frame_length - max_kernel_size + 1, frame_length * 2 - max_kernel_size + 1)])
        slices = slices.reshape(kernels.shape)
        res_block = np.tensordot(slices, kernels, axes=[[1, 2], [1, 2]])
        res = np.diag(res_block).reshape(ts.shape)
        return res



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
        self.args = args

    def out(self, ts: np.ndarray) -> np.ndarray:
        o = reduce(np.add, [inp(ts) for inp in self.args]) / (len(self.args))
        return o / 3


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



@dataclasses.dataclass()
class StepSequencer(Module):

    waveform_generator_cls: typing.Type[Module]
    base_frequency: Parameter
    speed: Parameter
    melody: typing.Sequence[int]
    steps: typing.Sequence[int]
    gate: str

    def __post_init__(self):
        assert all(1 <= m <= 12 for m in self.melody), self.melody
        self.seq_len = len(self.melody)
        self.frequency = Parameter(1)
        self.waveform_generator = self.waveform_generator_cls(frequency=self.frequency)

    def copy(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def out(self, ts: np.ndarray) -> np.ndarray:
        # TODO
        speed = self.speed(ts)[0]
        base_freq = self.base_frequency(ts)[0]
        t = int(np.ceil(ts[0] * speed).round().astype(np.int)) % self.seq_len
        m = self.melody[t]
        gate = self.gate[t]
        if gate == "H":
            self.frequency.set(base_freq * FreqFactors.STEP.value ** m)
            return self.waveform_generator(ts)
        elif gate == "S":
            return np.zeros_like(ts)


class Reverb(Module):
    def __init__(self, m, alpha: Module, max_decay: Module):
        self.m = m
        self.max_decay = max_decay
        self.alpha = alpha
        self.num_decays = 50
        self.state = State(initial_value=None)

    def out(self, ts: np.ndarray) -> np.ndarray:
        if self.state.get() is None or self.state.get().maxlen != self.num_decays:
            #num_samples, _ = ts.shape
            #state_size = len(self.envelope) // num_samples
            self.state.set(collections.deque((np.zeros_like(ts) for _ in range(self.num_decays)),
                                             maxlen=self.num_decays))
        d = self.state.get()

        max_decay = self.max_decay(ts)[0, 0]
        alpha = self.alpha(ts)[0, 0]
        decays = np.array([1/max(1e-4, (2**(alpha*i))) for i in range(self.num_decays)]) * max_decay
        o = self.m(ts) + sum(decay * s for decay, s in zip(reversed(decays), d))
        d.append(o)
        self.state.set(d)
        return o



class StepSequencing(Module):
    def __init__(self):
        self.base_freq = Parameter(220, lo=220/4, hi=440, knob="r_mixer_hi")
        self.base_freq2 = Parameter(220, lo=220/4, hi=440, knob="r_mixer_mi")
        self.speed = Parameter(2, lo=1, hi=10, knob="r_tempo")
        self.step = StepSequencer(
            SawSource,
            self.base_freq,
            self.speed,
            melody=[1, 4, 1, 7, 10, 11, 12],
            steps=[1, 1, 1, 1, 1, 1, 1],
            # Gate mode:
            # - H: hold for the step count,    XXXXXX
            # - E: Sound for each step count   X X X
            # - F: Sound for first step count  X
            # - S: Skip, no sound
            #gate='HSHSHS'
            gate='HHHHSHH'
        )

        self.step2 = self.step.copy(melody=[1, 1, 1],
                                    base_frequency=self.base_freq2,
                                    wave_generator_cls=SineSource,
                                    gate="HHH")

        self.step3 = self.step.copy(melody=[7, 4, 7],
                                    base_frequency=self.base_freq2)

        self.mixer = PlainMixer(self.step, self.step2)#, self.step3)
        self.lp = SimpleLowPass(self.mixer, window_size=Parameter(20, key="w"))
        self.r = Reverb(self.lp,
                        max_decay=Parameter(0.5, knob="fx2_1", lo=0, hi=0.8),
                        alpha=Parameter(1, knob="fx2_2", lo=0.0001, hi=10))
        self.out = self.r


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


