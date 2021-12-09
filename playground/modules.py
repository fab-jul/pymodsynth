
import collections
import dataclasses
import enum
import functools
import itertools
import operator
import random
import re
import typing
from typing import (
    Any,
    Mapping,
    Union,
    MutableMapping,
    Optional,
    TypeVar,
    Callable,
    NamedTuple,
    Type,
    Tuple,
    Sequence,
)

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal

from playground import collector_lib
from playground import midi_lib
from playground import plot_lib
from playground.tests import helper as tests_helper

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
        """Gets clock signal and increments i."""
        clock_signal = self.get_current_clock_signal()
        self.i += self.num_samples
        return clock_signal

    def get_current_clock_signal(self):
        sample_indices = self.i + self.arange
        ts = sample_indices / self.sample_rate
        # Broadcast `ts` into (num_samples, num_channels)
        ts = ts[..., np.newaxis] * np.ones((self.num_channels,))
        return ClockSignal(ts, sample_indices, self.sample_rate)

    def get_until(self, num_samples):
        sample_indices = np.arange(num_samples, dtype=int)
        ts = sample_indices / self.sample_rate
        ts = ts[..., np.newaxis] * np.ones((self.num_channels,))
        return ClockSignal(ts, sample_indices, self.sample_rate)


class State:

    __special_name__ = "State"

    def __init__(self, *, initial_value=None):
        self._value = initial_value

    @property
    def is_set(self) -> bool:
        return self._value is not None

    def get(self):
        return self._value

    def set(self, value):
        self._value = value



T = TypeVar("T")


def _is_module_subclass(instance):
    try:
        all_base_classes = instance.__class__.__mro__
    except AttributeError:
        return False
    for c in all_base_classes:
        if c.__name__ == Module.__name__:
            return True
    return False


class Module:
    """
    Module :: Signal -> Signal, in particular:
    Module :: [Sampling_Times] -> [Samples]

    Modules can be called on a nparray of sampling times,
    and calculate an output of the same size according to
    the module graph defined in its constructor.
    A subclass should overwrite self.out, respecting its signature.
    """
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

    def out_mean_int(self, clock_signal: ClockSignal) -> int:
        return round(np.mean(self.out(clock_signal)))

    def out_mean_float(self, clock_signal: ClockSignal) -> float:
        return np.mean(self.out(clock_signal))

    def out(self, clock_signal: ClockSignal) -> np.ndarray:
        raise Exception("not implemented")

    def __rtruediv__(self, other):
        return _MathModule(operator.truediv, other, self)

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
        out = self.out(clock_signal)
        if "TriggerSource" not in str(type(self)) and out.shape != clock_signal.shape:  # yes, a hack! because I violate the API in TriggerSource!
            raise ValueError(f"Failure at {self.__class__.__name__}. "
                             f"Shapes are {out.shape} vs. {clock_signal.shape}.")
        return out

    def get_params_by_name(self) -> MutableMapping[str, "Parameter"]:
        return self._get(Parameter)

    def get_states_by_name(self) -> MutableMapping[str, "State"]:
        return self._get(State)

    def _get(self, cls: Type[T], prefix="", include_root=True) -> MutableMapping[str, T]:
        """Recursively find all instances of `cls`."""
        result = {}
        if prefix == "" and include_root and isinstance(self, cls):
            result[""] = self
        for var_name, var_instance in vars(self).items():
            if getattr(var_instance, "__special_name__", None) == cls.__special_name__:
                full_name = f"{prefix}{var_name}"
                var_instance.name = full_name
                result[full_name] = var_instance
                continue
            # If it's a Module, we go into the recursion.
            if _is_module_subclass(var_instance):
                result.update(var_instance._get(cls, prefix=f"{var_name}."))
        return result

    def _set(self, var_name: str, factory: Callable[[], T]) -> Mapping[str, T]:
        """Recursively set `var_name` to `factory()` for all submodules."""
        modules = self._get(Module, include_root=True)
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
        _copy(src=src_params, target=self.get_params_by_name())
        _copy(src=src_state, target=self.get_states_by_name())

    def sample(self, num_samples: int):
        clock_signal = Clock.get_until(num_samples)
        res = self.__call__(clock_signal)
        return res


class Id(Module):
    """Every Monoid needs a neutral element ;)"""
    def __init__(self, inp: Module):
        self.out = inp


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

    __special_name__ = "Parameter"

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


P = Parameter

@tests_helper.mark_for_testing()
class Random(Module):
    """Output a constant random amplitude until a random change event changes the amplitude. Best explanation ever."""
    def __init__(self, max_amplitude: float = 1.0, change_chance: float = 0.5):
        super().__init__()
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
                 amplitude=Constant(1.0),
                 phase=Constant(0.0)):
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


@tests_helper.mark_for_testing()
class SawSource(Module):
    def __init__(self, frequency: Module = Constant(440.), amplitude=Constant(1.0), phase=Constant(0.0)):
        super().__init__()
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
    def __init__(self, frequency: Module = Constant(440.), amplitude=Constant(1.0), phase=Constant(0.0)):
        super().__init__()
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
    def __init__(self, inp: Module, carrier_frequency: Module, inner_amplitude=Constant(1.0)):
        super().__init__()
        self.carrier = SineSource(carrier_frequency, amplitude=inner_amplitude)
        self.inp = inp
        # self.out = MultiplierModule(self.postprocessor, inp) # TODO: consider multiplier module for nice composition

    def out(self, clock_signal: ClockSignal):
        out = self.carrier(clock_signal) * self.inp(clock_signal)
        return out


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


@tests_helper.mark_for_testing(inp=SineSource)
class ButterworthFilter(Module):
    """Simplest lowpass: average over previous <window> values"""
    def __init__(self, inp: Module,
                 f_low: Module = Constant(10.),
                 f_high: Module = Constant(100.),
                 mode: str = "hp", order: int = 10):
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
        filtered_signal = filtered_signal * np.ones(num_channels)

        return filtered_signal[-num_samples:, :]


@functools.lru_cache(maxsize=128)
def get_me_some_butter(order, fs, mode):
    print("MAKING BUTTER")
    return signal.butter(order, fs, mode, fs=SAMPLING_FREQUENCY, output='sos')



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

  
@tests_helper.mark_for_testing()
class BabiesFirstSynthie(Module):
    def __init__(self):
        self.lfo = SineSource(Parameter(1))
        self.sin1 = SineSource(frequency=Parameter(440))

        self.out = self.sin1


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

