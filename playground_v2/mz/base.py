import dataclasses
import functools
import collections
import contextlib
import numpy as np

from mz import helpers

from typing import Mapping, NamedTuple, Optional, Set, TypeVar


moduleclass = functools.partial(dataclasses.dataclass, eq=True)


class ClockSignal(NamedTuple):
    ts: np.ndarray
    sample_indices: np.ndarray
    sample_rate: float

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

    def __init__(self, num_samples: int = 2048, num_channels: int = 2, sample_rate: int = 44100.):
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.i = 0

        self.arange = np.arange(self.num_samples, dtype=int)  # Cache it.

    def __repr__(self) -> str:
        return f"Clock({self.num_samples}, {self.num_channels}, {self.sample_rate})"

    def __call__(self) -> ClockSignal:
        """Gets clock signal and increments i."""
        clock_signal = self.get_current_clock_signal()
        self.i += self.num_samples
        return clock_signal

    def get_current_clock_signal(self):
        sample_indices = self.i + self.arange
        return self.get_clock_signal(sample_indices)

    def get_clock_signal(self, sample_indices):
        ts = sample_indices / self.sample_rate
        # Broadcast `ts` into (num_samples, num_channels)
        ts = ts[..., np.newaxis] * np.ones((self.num_channels,))
        return ClockSignal(ts, sample_indices, self.sample_rate)

    def get_clock_signal_num_samples(self, num_samples: int):
        return self.get_clock_signal(np.arange(num_samples, dtype=int))


class ModuleMeta(type):
    def __new__(cls, name, bases, dct):
        return super().__new__(cls, name, bases, dct)


class NoCacheKeyError(Exception):
    """Used to indicate that a module has no cache key."""


@moduleclass
class BaseModule:#(metaclass=ModuleMeta):
    """Root class for everything.

    Supports recursively finding parameters and state.
    """

    @classmethod
    def _is_subclass(cls, instance):
        try:
            all_base_classes = instance.__class__.__mro__
        except AttributeError:
            return False
        for c in all_base_classes:
            if c.__name__ == cls.__name__:
                return True
        return False

    def __post_init__(self):
        self._modules = {k: v for k, v in vars(self).items() if isinstance(v, BaseModule)}

    def _iter_direct_submodules(self):
        for name, a in vars(self).items():
            if BaseModule._is_subclass(a):
                yield name, a

    def _get_cache_key_recursive(self):
        output = []
        for name, module in self._iter_direct_submodules():
            output += [(name + "." + key, value) for key, value in module.get_cache_key()]
        if not output:
            raise NoCacheKeyError()
        return tuple(output)

    def get_cache_key(self):
        return self._get_cache_key_recursive()

    # TODO: This is broken
    def _named_submodules(self,
                          memo: Optional[Set["BaseModule"]] = None,
                          prefix: str = ""):
        # `memo` is used to make sure we count everything at most once.
        if memo is None:
            memo = set()
        if self not in memo:
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def named_submodules(self) -> Mapping[str, "BaseModule"]:
        return {name: m for name, m in self._named_submodules()}

    def _get_params_non_recursive(self):
        pass

    def _get_cache_state(self):
        raise NoCacheKeyError()

    def reset_state(self):
        pass

    def get_state(self):
        pass

    def set_state(self, state):
        pass

    @contextlib.contextmanager
    def disable_state(self):
        state = self.get_state()
        self.reset_state()
        yield
        self.set_state(state)


class State:

    def __init__(self, initial_value=None) -> None:
        self._value = initial_value
    
    def get(self):
        return self._value

    def set(self, value):
        self._value = value


@moduleclass
class Module(BaseModule):
    """Root class for sound modules.

    API contract:
    """

    def __post_init__(self):
        super().__post_init__()
        single_value_modules = []
        other_modules = []
        for field in dataclasses.fields(self):
            if field.type == SingleValueModule:  # TODO: won't work with reloading
                single_value_modules.append(field.name)

#            elif issubclass(field.type, _BufferedModule):
#                buffer_modules.append(field.name)

            elif issubclass(field.type, Module):
                other_modules.append(field.name)
        self._single_value_modules = single_value_modules
        self._other_modules = other_modules

        self._sample_cache = helpers.LRUDict(capacity=16)

    def sample(self, clock: Clock, num_samples: int):
        try:
            key = (repr(clock), num_samples, self.get_cache_key())
        except NoCacheKeyError:
            key = None

        if key and key in self._sample_cache:
            return self._sample_cache[key]

        print("Sampling...")

        with self.disable_state():
            clock_signal = clock.get_clock_signal_num_samples(num_samples)
            result = self.out(clock_signal)

        if key:
            self._sample_cache[key] = result

        return result

    def out(self, clock_signal: ClockSignal):
        inputs = {}
        for name in self._single_value_modules:
            inputs[name] = getattr(self, name).out_single_value(clock_signal)
        for name in self._other_modules:
            inputs[name] = getattr(self, name).out(clock_signal)
        return self.out_given_inputs(clock_signal, **inputs)

    def __call__(self, clock_signal: ClockSignal):
        return self.out(clock_signal)

    def out_single_value(self, clock_signal: ClockSignal) -> float:
        return np.mean(self.out(clock_signal))

    def out_given_inputs(self, clock_signal: ClockSignal, **inputs):
        pass


class Foo(Module):

    def out(self, clock_signal):
        decy = self.decay.out_single_value()


@moduleclass
class Constant(Module):
    
    value: float

    def get_cache_key(self):
        return (("value", self.value),)

    def out(self, clock_signal: ClockSignal):
        return np.broadcast_to(self.value, clock_signal.shape)

    def out_single_value(self, _) -> float:
        return self.value


SingleValueModule = TypeVar("SingleValueModule", bound=Module)


@moduleclass
class SineSource(Module):
    frequency: Module = Constant(440.)
    amplitude: Module = Constant(1.0)
    phase: Module = Constant(0.0)
    _last_cumsum_value: State = State(0.)

    def out_given_inputs(self, clock_signal: ClockSignal, frequency: np.ndarray, amplitude: np.ndarray, phase: np.ndarray):
        dt = np.mean(clock_signal.ts[1:] - clock_signal.ts[:-1])
        cumsum = np.cumsum(frequency * dt, axis=0) + self._last_cumsum_value.get()
        self._last_cumsum_value.set(cumsum[-1, :])
        out = amplitude * np.sin((2 * np.pi * cumsum) + phase)
        return out


class _BufferedModule:
    pass


@functools.lru_cache()
def BufferedModule(t):
    return type(f"BufferedModule{t}", (_BufferedModule,), __dict={"t": t})

#BufferedModule(2)

@moduleclass
class Reverb(Module):

    src: Module
    delay: SingleValueModule = Constant(3000)
    echo: SingleValueModule = Constant(10000)
    p: SingleValueModule = Constant(0.05)

    def out_given_inputs(self,
                         clock_signal: ClockSignal,
                         src: np.ndarray,
                         delay: float,
                         echo: float,
                         p: float):
        return src * echo


class Foo(Module):

    bar: Module

    def out_given_inputs(self, clock_signal, bar):
        pass
    


def _foo():
    r = Reverb(SineSource())
    clock = Clock()
    clock_signal = clock()
    print(r(clock_signal))
