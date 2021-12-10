import dataclasses
import functools
import collections
import scipy.signal
import contextlib
import numpy as np

from mz import helpers

from typing import Mapping, NamedTuple, Optional, Set, TypeVar


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


class NoCacheKeyError(Exception):
    """Used to indicate that a module has no cache key."""


# TODO: This may actually not be needed if we only hot-reload a user file...!
def safe_is_subclass(cls, base_cls):
    try:
        all_base_classes = cls.__mro__
    except AttributeError:
        return False
    for c in all_base_classes:
        if c.__name__ == base_cls.__name__:
            return True
    return False


class ModuleMeta(type):
    """Meta class for Modules.
    
    This is used to convert all modules to dataclasses.
    """
    def __new__(cls, name, bases, dct):
        cls = super().__new__(cls, name, bases, dct)
        cls = dataclasses.dataclass(cls, eq=True)
        return cls


class BaseModule(metaclass=ModuleMeta):
    """Root class for everything.

    Supports recursively finding parameters and state.
    """

    def __post_init__(self):
        self._modules = {k: v for k, v in vars(self).items() if isinstance(v, BaseModule)}
        self._frame_buffers = collections.defaultdict(helpers.ArrayBuffer)
        self._sample_cache = helpers.LRUDict(capacity=16)

        single_value_modules = []
        other_modules = []
        for field in dataclasses.fields(self):
            if field.type == SingleValueModule:  # TODO: won't work with reloading
                single_value_modules.append(field.name)
            elif safe_is_subclass(field.type, BaseModule):
                other_modules.append(field.name)
        self._single_value_modules = single_value_modules
        self._other_modules = other_modules

        self.setup()

    def setup(self):
        """Subclasses can overwrite this to run code after an instance was constructed."""

    # TODO: Could this be a decorator?
    def cached_call(self, key_prefix, fn):
        try:
            key = (key_prefix, self.get_cache_key())
        except NoCacheKeyError:
            key = None

        if key and key in self._sample_cache:
            return self._sample_cache[key]

        result = fn()

        if key:
            self._sample_cache[key] = result

        return result

    def _iter_direct_submodules(self):
        for name, a in vars(self).items():
            if safe_is_subclass(a.__class__, BaseModule):
                yield name, a

    def prepend_past(self, key: str, current: np.ndarray, *, num_frames: int = 2) -> np.ndarray:
        frame_buffer = self._frame_buffers[key]
        frame_buffer.set_capacity(num_frames)
        frame_buffer.push(current)
        return frame_buffer.get()

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


class Module(BaseModule):
    """Root class for sound modules.

    API contract: TODO
    """

    def sample(self, clock: Clock, num_samples: int):
        def _sample():
            print("Sampling...")
            with self.disable_state():
                clock_signal = clock.get_clock_signal_num_samples(num_samples)
                result = self.out(clock_signal)

        return self.cached_call(key_prefix=(repr(clock), num_samples), fn=_sample)

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
        raise NotImplementedError("Must be implemented by subclasses!")


# Use to annotate single values, TODO
SingleValueModule = TypeVar("SingleValueModule", bound=Module)


class Constant(Module):
    
    value: float

    def get_cache_key(self):
        # We provide a concrete get_cache_key implementation.
        # In practise, this will be the only implementation.
        return (("value", self.value),)

    def out(self, clock_signal: ClockSignal):
        return np.broadcast_to(self.value, clock_signal.shape)

    def out_single_value(self, _) -> float:
        return self.value