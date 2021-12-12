import dataclasses
import enum
import operator
import functools
import collections
import typing
import scipy.signal
import contextlib
import numpy as np

from mz import helpers
from mz import midi_lib

from typing import Any, Callable, Iterable, Mapping, NamedTuple, Optional, Sequence, Set, Tuple, TypeVar, Union


class ClockSignal(NamedTuple):
    ts: np.ndarray
    sample_indices: np.ndarray
    sample_rate: float

    def assert_same_shape(self, a: np.ndarray, module_name: str):
        if a.shape != self.shape:
            raise ValueError(
                f"Error at `{module_name}`, output has shape {a.shape} != {self.shape}!")  

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


# This is extracted from flax, and makes sure BaseModule subclasses get
# auto-completion in IDEs. According to flax:
# > This decorator is interpreted by static analysis tools as a hint
#   that a decorator or metaclass causes dataclass-like behavior.
#   See https://github.com/microsoft/pyright/blob/main/specs/dataclass_transforms.md
#   for more information about the __dataclass_transform__ magic.
_T = TypeVar("_T")
def __dataclass_transform__(
    *,
    eq_default: bool = True,
    order_default: bool = False,
    kw_only_default: bool = False,
    field_descriptors: Tuple[Union[type, Callable[..., Any]], ...] = (()),
) -> Callable[[_T], _T]:
  # If used within a stub file, the following implementation can be
  # replaced with "...".
  return lambda a: a


@__dataclass_transform__()
class ModuleMeta(type):
    """Meta class for Modules.
    
    This is used to convert all modules to dataclasses.
    """
    def __new__(cls, name, bases, dct):
        cls = super().__new__(cls, name, bases, dct)
        cls = dataclasses.dataclass(
            cls, eq=True,
            # TODO: This kind of assumes modules are immutable, which they
            # are not really.
            unsafe_hash=True)
        return cls


class BaseModule(metaclass=ModuleMeta):
    """Root class for everything.

    Supports recursively finding parameters and state.

    Notes for subclasses
    --------------------

    Example:

      class Foo(BaseModule):
          var: int
          foo: Sequence[float] = dataclasses.field(default_factory=list)

          def setup(self):
              self._bar = self.var + 2

    Notes:
    - Subclasses should use dataclass attributes to specify instance variables.
      Like dataclasses, be careful with list efaults (use dataclasses.field).
    - Subclasses can overwrite `setup` to provide custom initialization code

    Functinality:
    # TODO
    """

    def __post_init__(self):
        # Will be used to raise exception if setattr is called after setup.
        self._locked = False

        # TODO: This should be State modules!
        self._frame_buffers = collections.defaultdict(helpers.ArrayBuffer)
        self._call_cache = helpers.LRUDict(capacity=16)

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

        # TODO: Test
        self._direct_submodules = [
            (name, a) for name, a in vars(self).items()
            if isinstance(a, BaseModule)]

        self._state_vars = [
            (name, a) for name, a in vars(self).items()
            if isinstance(a, State)]

        # After this, no more assignements allowed.
        # TODO: Use to ensure hash makes sense
        self._locked = self._should_auto_lock

    @property
    def _should_auto_lock(self):
        return True

    def unlock(self):
        self._locked = False

    @contextlib.contextmanager
    def unlocked(self):
        self._locked = False
        yield
        self._locked = self._should_auto_lock
        
    @property
    def name(self):
        return self.__class__.__name__

    def __setattr__(self, name: str, value: Any) -> None:
        # TODO
        if hasattr(self, "_locked") and self._locked and name != "_locked":
            raise ValueError(f"Cannot assign after setup, got setattr for `{name}`.")
        super().__setattr__(name, value)

    def setup(self):
        """Subclasses can overwrite this to run code after an instance was constructed."""

    # TODO: Could this be a decorator?
    def cached_call(self, key_prefix, fn):
        """Return fn() but cache result, only re-execute `fn` if the cache key changes."""
        try:
            # Always use the function name in the cache name.
            # Note that this means that collisions can occur
            # if callers use this function on the same function name.
            key = (fn.__name__, key_prefix, self.get_cache_key())
        except NoCacheKeyError:
            key = None

        if key and key in self._call_cache:
            return self._call_cache[key]

        result = fn()

        if key:
            self._call_cache[key] = result

        return result

    def prepend_past(self, key: str, current: np.ndarray, *, num_frames: int = 2) -> np.ndarray:
        """Call to transparently fetch previous versions of an array.
        
        Example usage, typically in a `Module` subclass:
        
        class SomeSubClass(Module):

            src: Module

            def out_with_inputs(self, clock_signal, src):
                # Now, `src_with_context` is a tensor of length 3 * num_samples,
                # containing the last 2 version of `src`, as well as the current, 
                # in that order
                src_with_context = self.prepend_past("src", current=src, num_frames=3)

        Args:
          key: Name of the tensor we fetch, used as a key for the frame buffer to use.
          current: Current version of the array, shape (S, C), where S = number of samples in a frame.
          num_frames: How many frames to get *in total*, i.e., including `current`.
            If this changes between calls, the underlying buffer is adapted.

        Returns:
          An array of shape (num_frames * S, C).
        """
        frame_buffer = self._frame_buffers[key]
        frame_buffer.set_capacity(num_frames)
        frame_buffer.push(current)
        return frame_buffer.get()

    def _get_cache_key_recursive(self) -> Sequence[Tuple[str, Any]]:
        output = []
        for name, module in self._direct_submodules:
            output += [(name + "." + key, value) 
                       # Go into recursion.
                       for key, value in module.get_cache_key()]
        if not output:
            raise NoCacheKeyError()
        return tuple(output)

    def get_cache_key(self) -> Sequence[Tuple[str, Any]]:
        """Get a key uniquely describing the state this module is in.
        
        If no such description is suitable, raise `NoCacheKeyError`.
        
        By default, we defer the cache key calculation to any immediate
        submodules, which in turn recursively do the same,
        until we land at leaf submodules, which *typically* are Constant
        or Parameter modules, which implement `get_cache_key` by returning their value.
        Thus, in practise, this function returns something like:

            [("current.child1.param1.value", 1),
             ("current.child1.param2.value", 2),
             ("current.child2.value", 3)]
        """
        # The default implementaiton of cache key defers to
        # any direct submodules, see `_get_cache_key_recursive`.
        return self._get_cache_key_recursive()

    # STATE: use explicit initial value, use find_submodules with all,
    # then get their state.

    def _iter_named_submodules(self,
                               memo: Optional[Set["BaseModule"]] = None,
                               prefix: str = "") -> Iterable[Tuple[str, "BaseModule"]]:
        # `memo` is used to make sure we count everything at most once.
        if memo is None:
            memo = set()
        if self not in memo:
            yield prefix, self
            for name, module in self._direct_submodules:
                submodule_prefix = prefix + ("." if prefix else "") + name
                for p, m in module._iter_named_submodules(memo, submodule_prefix):
                    yield p, m

    def find_submodules(self, cls=None) -> Mapping[str, "BaseModule"]:
        """Return all submodules, optionally filtering by `cls`."""
        if not cls:
            cls = BaseModule
        return {name: m for name, m in self._iter_named_submodules()
                if issubclass(m.__class__, cls)}

    def get_params_dict(self) -> Mapping[str, "BaseModule"]:
        return self.find_submodules(cls=Parameter)

    def reset_state(self):
        # TODO: Implement by inspecting default value of State params.
        pass

    def get_state_dict(self) -> Mapping[str, "BaseModule"]:
        # Get all modules, then ask them for their state.
        state_dict = {}
        for name, m in self._iter_named_submodules():
            for state_name, state in m._state_vars:
                state_dict[".".join((name, state_name))] = state
        return state_dict
        
    def set_state_from_dict(self, state_dict):
        _copy(state_dict, target=self.get_state_dict())

    def set_params_from_dict(self, params_dict):
        _copy(params_dict, target=self.get_params_dict())

    @contextlib.contextmanager
    def disable_state(self):
        state = self.get_state_dict()
        self.reset_state()
        yield
        self.set_state_from_dict(state)

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

    def out_single_value(self, clock_signal: ClockSignal) -> float:
        return np.mean(self.out(clock_signal))

    def out_given_inputs(self, clock_signal: ClockSignal, **inputs):
        raise NotImplementedError("Must be implemented by subclasses!")

    def __call__(self, clock_signal: ClockSignal):
        return self.out(clock_signal)

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


_GetAndSetModule = Union["Constant", "Parameter", "State"]


def _copy(src: Mapping[str, _GetAndSetModule], target: Mapping[str, _GetAndSetModule]):
    target = target.copy()
    for k, param in src.items():
        try:
            target[k].set(param.get())
        except KeyError as e:
            print(f"Parameter {k} disappered, caught: {e}")
        else:
            target.pop(k)
    if target:  # Some params were not set -> ignore.
        pass  # TOOD: Log.


_Operator = Callable[[np.ndarray, np.ndarray], np.ndarray]


class _MathModule(BaseModule):
    """Implement various mathematical operations on modules, see base class."""

    op: _Operator
    left: BaseModule
    right: BaseModule

    @property
    @functools.lru_cache()
    def both_are_modules(self) -> bool:
        return isinstance(self.left, Module) and isinstance(self.right, Module)

    @property
    def name(self):
        # TODO: test
        return f"Math(op={self.op.__name__}, left={self.left.name}, right={self.right.name})"

    def out(self, clock_signal: ClockSignal):
        left = self._maybe_call(self.left, clock_signal)
        right = self._maybe_call(self.right, clock_signal)
        result = self.op(left, right)
        if self.both_are_modules:
            clock_signal.assert_same_shape(result, self.name)
        return result

    @staticmethod
    def _maybe_call(module_or_number, clock_signal: ClockSignal):
        if isinstance(module_or_number, Module):
            return module_or_number(clock_signal)
        return module_or_number


class State:
    """Used to annotate state-ful parts of models."""

    def __init__(self, initial_value=None) -> None:
        self._value = initial_value
    
    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class Module(BaseModule):
    """Root class for sound modules.

    Same API as BaseModule except that we have a strict requirement that
    the shape of the array returned by `out` must be the same as the
    shape of `clock_signal`.
    """

    def __call__(self, clock_signal: ClockSignal):
        result = self.out(clock_signal)
        clock_signal.assert_same_shape(result, self.name)
        return result


# Use to annotate single values, TODO
SingleValueModule = TypeVar("SingleValueModule", bound=Module)


class Constant(Module):
    
    value: float

    def set(self, value):
        with self.unlocked():
            self.value = value

    def get(self):
        return self.value

    def get_cache_key(self) -> Sequence[Tuple[str, Any]]:
        # We provide a concrete get_cache_key implementation.
        # In practise, this will be the only implementation.
        return (("value", self.value),)

    def out(self, clock_signal: ClockSignal):
        return np.broadcast_to(self.value, clock_signal.shape)

    def out_single_value(self, _) -> float:
        return self.value


class Parameter(Constant):

    # Initial value
    value: float
    # Lowest sane value. Defaults to 0.1 * value.
    lo: Optional[float] = None
    # Highest sane value. Defaults to 1.9 * value.
    hi: Optional[float] = None
    # If given, a key on the keyboard that controls this parameter. Example: "f".
    key: Optional[str] = None
    # If given, a knob on a Midi controller that controls this parameter.
    knob: Optional[midi_lib.KnobConvertible] = None
    # Only used if `key` is set, in which case this sets how much
    #   more we change the parameter if SHIFT is pressed on the keyboard.
    shift_multiplier: float = 10
    # If True, clip to [lo, hi] in `set`.
    clip: bool = False

    def setup(self):
        if self.lo is None:
            self.lo = 0.1 * self.value
        if self.hi is None:
            self.hi = 1.9 * self.value
        if self.hi < self.lo:
            raise ValueError
        self.lo, self.hi = self.lo, self.hi
        self.span = self.hi - self.lo

    def set(self, value):
        if self.clip:
            self.value = np.clip(value, self.lo, self.hi)
        else:
            self.value = value

    def set_relative(self, rel_value: float):
        """Set with value in [0, 1], and we map to [lo, hi]."""
        self.set(self.lo + self.span * rel_value)

    def inc(self, diff):
        """Increment value by `diff`."""
        self.set(self.value + diff)

    @property
    def _should_auto_lock(self):
        # Parameter should remain unlocked for easy value changes.
        return False


class FreqFactors(enum.Enum):
    OCTAVE = 2.
    STEP = 1.059463
