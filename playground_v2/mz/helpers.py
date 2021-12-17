import collections
from typing import TypeVar
import numpy as np


def is_class_and_subclass(cls, t):
    """Safe version of `issubclass` that returns False if `cls` is not an actual class."""
    try:
        return issubclass(cls, t)
    except TypeError:
        return False


class LRUDict:

    def __init__(self, capacity: int = 64) -> None:
        self._capacity = capacity
        self._cache = {}
        self._keys_access_counts = {}
        self._counter = 0

    def __getitem__(self, key):
        if key not in self._cache:
            raise KeyError(key)
        self._keys_access_counts[key] = self._counter
        self._counter += 1
        return self._cache[key]

    def __contains__(self, key):
        return key in self._cache

    def keys(self):
        return self._cache.keys()

    def get(self, key, default=None):
        if key not in self._cache:
            return default
        return self.__getitem__(key)

    def __setitem__(self, key, value):
        if len(self._cache) == self._capacity:
            # Evict oldest.
            oldest_key = min(self._keys_access_counts.keys(), key=self._keys_access_counts.get)
            self._cache.pop(oldest_key)
            self._keys_access_counts.pop(oldest_key)
        self._cache[key] = value
        self._keys_access_counts[key] = self._counter
        self._counter += 1


class ArrayBuffer:

    def __init__(self, initial_capacity: int = 1) -> None:
        #self._capacity = initial_capacity
        self._buffer = collections.deque(maxlen=initial_capacity)

    def set_capacity(self, capacity: int):
        if capacity == self._buffer.maxlen:
            return
        elements = list(self._buffer)[-capacity:]
        self._buffer = collections.deque(elements, maxlen=capacity)

    def push(self, current: np.ndarray):
        self._buffer.append(current)

    def get(self):
        """Return concatenated buffer contents."""
        return np.concatenate(list(self.__iter__()), axis=0)

    def __iter__(self):
        if not self._buffer:
            raise ValueError("Empty ArrayBuffer, cannot iterate!")
        last = self._buffer[-1]
        for _ in range(self._buffer.maxlen - len(self._buffer)):
            yield np.zeros_like(last)
        yield from self._buffer
        

# Test Helpers------------------------------------------------------------ 


_MARKED_CLASSES = []


class MarkedClass:

    def __init__(self, cls, **kwargs):
        self.cls = cls
        self.kwargs = kwargs

    @property
    def name(self):
        return self.cls.__name__

    def get_instance(self):
        try:
            return self.cls(
                **{name: value() for name, value in self.kwargs.items()})
        except TypeError as e:
            raise TypeError(f"Cannot make {self.cls.__name__} with kwargs={self.kwargs}") from e


def mark_for_testing(**kwargs):
    """Mark module for testing."""
    def decorator(cls):
        _MARKED_CLASSES.append(MarkedClass(cls, **kwargs))
        return cls
    return decorator


def iter_marked_classes():
    yield from (marked_class for marked_class in _MARKED_CLASSES)
