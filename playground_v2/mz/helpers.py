import collections
from typing import TypeVar
import numpy as np
from mz import base


def cluster_mods(mods):
    """Take flat ordered list of modules with optional cluster names.
    Modules with empty cluster names become singleton lists.
    Modules with same cluster names go into same list, keeping the order."""
    empty_counter = 0
    cluster_dict = collections.defaultdict(list)
    for mod in mods:
        if mod.cluster_name == "":
            cluster_dict["#" + str(empty_counter)] = [mod]
            empty_counter += 1
            continue
        cluster_dict[mod.cluster_name].append(mod)
    return cluster_dict

def plot_module(mod_cls, start_frame=0, num_frames=5, cluster=True):
    import matplotlib.pyplot as plt
    base.Collect.buffer_size = num_frames
    mod = mod_cls()
    mod_out = []
    clock = base.Clock(num_samples=2048, sample_rate=44100)
    for i in range(start_frame + num_frames):
        clock_signal = clock()
        mod_out.append(mod(clock_signal))
    mod_out = mod_out[start_frame:]
    # data is in the Collect modules and mod_out now
    mods = list(mod._filter_submodules_by_cls(base.Collect).values())
    mods.append(mod)
    setattr(mod, "data", mod_out)
    setattr(mod, "name_coll", "output")
    setattr(mod, "cluster_name", "")
    setattr(mod, "coll_num", len(mods) - 1)
    mods = list(sorted(mods, key=lambda m: m.coll_num))
    print("Plotting modules ", [m.name_coll for m in mods])
    if not cluster:
        fig, axes = plt.subplots(len(mods), 1)
        for ax, module in zip(axes, mods):
            data = np.concatenate(module.data)
            ax.plot(data, label=module.name_coll)
            ax.legend()
    else:
        mod_clusters = cluster_mods(mods)
        print("clusters", mod_clusters)
        fig, axes = plt.subplots(len(mod_clusters), 1)
        for ax, (cluster_name, cluster) in zip(axes, mod_clusters.items()):
            if cluster_name[0] == "#":
                cluster_name = ""
            for module in cluster:
                data = np.concatenate(module.data)
                ax.plot(data, label=module.name_coll)
            ax.set_title(cluster_name)
            ax.legend()
    plt.show()


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
