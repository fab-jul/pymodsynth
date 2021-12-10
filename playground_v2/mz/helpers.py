import collections


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
