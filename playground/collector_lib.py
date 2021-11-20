from typing import Sequence, Tuple, Any, Iterable


class _ShiftCollector:

    def __init__(self, name):
        self.name = name
        self.cache = []

    def __lshift__(self, other):
        self.cache.append(other)
        return other

    def __repr__(self):
        return f"_ShiftCollect({self.name}: {self.cache})"


class Collector:

    def __init__(self):
        self.cache = {}

    def __call__(self, name) -> _ShiftCollector:
        return self.cache.setdefault(name, _ShiftCollector(name))

    def __iter__(self) -> Iterable[Tuple[str, Sequence[Any]]]:
        for name, shift_collector in self.cache.items():
            yield name, shift_collector.cache[:]

    def __repr__(self):
        return f"Collector({self.cache})"

    def __len__(self):
        return len(self.cache)


class FakeCollector(Collector):
    """A collector that raises an exception when it's used"""

    def __call__(self, _):
        raise ValueError("Collector not active!")


def test():
    collect = Collector()

    bar = collect("test") << "foo"
    assert bar == "foo"

    for _ in range(10):
        collect("test") << "foo"

    print(collect)


if __name__ == '__main__':
    test()
