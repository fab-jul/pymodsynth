
_MARKED_CLASSES = []


class MarkedClass:

    def __init__(self, cls, **kwargs):
        self.cls = cls
        self.kwargs = kwargs

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
    yield from (marked_class.get_instance()
                for marked_class in _MARKED_CLASSES)
