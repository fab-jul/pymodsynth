
import pytest
import numpy as np


from mz import helpers


def test_lru_dict():
    d = helpers.LRUDict(capacity=3)
    d[1] = "foo"
    d[2] = "bar"
    d[3] = "baz"
    # Fetch `1`, so `2` becomes least recently used.
    _ = d[1]
    d[4] = "bax"
    assert 2 not in d
    assert d[1] == "foo"
    assert d[3] == "baz"
    assert d[4] == "bax"


def test_array_buffer():
        
    buffer = helpers.ArrayBuffer(initial_capacity=3)
    with pytest.raises(ValueError):
        _ = buffer.get()

    buffer.push(np.array([1]))
    assert buffer.get().tolist() == [0, 0, 1]

    buffer.push(np.array([2]))
    assert buffer.get().tolist() == [0, 1, 2]

    buffer.push(np.array([3]))
    assert buffer.get().tolist() == [1, 2, 3]

    buffer.push(np.array([4]))
    assert buffer.get().tolist() == [2, 3, 4]

    # Make more space.
    buffer.set_capacity(5)
    assert buffer.get().tolist() == [0, 0, 2, 3, 4]
    buffer.push(np.array([5]))
    assert buffer.get().tolist() == [0, 2, 3, 4, 5]

    # Make less space, should keep last N elements.
    buffer.set_capacity(2)
    assert buffer.get().tolist() == [4, 5]

