
import numpy as np
import dataclasses
from typing import List
from numpy import exp
import pytest

from unittest import mock

from mz import base
from mz.base import BaseModule


class Node(base.BaseModule):
    src: base.Module
    trg: base.BaseModule


class NodeModule(base.Module):
    src: base.Module
    trg: base.BaseModule

    def out(self, clock_signal):
        return clock_signal.zeros()


def test_cached_sampling():
    clock = base.Clock()
    root = NodeModule(base.Constant(1), base.Constant(2))
    root.unlock()  # To overwrite out
    assert root is not None

    out_mock = mock.MagicMock()
    out_mock.side_effect = root.out
    root.out = out_mock

    root.sample(clock, num_samples=10)
    root.sample(clock, num_samples=10)
    root.sample(clock, num_samples=10)

    assert list(root._call_cache.keys()) == [
        ("_sample", (repr(clock), 10), root.get_cache_key())]
    assert out_mock.call_count == 1
    
    root.sample(clock, num_samples=11)
    assert out_mock.call_count == 2
    assert list(root._call_cache.keys()) == [
        ("_sample", (repr(clock), 10), root.get_cache_key()),
        ("_sample", (repr(clock), 11), root.get_cache_key())]


def test_direct_submodules():
    trg = Node(base.Constant(2), base.Constant(3))
    root = Node(src=base.Constant(1), trg=trg)
    actual = tuple(root._direct_submodules)
    expected = (("src", base.Constant(1)),
                ("trg", trg))
    assert actual == expected

def test_cache_key():
    root = Node(
        src=Node(
            src=Node(src=base.Constant(1), 
                     trg=Node(base.Constant(2), base.Constant(3))),
            trg=base.Constant(4)
        ),
        trg=Node(base.Constant(5), base.Constant(6))
    )

    expected_cache_key = (
        ("src.src.src.value", 1),
        ("src.src.trg.src.value", 2),
        ("src.src.trg.trg.value", 3),
        ("src.trg.value", 4),
        ("trg.src.value", 5),
        ("trg.trg.value", 6),
    )

    assert root.get_cache_key() == expected_cache_key


def test_cache_key_raises():

    class NoCacheKeyModule(base.Module):
        pass

    root = Node(src=base.Constant(1), trg=NoCacheKeyModule())

    with pytest.raises(base.NoCacheKeyError):
        root.get_cache_key()


def test_is_subclass():

    class M(base.Module):
        pass

    assert base.safe_is_subclass(base.Constant, base.BaseModule)
    assert base.safe_is_subclass(M, base.BaseModule)

    class N(base.Module):
        src: base.Module

    n = N(src=base.Constant(2.))
    assert n._other_modules == ["src"]


def test_prepend_past():

    class M(base.Module):

        src: base.Module
        _src_prepended: List[np.ndarray] = dataclasses.field(default_factory=list)

        def out_given_inputs(self, clock_signal, src):
            self._src_prepended.append(
                self.prepend_past("src", src, num_frames=3))
            return src

    class ArangeSource(base.Module):

        def out(self, clock_signal: base.ClockSignal):
            return clock_signal.sample_indices
            
    m = M(src=ArangeSource())
    clock = base.Clock(num_samples=4, num_channels=2, sample_rate=1)

    m(clock())
    m(clock())
    m(clock())
    m(clock())

    clock_2 = base.Clock(num_samples=4, num_channels=2, sample_rate=1)
    indices_0 = clock_2().sample_indices
    indices_1 = clock_2().sample_indices
    indices_2 = clock_2().sample_indices
    indices_3 = clock_2().sample_indices
    zeros = np.zeros_like(indices_0)

    expected_src_prepended = [
        np.concatenate([zeros,     zeros,     indices_0], axis=0),
        np.concatenate([zeros,     indices_0, indices_1], axis=0),
        np.concatenate([indices_0, indices_1, indices_2], axis=0),
        np.concatenate([indices_1, indices_2, indices_3], axis=0),
    ]

    np.testing.assert_array_equal(m._src_prepended, expected_src_prepended)


def test_named_submodules():

    class Leaf(base.BaseModule):
        v: int

    root = Node(
        src=Node(
            src=Node(src=Leaf(1), trg=Node(Leaf(2), Leaf(3))),
            trg=Leaf(4)
        ),
        trg=Node(Leaf(5), Leaf(6))
    )

    expected_submodules = {
        "src.src.src": Leaf(1),
        "src.src.trg.src": Leaf(2),
        "src.src.trg.trg": Leaf(3),
        "src.trg": Leaf(4),
        "trg.src": Leaf(5),
        "trg.trg": Leaf(6),
    }

    actual_submodules = root.find_submodules(cls=Leaf)
    assert expected_submodules == actual_submodules


def test_state():

    class Leaf(base.BaseModule):
        v: int
        
        def setup(self):
            self._state_a = base.State(initial_value=self.v)
            self._state_b = base.State(initial_value=self.v*10)

    root = Node(
        src=Node(
            src=Node(src=Leaf(1), trg=Node(Leaf(2), Leaf(3))),
            trg=Leaf(4)
        ),
        trg=Node(Leaf(5), Leaf(6))
    )

    expected_state = {
        "src.src.src._state_a": 1,
        "src.src.trg.src._state_a": 2,
        "src.src.trg.trg._state_a": 3,
        "src.trg._state_a": 4,
        "trg.src._state_a": 5,
        "trg.trg._state_a": 6,

        "src.src.src._state_b": 10,
        "src.src.trg.src._state_b": 20,
        "src.src.trg.trg._state_b": 30,
        "src.trg._state_b": 40,
        "trg.src._state_b": 50,
        "trg.trg._state_b": 60,
    }

    actual_state = {k: s._value for k, s in root.get_state_dict().items()}
    assert expected_state == actual_state


def test_copy_state_and_params():

    class Leaf(base.BaseModule):
        v: int

        def setup(self):
            self.state = base.State(self.v)
            self.param = base.Parameter(self.v * 10)

    module_a = Node(
        src=Node(
            src=Node(src=Leaf(1), trg=Node(Leaf(2), Leaf(3))),
            trg=Leaf(4)
        ),
        trg=Node(Leaf(5), Leaf(6))
    )

    module_b = Node(
        src=Node(
            src=Node(src=Leaf(-1), trg=Node(Leaf(-1), Leaf(-1))),
            trg=Leaf(-1)
        ),
        trg=Node(Leaf(-1), Leaf(-1))
    )

    module_b.set_state_from_dict(module_a.get_state_dict())
    module_b.set_params_from_dict(module_a.get_params_dict())

    expected_state_values = {
        "src.src.src.state": 1,
        "src.src.trg.src.state": 2,
        "src.src.trg.trg.state": 3,
        "src.trg.state": 4,
        "trg.src.state": 5,
        "trg.trg.state": 6,
    }

    expected_param_values = {
        "src.src.src.param": 10,
        "src.src.trg.src.param": 20,
        "src.src.trg.trg.param": 30,
        "src.trg.param": 40,
        "trg.src.param": 50,
        "trg.trg.param": 60,
    }

    actual_state_values = {k: v.get() for k, v in module_b.get_state_dict().items()}
    actual_param_values = {k: v.get() for k, v in module_b.get_params_dict().items()}
    assert actual_state_values == expected_state_values
    assert actual_param_values == expected_param_values



def _assert_outputs_similar(clock_signal, module_a, module_b):
    out_a = module_a(clock_signal)
    out_b = module_b(clock_signal)
    np.testing.assert_allclose(out_a, out_b)


@pytest.fixture(params=[128, 2048, 1234], ids=lambda s: f"num_samples={s}")
def num_samples(request):
    return request.param


@pytest.fixture(params=[1, 2], ids=lambda c: f"num_channels={c}")
def num_channels(request):
    return request.param


@pytest.fixture()
def clock_signal(num_samples, num_channels):
    clock = base.Clock(num_samples=num_samples, num_channels=num_channels)
    clock_signal = clock()
    assert clock_signal.shape == (num_samples, num_channels)
    return clock_signal


def test_math(clock_signal):
    const_ten = base.Constant(10.)
    const_five = base.Constant(5.)

    # Test using a scalar and a module.
    for other in (5., const_five):
        _assert_outputs_similar(clock_signal, const_ten * other, base.Constant(50.))
        _assert_outputs_similar(clock_signal, const_ten / other, base.Constant(2.))
        _assert_outputs_similar(clock_signal, const_ten + other, base.Constant(15.))
        _assert_outputs_similar(clock_signal, const_ten - other, base.Constant(5.))

        _assert_outputs_similar(clock_signal, other * const_ten, base.Constant(50.))
        _assert_outputs_similar(clock_signal, other / const_ten, base.Constant(0.5))
        _assert_outputs_similar(clock_signal, other + const_ten, base.Constant(15.))
        _assert_outputs_similar(clock_signal, other - const_ten, base.Constant(-5.))


def test_math_name(clock_signal):
    const_ten = base.Constant(10.)
    const_five = base.Constant(5.)
    result = const_ten + const_five
    assert result.name == "Math(op=add, left=Constant, right=Constant)"