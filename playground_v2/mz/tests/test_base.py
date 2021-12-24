
import numpy as np
import dataclasses
from typing import List
from numpy import exp, sign
import pytest

from unittest import mock

from mz import base
from mz.base import BaseModule


class Node(base.BaseModule):
    src: base.Module
    trg: base.BaseModule

    def out_given_inputs(self, clock_signal: base.ClockSignal, src, trg):
        return src, trg

class NodeModule(base.Module):
    src: base.Module
    trg: base.BaseModule

    def out(self, clock_signal):
        return clock_signal.zeros()


def test_clock_signal():
    num_samples = 128
    c = base.ClockSignal(np.linspace(0, 1, num_samples),
                         np.arange(num_samples, dtype=base.SAMPLE_INDICES_DTYPE),
                         sample_rate=44100,
                         clock=base.Clock(num_samples, 44100))
    assert c.num_samples == num_samples
    c.assert_same_shape(np.ones(num_samples,), module_name="Testing")
    with pytest.raises(ValueError):
        c.assert_same_shape(np.ones(num_samples + 1,), module_name="Testing")
    assert c.zeros().shape == (num_samples,)
    assert c.ones().shape == (num_samples,)

    # Make c shorter.
    new_c = c.change_length(64)
    assert new_c.num_samples == 64
    assert new_c.sample_indices[0] == c.sample_indices[0]

    # Make c longer.
    new_c = c.change_length(192)
    assert new_c.num_samples == 192
    assert new_c.sample_indices[0] == c.sample_indices[0]

def test_clock_signal_pad_or_truncate():
    num_samples = 128
    c = base.ClockSignal(np.linspace(0, 1, num_samples),
                         np.arange(num_samples, dtype=base.SAMPLE_INDICES_DTYPE),
                         sample_rate=44100,
                         clock=base.Clock(num_samples, 44100))

    # Same length
    signal = np.arange(128)
    np.testing.assert_equal(signal, c.pad_or_truncate(signal))

    # Truncate
    signal = np.arange(192)
    np.testing.assert_equal(signal[:128], c.pad_or_truncate(signal))

    # Pad
    signal = np.arange(64)
    output = c.pad_or_truncate(signal)
    np.testing.assert_equal(output[:64], signal)
    np.testing.assert_equal(output[64:], 0)


def test_cached_sampling():
    raise pytest.skip("Need to revisit sampling.")
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


def test_cacheable_base_module():

    class M(base.CacheableBaseModule):

        def out(self, clock_signal):
            return clock_signal.zeros()

    m = M()
    out_mock = mock.MagicMock()
    out_mock.side_effect = m.out
    m.out = out_mock

    signal = base.ClockSignal.test_signal()
    m.cached_out(signal)
    assert out_mock.call_count == 1
    m.cached_out(signal)
    assert out_mock.call_count == 1
    m.cached_out(signal)
    assert out_mock.call_count == 1

    signal.sample_indices[0] += 1
    m.cached_out(signal)
    assert out_mock.call_count == 2


class test_multi_output_module():

    class M(base.MultiOutputModule):

        def out(self, clock_signal):
            return {"foo": clock_signal.zeros() + 1,
                    "bar": clock_signal.zeros() + 10}

    m = M()
    signal = base.ClockSignal.test_signal()
    foo = m.output("foo")
    bar = m.output("bar")
    foo_out = foo(signal)
    assert foo_out[0] == 1

    final = foo + bar

    final_out = final(signal)
    assert final_out[0] == 11


def test_direct_submodules():
    trg = Node(base.Constant(2), base.Constant(3))
    root = Node(src=base.Constant(1), trg=trg)
    actual = root._direct_submodules
    expected = {"src": base.Constant(1),
                "trg": trg}
    assert actual == expected


def test_cache_key():
    raise pytest.skip("Need to revisit sampling.")
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

    # A module without any children should not rais.e
    class NoCacheKeyModule(base.Module):
        pass

    root = Node(src=base.Constant(1), trg=NoCacheKeyModule())
    root.get_cache_key()


def test_prepend_past():

    class M(base.BaseModule):

        src: base.BaseModule
        _src_prepended: List[np.ndarray] = dataclasses.field(default_factory=list)

        def out_given_inputs(self, clock_signal, src):
            self._src_prepended.append(
                self.prepend_past("src", src, num_frames=3))
            return src

    class ArangeSource(base.BaseModule):

        def out(self, clock_signal: base.ClockSignal):
            return clock_signal.sample_indices 
            
    m = M(src=ArangeSource())
    clock = base.Clock(num_samples=4, sample_rate=1)

    m(clock())
    m(clock())
    m(clock())
    m(clock())

    clock_2 = base.Clock(num_samples=4, sample_rate=1)
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

    actual_submodules = root._filter_submodules_by_cls(cls=Leaf)
    assert expected_submodules == actual_submodules


def test_named_submodules_sum():

    class Leaf(base.BaseModule):
        v: int

        def setup(self):
            self.state = base.Stateful(self.v)
            self.param = base.Parameter(self.v)

    class M(base.BaseModule):

        def setup(self):
            self.out = 220 * sum(Leaf(i) for i in range(100))

    m = M()
    state_values = set(v["state"] for v in m.get_state_dict().values())
    assert state_values == set(range(100))

    param_values = set(p.get() for p in m.get_params_dict().values())
    assert param_values == set(range(100))


def test_state():

    class Leaf(base.BaseModule):
        inp: int
        
        def setup(self):
            self._state = base.Stateful(self.inp*10)
            self._more_state = base.Stateful(self.inp)
            self._not_state = 2.
            self._also_not_state = base.Parameter(1.)

        def out(self, clock_signal):
            self._state = 27
            return clock_signal.zeros() * self._state

    root = Node(
        src=Node(
            src=Node(src=Leaf(1), trg=Node(Leaf(2), Leaf(3))),
            trg=Leaf(4)
        ),
        trg=Node(Leaf(5), Leaf(6))
    )

    expected_state = {
        "src.src.src": {"_more_state": 1, "_state": 10}, 
        "src.src.trg.src": {"_more_state": 2, "_state": 20}, 
        "src.src.trg.trg": {"_more_state": 3, "_state": 30}, 
        "src.trg": {"_more_state": 4, "_state": 40}, 
        "trg.src": {"_more_state": 5, "_state": 50}, 
        "trg.trg": {"_more_state": 6, "_state": 60}, 
    }

    actual_state = root.get_state_dict()
    assert expected_state == actual_state

    # Update state in out
    _ = root(base.ClockSignal.test_signal())
    expected_state = {
        "src.src.src": {"_more_state": 1, "_state": 27}, 
        "src.src.trg.src": {"_more_state": 2, "_state": 27}, 
        "src.src.trg.trg": {"_more_state": 3, "_state": 27}, 
        "src.trg": {"_more_state": 4, "_state": 27}, 
        "trg.src": {"_more_state": 5, "_state": 27}, 
        "trg.trg": {"_more_state": 6, "_state": 27}, 
    }
    actual_state = root.get_state_dict()
    assert expected_state == actual_state


def test_copy_state_and_params():

    class Leaf(base.BaseModule):
        v: int

        def setup(self):
            self.state_a = base.Stateful(self.v)
            self.state_b = base.Stateful(self.v * 10)
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
            src=Node(src=Leaf(0), trg=Node(Leaf(0), Leaf(0))),
            trg=Leaf(0)
        ),
        trg=Node(Leaf(0), Leaf(0))
    )

    expected_state_values = {
        "src.src.src": {"state_a": 1, "state_b": 10},
        "src.src.trg.src": {"state_a": 2, "state_b": 20},
        "src.src.trg.trg": {"state_a": 3, "state_b": 30},
        "src.trg": {"state_a": 4, "state_b": 40},
        "trg.src": {"state_a": 5, "state_b": 50},
        "trg.trg": {"state_a": 6, "state_b": 60},
    }

    expected_param_values = {
        "src.src.src.param": 10,
        "src.src.trg.src.param": 20,
        "src.src.trg.trg.param": 30,
        "src.trg.param": 40,
        "trg.src.param": 50,
        "trg.trg.param": 60,
    }

    module_b.copy_state_from(module_a)
    module_b.set_params_from_dict(module_a.get_params_dict())
    actual_state_values = module_b.get_state_dict()
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


@pytest.fixture()
def clock_signal(num_samples):
    clock = base.Clock(num_samples=num_samples)
    clock_signal = clock()
    assert clock_signal.num_samples == num_samples
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
        _assert_outputs_similar(clock_signal, const_ten ** other, base.Constant(10**5.))

        _assert_outputs_similar(clock_signal, other * const_ten, base.Constant(50.))
        _assert_outputs_similar(clock_signal, other / const_ten, base.Constant(0.5))
        _assert_outputs_similar(clock_signal, other + const_ten, base.Constant(15.))
        _assert_outputs_similar(clock_signal, other - const_ten, base.Constant(-5.))
        _assert_outputs_similar(clock_signal, other ** const_ten, base.Constant(5**10))


def test_math_name():
    const_ten = base.Constant(10.)
    const_five = base.Constant(5.)
    result = const_ten + const_five
    assert result.name == "Math(op=add, left=Constant, right=Constant)"


def test_math_out():
    const_ten = base.Constant(10.)
    const_five = base.Constant(5.)
    result = const_ten + const_five
    signal = base.ClockSignal.test_signal()
    out = result(signal)
    assert out.shape == (signal.num_samples,)


def test_block_future_cache():
    c = base.BlockFutureCache()
    o = c.get(num_samples=10, future=np.array([1, 2, 3]))
    assert o.tolist() == [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
    assert c._cache.tolist()[:2] == [2, 3]

    c = base.BlockFutureCache()
    o = c.get(num_samples=2, future=np.array([1, 2, 3]))
    assert o.tolist() == [1, 2]
    assert c._cache.tolist()[0] == 3


def test_join_lines():
    lines = [
        "hello, world",
        "   indented lines are cool and should be"]
    output = [
        "hello, world",
        "   indented lines",
        "     are cool and",
        "     should be",
    ]
    assert base._join_lines_limit_width(*lines, width=18) == "\n".join(output)
