
import numpy as np
import dataclasses
from typing import List
from numpy import exp
import pytest

from unittest import mock

from mz import base
from playground_v2.mz.base import BaseModule


@base.moduleclass
class Node(base.BaseModule):
    src: base.Module
    trg: base.BaseModule

@base.moduleclass
class NodeModule(base.Module):
    src: base.Module
    trg: base.BaseModule


def test_cached_sampling():
    clock = base.Clock()
    root = NodeModule(base.Constant(1), base.Constant(2))
    assert root is not None

    out_mock = mock.MagicMock()
    out_mock.side_effect = root.out
    root.out = out_mock

    root.sample(clock, num_samples=10)
    root.sample(clock, num_samples=10)
    root.sample(clock, num_samples=10)

    assert out_mock.call_count == 1
    assert list(root._sample_cache.keys()) == [(repr(clock), 10, root.get_cache_key())]
    
    root.sample(clock, num_samples=11)
    assert out_mock.call_count == 2
    assert list(root._sample_cache.keys()) == [
        (repr(clock), 10, root.get_cache_key()),
        (repr(clock), 11, root.get_cache_key())]


def test_iter_direct_submodules():
    trg = Node(base.Constant(2), base.Constant(3))
    root = Node(src=base.Constant(1), trg=trg)
    actual = tuple(root._iter_direct_submodules())
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

    @base.moduleclass
    class M(base.Module):
        pass

    assert base.safe_is_subclass(base.Constant, base.BaseModule)
    assert base.safe_is_subclass(M, base.BaseModule)

    @base.moduleclass
    class N(base.Module):
        src: base.Module

    n = N(src=base.Constant(2.))
    assert n._other_modules == ["src"]


def test_prepend_past():

    @base.moduleclass
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

    raise pytest.skip("WIP")

    @base.moduleclass
    class Leaf(base.BaseModule):
        v: int

    root = Node(
        src=Node(
            src=Node(src=Leaf(1), trg=Node(Leaf(2), Leaf(3))),
            trg=Leaf(4)
        ),
        trg=Node(Leaf(5), Leaf(6))
    )

    expected_named_submodules = {
        "src.src.src": Leaf(1),
        "src.src.trg.src": Leaf(2),
        "src.src.trg.trg": Leaf(2),
        "trg.src": Leaf(5),
        "trg.trg": Leaf(6),
    }

    actual_named_submodules = root.named_submodules()
    assert expected_named_submodules == actual_named_submodules



