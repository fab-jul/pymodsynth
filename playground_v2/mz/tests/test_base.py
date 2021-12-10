
import dataclasses
from numpy import exp
import pytest

from unittest import mock

from mz import base


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



