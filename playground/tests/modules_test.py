"""Modules test.

Run in PyCharm by:
1. Go to settings > Tools > Python Integrated Tools, set `pytest` as default runner, click "Fix" if needed
2. Right click this file > "Run `pytest in modules...`"
"""

import numpy as np
import pytest

from playground import modules
from playground.tests import helper as tests_helper

# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture(params=tests_helper.iter_marked_classes(),
                # Give the tests nice names:
                ids=lambda cls: cls.__class__.__name__)
def marked_module_instance(request):
    return request.param


@pytest.fixture()
def sine():
    return modules.SineSource()


@pytest.fixture()
def clock_signal():
    clock = modules.Clock()
    clock_signal = clock()
    return clock_signal

@pytest.fixture()
def ts(clock_signal):
    return clock_signal.ts

# ------------------------------------------------------------------------------
# Actual Tests
# ------------------------------------------------------------------------------


def test_marked_modules(marked_module_instance: modules.Module, clock_signal):
    """Instantiates and calls all marked modules."""
    out = marked_module_instance(clock_signal)
    assert out.shape == clock_signal.shape


def _assert_outputs_similar(ts, module_a, module_b):
    out_a = module_a(ts)
    out_b = module_b(ts)
    np.testing.assert_allclose(out_a, out_b)


def test_math(ts):
    const_ten = modules.Constant(10.)
    const_five = modules.Constant(5.)

    # Test using a scalar and a module.
    for other in (5., const_five):
        _assert_outputs_similar(ts, const_ten * other, modules.Constant(50.))
        _assert_outputs_similar(ts, const_ten / other, modules.Constant(2.))
        _assert_outputs_similar(ts, const_ten + other, modules.Constant(15.))
        _assert_outputs_similar(ts, const_ten - other, modules.Constant(5.))

        _assert_outputs_similar(ts, other * const_ten, modules.Constant(50.))
        _assert_outputs_similar(ts, other / const_ten, modules.Constant(0.5))
        _assert_outputs_similar(ts, other + const_ten, modules.Constant(15.))
        _assert_outputs_similar(ts, other - const_ten, modules.Constant(-5.))
