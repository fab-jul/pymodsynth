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


def _assert_outputs_similar(clock_signal, module_a, module_b):
    out_a = module_a(clock_signal)
    out_b = module_b(clock_signal)
    np.testing.assert_allclose(out_a, out_b)


def test_math(clock_signal):
    const_ten = modules.Constant(10.)
    const_five = modules.Constant(5.)

    # Test using a scalar and a module.
    for other in (5., const_five):
        _assert_outputs_similar(clock_signal, const_ten * other, modules.Constant(50.))
        _assert_outputs_similar(clock_signal, const_ten / other, modules.Constant(2.))
        _assert_outputs_similar(clock_signal, const_ten + other, modules.Constant(15.))
        _assert_outputs_similar(clock_signal, const_ten - other, modules.Constant(5.))

        _assert_outputs_similar(clock_signal, other * const_ten, modules.Constant(50.))
        _assert_outputs_similar(clock_signal, other / const_ten, modules.Constant(0.5))
        _assert_outputs_similar(clock_signal, other + const_ten, modules.Constant(15.))
        _assert_outputs_similar(clock_signal, other - const_ten, modules.Constant(-5.))


def test_parameter_finding():

    class Nested(modules.Module):
        def __init__(self):
            self.p = modules.Parameter(1.)

    class Synth(modules.Module):
        def __init__(self):
            self.p = modules.Parameter(2.)
            # A parameter that has the same name as in a module.
            self.frequency = modules.Parameter(3.)
            self.sine0 = modules.SineSource(frequency=self.frequency)
            self.nested = Nested()
            self.sine1 = modules.SineSource(frequency=modules.Parameter(4.))
            self.sine2 = modules.SineSource(frequency=modules.Parameter(5.))
            not_assigned_to_self = modules.Parameter(6.)
            self.sine3 = modules.SineSource(frequency=not_assigned_to_self)

    synth = Synth()
    params = synth.get_params_by_name()
    param_values = {k: param.get() for k, param in params.items()}
    assert param_values == {
        "nested.p": 1.,
        "p": 2.,
        "frequency": 3.,
        "sine0.frequency": 3.,
        "sine1.frequency": 4.,
        "sine2.frequency": 5.,
        "sine3.frequency": 6.,
    }

