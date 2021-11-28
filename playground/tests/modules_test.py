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


# ------------------------------------------------------------------------------
# Actual Tests
# ------------------------------------------------------------------------------


def test_marked_modules(marked_module_instance: modules.Module):
    """Instantiates and calls all marked modules."""
    ts = np.arange(100).reshape(-1, 1) * np.ones(2)
    out = marked_module_instance(ts)
    assert out.shape == ts.shape


def _assert_outputs_similar(ts, module_a, module_b):
    out_a = module_a(ts)
    out_b = module_b(ts)
    np.testing.assert_allclose(out_a, out_b)


def test_math():
    ts = np.arange(10.)  # TODO

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
