
import numpy as np
from mz import base
from mz import sources  # Import to populate helpers._MARKED_CLASSES.
from mz import filters  # Import to populate helpers._MARKED_CLASSES.
from mz import helpers

import pytest


# Fixtures
# ------------------------------------------------------------------------------


@pytest.fixture(params=helpers.iter_marked_classes(),
                ids=lambda marked_class: marked_class.name)
def marked_class(request):
    return request.param


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


# Actual Test
# ------------------------------------------------------------------------------



def test_marked_modules(marked_class: helpers.MarkedClass, clock_signal):
    """Instantiates and calls all marked modules."""
    # Make sure we get a fresh instance.
    marked_module_instance = marked_class.get_instance()
    _ = marked_module_instance(clock_signal)
