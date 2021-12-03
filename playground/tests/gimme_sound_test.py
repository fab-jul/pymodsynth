import numpy as np
import pytest
from unittest import mock
from playground import gimme_sound
from playground import modules


@pytest.fixture()
def time_stamps():
    """Returns an instance with the same attribute as the sound_device timestamps."""
    class _TimeStamps:
        pass

    timestamps = _TimeStamps()
    timestamps.inputBufferAdcTime = 1.
    timestamps.outputBufferDacTime = 1.
    timestamps.currentTime = 1.
    return timestamps


@pytest.fixture()
def basic_module():
    return modules.SineSource()


@pytest.fixture()
def num_samples():
    return 2048


@pytest.fixture()
def num_channels():
    return 2


def _make_synthesizer_controller(
        output_gen_class, num_samples, num_channels):
    return gimme_sound.SynthesizerController(
        output_gen_class,
        sample_rate=44100,
        num_samples=num_samples,
        num_channels=num_channels,
        signal_window=mock.MagicMock()
    )


def test_synthesizer_controller(num_samples, num_channels, time_stamps, basic_module):
    synthesizer_controller = _make_synthesizer_controller(
        output_gen_class=basic_module.__class__.__name__,
        num_samples=num_samples, num_channels=num_channels)
    outdata = np.empty((num_samples, num_channels), modules.OUT_DTYPE)
    synthesizer_controller.callback(
        outdata, num_samples=2048, timestamps=time_stamps, status=None)
