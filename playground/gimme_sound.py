#!/usr/bin/env python3
"""Play a sine signal.

You need sounddevice==0.4.3, which needs portaudio, seems tough on Windows, see doc:
https://python-sounddevice.readthedocs.io/en/0.4.3/installation.html

Also need moderngl and moderngl_window
https://github.com/moderngl/moderngl-window
"""

import argparse
import collections
import itertools
import queue
import random
import select
import sys
import threading
import time
import typing

import hot_reloader

import moderngl_window
import numpy as np
import sounddevice as sd

import live_graph_modern_gl
import modules


# Can contain:
# - KeyAndMouseEvent
EVENT_QUEUE = collections.deque(maxlen=100)


# TODO: This is work in progress.
class ParamSpec(typing.NamedTuple):
    param_name: str
    step: float
    # Limits of the parameter.
    lo: float = float('-inf')
    hi: float = float('inf')


class MakeSignal:

    def __init__(self, output_gen: modules.Module, sample_rate, num_channels):
        self.output_gen = output_gen
        self.params = self.output_gen.find_params()
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.i = 0
        self.last_t = time.time()
        self.t0 = -1
        self.monitor = modules.Monitor()
        self.output_gen.attach_monitor(self.monitor)

        self.mapping = {
            "f": ParamSpec("lfo.frequency", 1/100, lo=1., hi=10.),
            "s": ParamSpec("src.frequency", 1/10,  lo=10., hi=500.),
            "w": ParamSpec("lowpass.window_size", 1, lo=1., hi=500.),
        }
        live_graph_modern_gl.INTERESTING_KEYS = [k for k in self.mapping.keys()]

    def callback(self, outdata: np.ndarray, num_samples: int, timestamps, status):
        """Callback.

        Properties of `timestamps`, from the docs:
            timestamps.inputBufferAdcTime:
                ADC capture time of the first sample in the input buffer
            timestamps.outputBufferDacTime:
                DAC output time of the first sample in the output buffer
            timestamps.currentTime:
                and the time the callback was invoked.

            All are synchronized with time.time().

        More notes:
            Can raise `CallbackStop()` to finish the stream.
        """
        t = timestamps.outputBufferDacTime  # TODO(fab-jul): Use to sync.
        delta = t - self.last_t
        self.last_t = t
        if status:
            print(status, file=sys.stderr)
        ts = (self.i + np.arange(num_samples)) / self.sample_rate
        # Broadcast `ts` into (num_samples, num_channels)
        ts = ts[..., np.newaxis] * np.ones((self.num_channels,))
        assert ts.shape == (num_samples, self.num_channels)
        outdata[:] = self.output_gen(ts)
        if EVENT_QUEUE:
            s = time.time()
            # Ingest all events.
            while EVENT_QUEUE:
                event = EVENT_QUEUE.pop()
                if type(event) == live_graph_modern_gl.KeyAndMouseEvent:
                    #event: live_graph_modern_gl.KeyAndMouseEvent = EVENT_QUEUE.pop()
                    # Unpacking is supposedly faster than name access.
                    dx, dy, keys, shift_is_on = event
                    # The first key needs left/right movement ("x"),
                    # the second up/down ("y"). NOTE: we only support 2 keys.
                    for offset, k in zip((dx, dy), keys):
                        param, multiplier, lo, hi = self.mapping[k]
                        if shift_is_on:
                            multiplier *= 10
                        out = self.params[param].value + (offset * multiplier)
                        self.params[param].value = np.clip(out, lo, hi)
                #
                if type(event) == live_graph_modern_gl.SwitchMonitorEvent:
                    #event2 = live_graph_modern_gl.SwitchMonitorEvent = EVENT_QUEUE.pop()
                    self.output_gen.detach_monitor()
                    self.output_gen.sin0.attach_monitor(self.monitor)
                #
            duration = time.time() - s
            if duration > 1e-4:
                print("WARN: slow event ingestion!")
        #live_graph_modern_gl.SIGNAL[:] = outdata[:]
        live_graph_modern_gl.SIGNAL[:] = self.monitor.get_data()
        # if random.random() > 0.99:
        #     self.output_gen.detach_monitor()
        #     self.output_gen.sin0.attach_monitor(self.monitor)
        self.i += num_samples


def start_app():
    try:
        live_graph_modern_gl.run_window_config(
            live_graph_modern_gl.RandomPlot, event_queue=EVENT_QUEUE)
    except live_graph_modern_gl.QuitException:
        # Raised when the window catches the quit event.
        pass
    except KeyboardInterrupt:
        pass
    print("Window did close.")


def start_sound(device):
    # TODO: actually not working on mac.
    if device is None:
        device = 3
    sample_rate = sd.query_devices(device, 'output')['default_samplerate']

    # TODO: Turn module name into a flag and dynamically load.
    output_gen = modules.BabiesFirstSynthie()

    block_size = 512
    channels = 1
    sin = MakeSignal(output_gen, sample_rate, num_channels=channels)

    with sd.OutputStream(
            device=device, blocksize=block_size,
            latency="low",
            channels=channels, callback=sin.callback, samplerate=sample_rate):
        start_app()


def _int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        '-d', '--device', type=_int_or_str,
        help='output device (numeric ID or substring)')
    args = parser.parse_args(remaining)

    start_sound(args.device)


if __name__ == "__main__":
    main()
