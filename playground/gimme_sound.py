#!/usr/bin/env python3
"""Play a sine signal.

You need sounddevice==0.4.3, which needs portaudio, seems tough on Windows, see doc:
https://python-sounddevice.readthedocs.io/en/0.4.3/installation.html

Also need moderngl and moderngl_window
https://github.com/moderngl/moderngl-window
"""

import argparse
import collections
import midi_lib
import dataclasses
import importlib
import os.path
import traceback

import params_lib
import filewatcher
import sys
import time
import typing

import numpy as np
import sounddevice as sd

import live_graph_modern_gl
import modules


# Can contain:
# - KeyAndMouseEvent
EVENT_QUEUE = collections.deque(maxlen=100)


# Store all current timers.
_TIMERS: typing.List["Timer"] = []


@dataclasses.dataclass
class Timer:
    # Seconds, may have fractional seconds
    fire_every: float
    repeats: bool
    callback: typing.Callable[..., None]

    def register(self):
        next_fire = time.time() + self.fire_every
        _TIMERS.append((next_fire, self))


def call_timers():
    now = time.time()
    indices_of_timers_to_delete = []
    for i, (next_fire, timer) in enumerate(_TIMERS):
        if now >= next_fire:
            timer.callback()
            if timer.repeats:
                _TIMERS[i] = (now + timer.fire_every, timer)
            else:
                indices_of_timers_to_delete.append(i)
    for i in reversed(indices_of_timers_to_delete):
        del _TIMERS[i]


def _get_modules_path():
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, "modules.py")


class MakeSignal:

    def __init__(self, output_gen_class: str, sample_rate, num_channels):
        self.sample_rate = sample_rate
        modules.SAMPLING_FREQUENCY = sample_rate  # TODO: a hack until it's clear how to pass
        self.num_channels = num_channels
        self.i = 0
        self.last_t = time.time()
        self.t0 = -1

        # TODO: hot reload
        # TODO: adapt keys
        self.output_gen_class = output_gen_class
        self.output_gen = self._make_output_gen()

        self.params = self.output_gen.find_params()
        self.state = self.output_gen.find_state()

        self.modules_watcher = filewatcher.FileModifiedTimeTracker(
            _get_modules_path())
        self.modules_watcher.did_read_file_just_now()

        self.monitor = modules.Monitor()
        self.midi_controller: typing.Optional[midi_lib.Controller] = midi_lib.Controller.make()
        if self.midi_controller:
            def make_callback(key):
                def callback(_, d):
                    EVENT_QUEUE.appendleft(
                        live_graph_modern_gl.KeyAndMouseEvent(
                            d, d, key, True))
                return callback
            self.midi_controller.register_callback(
                midi_lib.Knob(30, 5), make_callback("f"))
            self.midi_controller.register_callback(
                midi_lib.Knob(31, 5), make_callback("w"))
        self.output_gen.attach_monitor(self.monitor)

        # TODO(fab-jul): Push this into Parameter class,
        #  and ditch params_* files.
        # Setup params to key_mapping stuff.
        self.params_file = f"params_{output_gen_class}.txt"
        if not os.path.isfile(self.params_file):
            print(f"Creating {self.params_file}...")
            # Create initial dump.
            # TODO: have to update when generator function changes!
            param_specs = [params_lib.ParamSpec(name) for name in
                           self.params.keys()]
            params_lib.write_params(param_specs, self.params_file)
        self.params_watcher = filewatcher.FileModifiedTimeTracker(
            self.params_file)
        self.key_mapping = {}

        # This will be called periodically in the event loop via a Timer.
        Timer(fire_every=1,
              repeats=True,
              callback=self.update_keymapping_from_params_file).register()

        Timer(fire_every=1,
              repeats=True,
              callback=self.reload_modules).register()

        self.time_of_last_timer_update = 0.0

    def _make_output_gen(self) -> modules.Module:
        avaiable_vars = vars(modules)
        if self.output_gen_class not in avaiable_vars:
            raise ValueError(f"Invalid class: {self.output_gen_class}")
        print(f"Creating {self.output_gen_class}...")
        return avaiable_vars[self.output_gen_class]()

    def reload_modules(self):
        if not self.modules_watcher.has_changes:
            return

        self.modules_watcher.did_read_file_just_now()

        print("Reading modules.py ...")
        importlib.reload(modules)
        try:
            new_instance = self._make_output_gen()
        except:
            traceback.print_exc()
            print(f"Blanked catch, not reloading...")
            return

        # Copy old state and params.
        new_instance.copy_params_and_state_from(
            src_params=self.params,
            src_state=self.state)

        # TODO
        self.output_gen.detach_monitor()
        self.output_gen = new_instance
        self.params = self.output_gen.find_params()
        self.state = self.output_gen.find_state()
        self.output_gen.attach_monitor(self.monitor)

    def update_keymapping_from_params_file(self):
        if not self.params_watcher.has_changes:
            # No changes, nothing to read!
            return
        print("Reading updated params...")
        param_specs = params_lib.parse_params(self.params_file)
        self.key_mapping = {
            param_spec.key: param_spec for param_spec in param_specs
            if param_spec.key is not None}

        if window := live_graph_modern_gl.get_current_window():
            window.set_interesting_keys(self.key_mapping.keys())
            self.params_watcher.did_read_file_just_now()  # Signal that we ingested changes.
        else:
            print("WARN: No window available!")


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
            All are synchronized with time.time().

        More notes:
            Can raise `CallbackStop()` to finish the stream.
        """
        if controller := self.midi_controller:
            controller.read_events()
        t = timestamps.outputBufferDacTime  # TODO(fab-jul): Use to sync.
        # For performance, only update timers at most 1 per second
        if t - self.time_of_last_timer_update >= 1.:
            call_timers()
            self.time_of_last_timer_update = t
        # delta = t - self.last_t
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
                        param, multiplier, lo, hi, _ = self.key_mapping[k]
                        if shift_is_on:
                            multiplier *= 10
                        out = self.params[param].value + (offset * multiplier)
                        self.params[param].value = np.clip(out, lo, hi)
                #
                if type(event) == live_graph_modern_gl.SwitchMonitorEvent:
                    print("Attaching to sin")
                    #event2 = live_graph_modern_gl.SwitchMonitorEvent = EVENT_QUEUE.pop()
                    self.output_gen.detach_monitor()
                    self.output_gen.sin0.frequency.attach_monitor(self.monitor)
                #
            duration = time.time() - s
            if duration > 1e-4:
                print("WARN: slow event ingestion!")
        if window := live_graph_modern_gl.get_current_window():
            window.set_signal(self.monitor.get_data())
        # if random.random() > 0.99:
        #     self.output_gen.detach_monitor()
        #     self.output_gen.sin0.attach_monitor(self.monitor)
        self.i += num_samples


def start_app(num_samples: int, num_channels: int):
    try:
        live_graph_modern_gl.run_window_config(
            live_graph_modern_gl.RandomPlot,
            event_queue=EVENT_QUEUE,
            num_samples=num_samples,
            num_channels=num_channels)
    except live_graph_modern_gl.QuitException:
        # Raised when the window catches the quit event.
        pass
    except KeyboardInterrupt:
        pass
    print("Window did close.")


def start_sound(output_gen_class: str, device: int):
    if device < 0:
        device = None  # Auto-select.
    sample_rate = sd.query_devices(device, 'output')['default_samplerate']

    # TODO(fab-jul): Investigate how large we can make this.
    num_samples = 2048
    num_channels = 1
    sin = MakeSignal(output_gen_class=output_gen_class,
                     sample_rate=sample_rate,
                     num_channels=num_channels)

    with sd.OutputStream(
            device=device, blocksize=num_samples,
            latency="low",
            channels=num_channels, callback=sin.callback, samplerate=sample_rate):
        start_app(num_samples, num_channels)


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
        '-d', '--device', type=int, default=3,  # TODO: Go back to None
        help='output device (numeric ID or substring)')
    parser.add_argument(
        "--output_gen_class", default="ClickModulation")
    args = parser.parse_args(remaining)

    start_sound(args.output_gen_class, args.device)


if __name__ == "__main__":
    main()
