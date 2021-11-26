#!/usr/bin/env python3
"""Play a sine signal.

You need sounddevice==0.4.3, which needs portaudio, seems tough on Windows, see doc:
https://python-sounddevice.readthedocs.io/en/0.4.3/installation.html

Also need moderngl and moderngl_window
https://github.com/moderngl/moderngl-window
"""

import argparse
import collections
import threading

import sounddevice

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

    def __init__(self, output_gen_class: str, sample_rate, num_samples, num_channels,
                 signal_window: live_graph_modern_gl.SignalWindow):
        self.sample_rate = sample_rate
        modules.SAMPLING_FREQUENCY = sample_rate  # TODO: a hack until it's clear how to pass
        self.num_channels = num_channels
        self.clock = modules.Clock(num_samples, num_channels, sample_rate)
        self.last_t = time.time()
        self.t0 = -1
        self.signal_window = signal_window

        # Set defaults.
        self.params: typing.Dict[str, modules.Parameter] = {}
        self.state: typing.Dict[str, modules.State] = {}
        self.key_mapping: typing.Dict[str, modules.Parameter] = {}
        self.knob_mapping: typing.Dict[midi_lib.Knob, modules.Parameter] = {}

        try:
            # TODO: make file a flag!
            # TODO: Support file reloading.
            self.known_knobs = midi_lib.KnownKnobs.from_file("traktor_kontrol_knobs.txt")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            self.known_knobs = midi_lib.KnownKnobs({})

        self.monitor = modules.Monitor()

        self.output_gen_class = output_gen_class
        self.output_gen = self._make_output_gen()

        self.modules_watcher = filewatcher.FileModifiedTimeTracker(
            _get_modules_path())
        self.modules_watcher.did_read_file_just_now()

        self.midi_controller: typing.Optional[midi_lib.Controller] = None
        try:
            self.midi_controller = midi_lib.Controller.make()
        except midi_lib.ControllerError as e:
            print(f"Caught: {e}")

        Timer(fire_every=1,
              repeats=True,
              callback=self.reload_modules).register()

        self.time_of_last_timer_update = 0.0

        self._set_output_gen(self.output_gen)

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

        self._set_output_gen(new_instance)

    def _set_output_gen(self, output_gen: modules.Module):
        """Called on init and when modules.py changes."""
        self.output_gen.detach_monitor()
        self.output_gen = output_gen
        self.params = self.output_gen.find_params()
        self.state = self.output_gen.find_state()
        self.output_gen.attach_monitor(self.monitor)
        self._update_key_mapping()
        self._update_knob_mapping()

    def _update_key_mapping(self):
        self.key_mapping = {param.key: param
                            for _, param in self.params.items()
                            if param.key}

        print("Did update keymapping, keys=", self.key_mapping.keys())
        self.signal_window.set_interesting_keys(self.key_mapping.keys())

    def _update_knob_mapping(self):
        if not self.midi_controller:
            return

        self.knob_mapping = {self.known_knobs.get(param.knob): param
                             for _, param in self.params.items()
                             if param.knob}
        print("Knob_mapping=", self.knob_mapping)
        print("self.params=", self.params)

        self.midi_controller.reset_interesting_knobs()
        for knob in self.knob_mapping:
            self.midi_controller.register_interesting_knob(knob)

        print("Did update interesting knobs to", self.midi_controller.interesting_knobs)

    def _process_midi_controller_events(self):
        if controller := self.midi_controller:
            for event in controller.read_events():
                EVENT_QUEUE.append(event)

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
        self._process_midi_controller_events()
        t = timestamps.outputBufferDacTime  # TODO(fab-jul): Use to sync.
        # For performance, only update timers at most 1 per second
        if t - self.time_of_last_timer_update >= 1.:
            call_timers()
            self.time_of_last_timer_update = t
        # delta = t - self.last_t
        self.last_t = t
        if status:
            print(status, file=sys.stderr)

        clock_signal = self.clock()

        outdata[:] = self.output_gen(clock_signal)

        if EVENT_QUEUE:
            s = time.time()
            # Ingest all events.
            while EVENT_QUEUE:
                event = EVENT_QUEUE.popleft()  # We are a queue, pop from the left, append to the right.
                if isinstance(event, live_graph_modern_gl.KeyAndMouseEvent):
                    # Unpacking is supposedly faster than name access.
                    dx, dy, keys, shift_is_on = event
                    # The first key needs left/right movement ("x"),
                    # the second up/down ("y"). NOTE: we only support 2 keys.
                    for offset, k in zip((dx, dy), keys):
                        param: modules.Parameter = self.key_mapping[k]
                        multiplier = param.shift_multiplier if shift_is_on else 1
                        param.inc(offset * multiplier)

                elif isinstance(event, midi_lib.KnobEvent):
                    knob, rel_value = event
                    param: modules.Parameter = self.knob_mapping[knob]
                    param.set_relative(rel_value)

                # TODO: Clean up
                elif isinstance(event, live_graph_modern_gl.SwitchMonitorEvent):
                    print("Attaching to sin")
                    #event2 = live_graph_modern_gl.SwitchMonitorEvent = EVENT_QUEUE.pop()
                    self.output_gen.detach_monitor()
                    self.output_gen.sin0.frequency.attach_monitor(self.monitor)

                else:
                    raise TypeError(event)

#            duration = time.time() - s
#            if duration > 1e-4:
#                print("WARN: slow event ingestion!")
        self.signal_window.set_signal(self.monitor.get_data())
        # if random.random() > 0.99:
        #     self.output_gen.detach_monitor()
        #     self.output_gen.sin0.attach_monitor(self.monitor)


def start_sound(output_gen_class: str, device: int):
    if device < 0:
        device = None  # Auto-select.
    sample_rate = sd.query_devices(device, 'output')['default_samplerate']

    # TODO(fab-jul): Investigate how large we can make this.
    num_samples = 2048
    num_channels = 1

    # We first make a window, so that we always have that.
    window, timer, signal_window = live_graph_modern_gl.prepare_window(
        EVENT_QUEUE, num_samples=num_samples, num_channels=num_channels)

    # Now we make the signal maker.
    sin = MakeSignal(output_gen_class=output_gen_class,
                     sample_rate=sample_rate,
                     num_samples=num_samples,
                     num_channels=num_channels,
                     signal_window=signal_window)

    # Start audio stream.
    with sd.OutputStream(
            device=device, blocksize=num_samples,
            latency="low", channels=num_channels, callback=sin.callback, samplerate=sample_rate):
        # Start window event loop.
        live_graph_modern_gl.run_window_loop(window, timer)


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
