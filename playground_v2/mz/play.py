#!/usr/bin/env python3
"""Play a sine signal.

You need sounddevice==0.4.3, which needs portaudio, seems tough on Windows, see doc:
https://python-sounddevice.readthedocs.io/en/0.4.3/installation.html

Also need moderngl and moderngl_window
https://github.com/moderngl/moderngl-window
"""

import collections
import datetime
import getpass
from os import getcwd
import re
import dataclasses
import importlib
import os.path
import traceback

import sys
import time
import typing

import numpy as np
import scipy.io.wavfile
import sounddevice as sd

from mz import filewatcher
from mz import midi_lib

from mz import io
from mz import base


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


DURATION_FIVE_MIN = 5 * 60


class Recorder:

    def __init__(self,
                 frames_per_second: int,
                 sample_rate: int,
                 max_recording_time_s: int = DURATION_FIVE_MIN):
        self._is_recording = False
        self._sample_rate = sample_rate

        self._current_file_name = None

        num_frames_to_record = max_recording_time_s * frames_per_second
        print(f"Storing at most {num_frames_to_record} frames for recorder.")
        self._buffer = collections.deque(maxlen=num_frames_to_record)

    @staticmethod
    def _get_audio_out_dir(root="audio_out"):
        root_for_user = os.path.join(root, getpass.getuser())
        os.makedirs(root_for_user, exist_ok=True)
        return root_for_user

    @staticmethod
    def _file_name():
        now_as_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_out_filename = os.path.join(Recorder._get_audio_out_dir(), now_as_str + ".wav")
        return audio_out_filename

    def toggle_is_recording(self) -> bool:
        self._is_recording = not self._is_recording
        if self._is_recording:
            self._current_file_name = self._file_name()
        else:
            self.write_out()
        return self._is_recording

    def push(self, outdata: np.ndarray):
        if self._is_recording:
            if self._buffer is None:
                raise ValueError("Need to call setup() before push()!")
            self._buffer.append(outdata.copy())
            print("Buffer has", len(self._buffer), "els")

    def write_out(self):
        print(f"Saving to {self._current_file_name}...")
        # TODO: Figure out why the heck divide by 2.
        sig = np.concatenate(self._buffer, axis=0)[:, 0] / 2
        normalize_sig = np.clip(sig, -1, 1)
        scipy.io.wavfile.write(self._current_file_name, self._sample_rate, normalize_sig)



class SynthesizerController:

    def __init__(self,
                 modules_file_name: str, output_gen_class: str, sample_rate, num_samples, num_channels,
                 signal_window: io.SignalWindow, recorder: Recorder,
                 midi_knobs_file: str, midi_port_name_regex: str):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.clock = base.Clock(num_samples, num_channels, sample_rate)
        self.last_t = time.time()
        self.signal_window = signal_window
        self.recorder = recorder

        # Set defaults.
        self.params: typing.Dict[str, base.Parameter] = {}
        self.state: typing.Dict[str, base.State] = {}
        self.key_mapping: typing.Dict[str, base.Parameter] = {}
        self.knob_mapping: typing.Dict[midi_lib.Knob, base.Parameter] = {}

        try:
            # TODO: make a flag!
            # TODO: Support reloading
            self.known_knobs = midi_lib.KnownKnobs.from_file(midi_knobs_file)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            self.known_knobs = midi_lib.KnownKnobs({})

        self.modules_file_name = modules_file_name
        self.output_gen_class = output_gen_class
        self.output_gen = self._make_output_gen()

        modules_file_path = os.path.join(os.getcwd(), modules_file_name)
        if not os.path.isfile(modules_file_path):
            raise FileNotFoundError(f"File not found: {modules_file_path}!")
        self.modules_watcher = filewatcher.FileModifiedTimeTracker(modules_file_path)
        self.modules_watcher.did_read_file_just_now()

        self.midi_controller: typing.Optional[midi_lib.Controller] = None
        try:
            self.midi_controller = midi_lib.Controller.make(port_name_regex=midi_port_name_regex)
        except Exception as e:
            print(f"Failed to create midi controller, caught `{e}`", file=sys.stderr)

        Timer(fire_every=1,
              repeats=True,
              callback=self.reload_modules).register()

        self.time_of_last_timer_update = 0.0

        self._set_output_gen(self.output_gen)

    @property
    def modules_import_name(self) -> str:
        """Strip the .py!"""
        return re.sub(r"\.py$", "", self.modules_file_name)

    def _make_output_gen(self, module_to_check=None) -> base.Module:
        if module_to_check is None:
            module_to_check = importlib.import_module(self.modules_import_name, package="playground")
        available = vars(module_to_check)
        if self.output_gen_class not in available:
            raise ValueError(
                f"Invalid class: `{self.output_gen_class}`, not in file `{self.modules_file_name}`!"
                f"Available: " + "\n".join(sorted(available.keys())))
        print(f"Creating {self.output_gen_class}...")
        return available[self.output_gen_class]()

    def reload_modules(self):
        if not self.modules_watcher.has_changes:
            return

        self.modules_watcher.did_read_file_just_now()

        print(f"Trying to reload {self.modules_file_name}...")
        # noinspection PyBroadException
        try:
            modules_new = importlib.import_module(self.modules_import_name, "playground")
            importlib.reload(modules_new)
            new_instance = self._make_output_gen(module_to_check=modules_new)
        except:
            traceback.print_exc()
            print("*** Caught exception while reloading, rolling back...", file=sys.stderr)
            return

        # Try passing through one clock signal to catch errors in `out`.
        clock_signal = self.clock.get_current_clock_signal()

        # noinspection PyBroadException
        try:
            new_instance(clock_signal)
        except:
            traceback.print_exc()
            print("*** Caught exception while reloading, rolling back...", file=sys.stderr)
            return

        # Copy old state and params.
        new_instance.set_params_from_dict(self.params)
        new_instance.set_state_from_dict(self.state)

        # All good, can use new code.
        self._set_output_gen(new_instance)
        print(f"Reloaded {self.modules_file_name}!")

    def _set_output_gen(self, output_gen: base.Module):
        """Called on init and when `self.modules_file_name` changes."""
        self.output_gen = output_gen
        self.params = self.output_gen.get_params_dict()
        self.state = self.output_gen.get_state_dict()
        self._update_key_mapping()
        self._update_knob_mapping()

    def _update_key_mapping(self):
        self.key_mapping = {}
        for param in set(self.params.values()):
            if not param.key:
                continue
            if param.key in self.key_mapping:
                print(f"*** Duplicate key, `{param.key}` already used! Ignoring...", file=sys.stderr)
                continue
            self.key_mapping[param.key] = param
        print("Did update keymapping, keys=", self.key_mapping.keys())
        self.signal_window.set_interesting_keys(self.key_mapping.keys())

    def _update_knob_mapping(self):
        if not self.midi_controller:
            return
        self.knob_mapping = {}
        self.midi_controller.reset_interesting_knobs()
        for param in set(self.params.values()):
            if not param.knob:
                continue
            if param.knob in self.knob_mapping:
                print(f"*** Duplicate knob, `{param.knob}` already used! Ignoring...", file=sys.stderr)
                continue
            knob = self.known_knobs.get(param.knob)
            self.knob_mapping[knob] = param
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

        # Ingest all events.
        while EVENT_QUEUE:
            event = EVENT_QUEUE.popleft()  # We are a queue, pop from the left, append to the right.
            if isinstance(event, io.KeyAndMouseEvent):
                # Unpacking is supposedly faster than name access.
                dx, dy, keys, shift_is_on = event
                # The first key needs left/right movement ("x"),
                # the second up/down ("y"). NOTE: we only support 2 keys.
                for offset, k in zip((dx, dy), keys):
                    param: base.Parameter = self.key_mapping[k]
                    multiplier = param.shift_multiplier if shift_is_on else 1
                    param.inc(offset * multiplier)

            elif isinstance(event, midi_lib.KnobEvent):
                knob, rel_value = event
                param: base.Parameter = self.knob_mapping[knob]
                param.set_relative(rel_value)

            elif isinstance(event, io.RecordKeyPressedEvent):
                is_recording = self.recorder.toggle_is_recording()
                print("Recording:", is_recording)

            else:
                raise TypeError(event)

        self.signal_window.set_signal(outdata)
        self.recorder.push(outdata)


def start_sound_loop(modules_file_name: str,
                     output_gen_class: str,
                     device: int,
                     record_max_minutes: int,
                     midi_knobs_file: str,
                     midi_port_name_regex: str):
    """Start the sound event loop."""
    if device < 0:
        device = None  # Auto-select.
    sample_rate = sd.query_devices(device, "output")["default_samplerate"]

    # TODO(fab-jul): Investigate how large we can make this.
    num_samples = 2048
    num_channels = 1

    # TODO: document that we record when we press `0`.
    recorder = Recorder(frames_per_second=round(sample_rate / num_samples),
                        sample_rate=round(sample_rate),
                        max_recording_time_s=record_max_minutes * 60)

    window, timer, signal_window = io.prepare_window(
        EVENT_QUEUE, num_samples=num_samples, num_channels=num_channels)

    syntheziser_controller = SynthesizerController(
        modules_file_name=modules_file_name,
        output_gen_class=output_gen_class,
        sample_rate=sample_rate,
        num_channels=num_channels,
        num_samples=num_samples,
        signal_window=signal_window,
        recorder=recorder,
        midi_knobs_file=midi_knobs_file,
        midi_port_name_regex=midi_port_name_regex)

    # Start audio stream.

    with sd.OutputStream(
            device=device,
            blocksize=num_samples,
            latency="low",
            channels=num_channels,
            callback=syntheziser_controller.callback,
            samplerate=sample_rate):
        # Start window event loop. The audio stream will live
        # as long as the window is open.
        io.run_window_loop(window, timer)


def list_devices():
    print(sd.query_devices())