# Needs
# pip install rtmidi
#
# https://github.com/patrickkidd/pyrtmidi
#
import argparse
import collections
import dataclasses
import enum
import queue
import re
import sys
import threading
import typing
import platform

if platform.system() == 'Windows':
    import rtmidi2 as rtmidi
else:
    import rtmidi

MidiValue = int  # 0-127
MidiValueNormalized = float  # 0...1


def normalize(value: MidiValue) -> MidiValueNormalized:
    return value / 127


class Knob(typing.NamedTuple):
    controller_number: int
    channel: int


# Either a `str` in a `KnownKnob` or a Knob instance.
KnobConvertible = typing.Union[str, Knob]


class KnownKnobs:

    @classmethod
    def from_file(cls, filename):
        return cls(_Recorder.parse_knobs_file(filename))

    def __init__(self, knobs: typing.Mapping[str, Knob]):
        self._knobs = knobs

    def __str__(self):
        return (f"KnownKnobs(\n  " +
                "\n  ".join(f"{key} -> {knob}" for key, knob in self._knobs.items()) +
                "\n)")

    def get(self, name_or_knob: KnobConvertible) -> Knob:
        if isinstance(name_or_knob, Knob):
            return name_or_knob
        # It's a string -> look it up.
        return self._knobs[name_or_knob]


class KnobEvent(typing.NamedTuple):
    knob: Knob
    rel_value: MidiValueNormalized



def _print_midi_message(midi):
    if midi.isNoteOn():
        print('ON: ', midi.getMidiNoteName(midi.getNoteNumber()), midi.getVelocity())
    elif midi.isNoteOff():
        print('OFF:', midi.getMidiNoteName(midi.getNoteNumber()))
    print("UNKNOWN midi signal", midi)


class ControllerError(Exception):
    pass


def iter_ports(midiin):
    ports = range(midiin.getPortCount())
    if not ports:
        raise ControllerError("No ports!")

    for i in ports:
        port_name = midiin.getPortName(i)
        yield i, port_name


class Controller:

    @classmethod
    def make(cls, port_name_regex="Traktor Kontrol") -> "Controller":
        """Create a Controller using the port matching `port_name_regex`.

        Returns:
            Controller instance.

        Raises:
            ControllerError: If the controller cannot be generated.
        """
        midiin = rtmidi.RtMidiIn()
        port_to_use = None
        for i, port_name in iter_ports(midiin):
            sel = bool(re.search(port_name_regex, port_name))
            sel_str = ">" if sel else " "
            print(f"{sel_str} MIDI Port {i}: {port_name}")
            if sel:
                if port_to_use is not None:
                    raise ControllerError("Already had a match!")
                port_to_use = i
        if port_to_use is None:
            raise ControllerError("Nothing found for", port_name_regex)
        print(f"Found device matching {port_to_use}: port #{port_to_use}")
        return Controller(midiin, port_to_use)

    def __init__(self, midiin, port_to_use):
        self.midiin = midiin
        self.midiin.openPort(port_to_use)
        self.midiin.setCallback(self._callback)

        self._interesting_knobs = set()
        self._unknown_knob_callback = None
        self._knob_values_cache: typing.Dict[Knob, MidiValue] = {}
        self._controller_events = queue.Queue()

    @property
    def interesting_knobs(self):
        return self._interesting_knobs

    def reset_interesting_knobs(self):
        self._interesting_knobs = set()

    def register_interesting_knob(self, knob: Knob):
        """Will mark `knob` as interesting, and track its events """
        self._interesting_knobs.add(knob)

    def register_unknown_knob_callback(self, callback):
        """The callback will be called if an unknown knob is turned.

        Here, known means that a callback was registered.
        """
        self._unknown_knob_callback = callback

    def read_events(self) -> typing.Iterable[KnobEvent]:
        """This function must be called periodically to process callback events.

        This is because `_callback` is called in a background thread by rtmidi,
        and we need to get them to the main thread.
        """
        while not self._controller_events.empty():
            yield self._controller_events.get_nowait()

    def _callback(self, message):
        """Called when a new midi event happens."""
        if not message.isController():
            _print_midi_message(message)
            return
        knob = Knob(message.getControllerNumber(), message.getChannel())
        value = message.getControllerValue()
        if knob not in self._interesting_knobs:
            if callback := self._unknown_knob_callback:
                callback(knob, value)
            else:
                print("Ignoring", knob, self._interesting_knobs)
            return
        self._controller_events.put(KnobEvent(knob, normalize(value)))


class _Recorder:
    """Helper for mapping human readable names to `Knob` instances.

    Supports interactive use. See `interactively_make_name_mapping` below.
    """

    def __init__(self, verbose=False):
        self._next_event_name = None
        self._knob_queue = queue.Queue()
        self._knobs_by_name: typing.List[typing.Tuple[str, Knob]] = []
        self._vprint = print if verbose else (lambda *_, **k: None)

        self._names = set()
        self._knobs = set()

    @property
    def names(self):
        return self._names

    @property
    def knobs(self):
        return self._knobs

    def set_next_event_name(self, name: str):
        self._next_event_name = name
        self._vprint("Draining queue...")
        while True:
            try:
                self._knob_queue.get_nowait()
                self._vprint("-", end='', flush=True)
            except queue.Empty:
                break

    def knob_changed(self, knob: Knob, value: MidiValue):
        self._vprint("e", end='', flush=True)
        self._knob_queue.put(knob)

    def wait_for_knob(self):
        self._vprint("Waiting")
        knob = self._knob_queue.get(block=True)
        # TODO: We could detect if a knob is binary by recording a few
        #  values, and checking whether they are {0, 127}.
        self._knobs_by_name.append((self._next_event_name, knob))
        self._vprint("done")
        print(f"Got knob for `{self._next_event_name}`: {knob}.")

    def to_file(self, p):
        with open(p, "a") as fout:
            for name, knob in self._knobs_by_name:
                fout.write(f"{name}:{knob}\n")

    @staticmethod
    def parse_knobs_file(p) -> typing.Mapping[str, Knob]:
        mapping = {}
        with open(p, "r") as fin:
            for line in fin:
                m = re.search(r"(.+):Knob\(controller_number=(\d+), channel=(\d+)\)",
                              line.strip())
                if not m:
                    raise ValueError(f"Invalid line: {line}")
                name, controller_number, channel = m.groups()
                mapping[name] = Knob(int(controller_number), int(channel))
        return mapping


def interactively_make_name_mapping(port_name_regex, out_file):
    c = Controller.make(port_name_regex=port_name_regex)
    recorder = _Recorder()
    c.register_unknown_knob_callback(recorder.knob_changed)
    while True:
        key_name = input("> Name the knob, or `q` to quit: ").strip()  # Poll
        if key_name == "q":
            break
        if not key_name:
            print("ERROR: Empty name.")
            continue
        if key_name in recorder.names:
            print("ERROR: Name already taken!")
            pass
        recorder.set_next_event_name(key_name)
        print("Now interact with the knob.", end=' ', flush=True)
        recorder.wait_for_knob()
    recorder.to_file(out_file)
    # Testing.
    print(KnownKnobs.from_file(out_file))


def explore(port_name_regex):
    c = Controller.make(port_name_regex)
    c.register_unknown_knob_callback(print)
    input()  # Block


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # TODO: Add list device functionality.
    p.add_argument("-p", "--port_name_regex", type=str,
                   help="Regex to find the device you connected.")

    s = p.add_subparsers(help="Mode", dest="mode", required=True)

    list_devices_p = s.add_parser("list_devices")

    map_device_p = s.add_parser("map_device")
    map_device_p.add_argument("out_file", type=str,
                              help="File for assignments. Note: will append.")

    explore_parser = s.add_parser("explore")

    flags = p.parse_args()
    if flags.mode == "list_devices":
        midiin = rtmidi.RtMidiIn()
        print("MiDi Devices:")
        for _, port in iter_ports(midiin):
            print(port)
    elif flags.mode == "map_device":
        interactively_make_name_mapping(
            flags.port_name_regex, flags.out_file)
    elif flags.mode == "explore":
        explore(flags.port_name_regex)
    else:
        p.print_usage(file=sys.stderr)
