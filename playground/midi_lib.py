# Needs
# pip install rtmidi
#
# https://github.com/patrickkidd/pyrtmidi
#
import collections
import dataclasses
import queue
import re
import threading
import typing

import rtmidi


class Knob(typing.NamedTuple):
    controller_number: int
    channel: int


class Event(typing.NamedTuple):
    knob: Knob
    diff: float



_VALUES = {}


def print_message(midi):
    if midi.isNoteOn():
        print('ON: ', midi.getMidiNoteName(midi.getNoteNumber()), midi.getVelocity())
    elif midi.isNoteOff():
        print('OFF:', midi.getMidiNoteName(midi.getNoteNumber()))
    print("UNKNOWN midi signal", midi)


EVENT_QUEUE = queue.Queue()


class Controller:
    @classmethod
    def make(cls, port_name_regex="Traktor Kontrol"):
        midiin = rtmidi.RtMidiIn()
        ports = range(midiin.getPortCount())
        if not ports:
            print("No ports!")
            return None
        port_to_use = None
        for i in ports:
            port_name = midiin.getPortName(i)
            sel = bool(re.search(port_name_regex, port_name))
            sel_str = ">" if sel else " "
            print(f"{sel_str} MIDI Port {i}: {port_name}")
            if sel:
                port_to_use = i
        if not port_to_use:
            print("Nothing found for", port_name_regex)
            return None

        return Controller(midiin, port_to_use)

    def __init__(self, midiin, port_to_use):
        self.midiin = midiin
        self.midiin.openPort(port_to_use)
        self.midiin.setCallback(self.callback)

        self._callbacks = {}

    def register_callback(self, knob: Knob, callback):
        self._callbacks[knob] = callback

    def read_events(self):
        while not EVENT_QUEUE.empty():
            event = EVENT_QUEUE.get_nowait()
            self._callbacks[event.knob](event.knob, event.diff)

    def callback(self, message):
        if not message.isController():
            print_message(message)
            return
        knob = Knob(message.getControllerNumber(), message.getChannel())
        if knob not in self._callbacks:
            print("Ignoring", knob)
            return
        value = message.getControllerValue()
        if knob in _VALUES:
            diff = value - _VALUES[knob]
            #kprint("Diff", diff, threading.current_thread())
            #self._callbacks[knob](diff)
            EVENT_QUEUE.put(Event(knob, diff))
        else:
            print("Got initial value for", knob)
        _VALUES[knob] = value



if __name__ == '__main__':
    Controller.make()
    input()  # Poll