#!/usr/bin/env python3
"""Play a sine signal.

You need sounddevice==0.4.3, which needs portaudio, seems tough on Windows, see doc:

https://python-sounddevice.readthedocs.io/en/0.4.3/installation.html
"""

import argparse
import queue
import sys
import threading

import numpy as np
import sounddevice as sd


_COMMAND_QUEUE = queue.Queue()


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


class MakeSin:
    def __init__(self, sample_rate, amplitude, frequency):
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self.frequency = frequency
        self.i = 0

    def callback(self, outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        t = (self.i + np.arange(frames)) / self.sample_rate
        t = t.reshape(-1, 1)
        outdata[:] = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        self.i += frames

    def up(self):
        self.frequency += 10

    def down(self):
        self.frequency -= 10


_COMMANDS = {
    "help": "Show help",
    "q": "Quit",
    "u": "Increase frequency",
    "d": "Decrease frequency",
}


def _print_help():
    print("Commands:")
    for q, info in _COMMANDS.items():
        print(q, ":", info)


def gimme_sound(device, amplitude, frequency):
    t = threading.Thread(target=start_sound, kwargs=dict(
        device=device, amplitude=amplitude, frequency=frequency))
    t.start()
    print("Started. Type `help` to get started")
    while 1:
        try:
            print("Command:")
            # This conveniently blocks this thread until input appears.
            i = input()
            if i not in _COMMANDS:
                print(f"Invalid command: `{i}`. Type `help` for help.")
                continue
            if i == "help":
                _print_help()
                continue
            _COMMAND_QUEUE.put(i, block=False)
            if i == "q":
                break
        except KeyboardInterrupt:
            _COMMAND_QUEUE.put("q")
            break
    sys.exit()


def start_sound(*, device, amplitude, frequency):
    sample_rate = sd.query_devices(device, 'output')['default_samplerate']
    sin = MakeSin(sample_rate, amplitude, frequency)

    with sd.OutputStream(
            device=device, channels=1, callback=sin.callback, samplerate=sample_rate):
        while 1:
            command = _COMMAND_QUEUE.get(block=True)
            if command == "q":
                break
            if command == "u":
                sin.up()
            elif command == "d":
                sin.down()
            else:
                print("Warning, unknown command:", command)


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
        'frequency', nargs='?', metavar='FREQUENCY', type=float, default=500,
        help='frequency in Hz (default: %(default)s)')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='output device (numeric ID or substring)')
    parser.add_argument(
        '-a', '--amplitude', type=float, default=0.2,
        help='amplitude (default: %(default)s)')
    args = parser.parse_args(remaining)

    gimme_sound(args.device, amplitude=args.amplitude, frequency=args.frequency)


if __name__ == "__main__":
    main()