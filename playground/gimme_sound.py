#!/usr/bin/env python3
"""Play a sine signal."""
import argparse
import queue
import sys
import threading

import numpy as np
import sounddevice as sd


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
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


q = queue.Queue()


class MakeSin:
    def __init__(self, samplerate, freq):
        self.samplerate = samplerate
        self.freq = freq
        self.i = 0

    def callback(self, outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        t = (self.i + np.arange(frames)) / self.samplerate
        t = t.reshape(-1, 1)
        outdata[:] = args.amplitude * np.sin(2 * np.pi * self.freq * t)
        self.i += frames

    def up(self):
        self.freq += 10

    def down(self):
        self.freq -= 10


COMMANDS = {
    "help": "Show help",
    "q": "Quit",
    "u": "Increase frequency",
    "d": "Decrease frequency",
}


def _print_help():
    print("Commands:")
    for q, info in COMMANDS.items():
        print(q, ":", info)


def gimme_sound():
    t = threading.Thread(target=start_sound_thread)
    t.start()
    print("Started. Type `help` to get started")
    while 1:
        try:
            print("Command:")
            i = input()
            if i == "help":
                _print_help()
                continue

            q.put(i, block=False)
            if i == "q":
                print("stopping...")
                break
        except KeyboardInterrupt:
            q.put("q")
            break
    sys.exit()


def start_sound_thread():
    samplerate = sd.query_devices(args.device, 'output')['default_samplerate']
    sin = MakeSin(samplerate, freq=args.frequency)

    with sd.OutputStream(
            device=args.device, channels=1, callback=sin.callback, samplerate=samplerate):
        while 1:
            entry = q.get(block=True)
            if entry == "q":
                break
            if entry == "u":
                sin.up()
            elif entry == "d":
                sin.down()



if __name__ == "__main__":
    gimme_sound()