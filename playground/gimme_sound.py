#!/usr/bin/env python3
"""Play a sine signal.

You need sounddevice==0.4.3, which needs portaudio, seems tough on Windows, see doc:
https://python-sounddevice.readthedocs.io/en/0.4.3/installation.html

Also need moderngl and moderngl_window
https://github.com/moderngl/moderngl-window
"""

import argparse
import queue
import select
import sys
import threading
import time
import v1
import hot_reloader

import moderngl_window
import numpy as np
import sounddevice as sd

import live_graph_modern_gl


# Contains input commands.
_COMMAND_QUEUE = queue.Queue()


# Used to signal need to stop program.
_QUIT_EVENT = threading.Event()

import Modules

class MakeSignal:

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.i = 0
        self.last_t = time.time()
        self.t0 = -1

        #self.output_generator = v1.OutputGeneratorV1()
        self.output_gen = Modules.BabiesFirstSynthie()

        #self.output_generator = hot_reloader.parse(PRG_SIN)

    def callback(self, outdata: np.ndarray, frames: int, timestamps, status):
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
        ts = (self.i + np.arange(frames)) / self.sample_rate
        outdata[:] = self.output_gen(ts).reshape(-1,1)
        print("ts", ts)

        live_graph_modern_gl.SIGNAL[:] = outdata[:]
        self.i += frames



_COMMANDS = {
    "h": "Show help",
    "q": "Quit",
    "u": "Increase frequency",
    "d": "Decrease frequency",
}


def _print_help():
    print("Commands:")
    for q, info in _COMMANDS.items():
        print(q, ":", info)


def gimme_sound(device, amplitude, frequency):
    # We start sound and input fetchers in background threads,
    # and the app in the main thread. We use _QUIT_EVENT to signal others
    # if one of them wants to quit (i.e., when the window is closed,
    # or when `q` is typed in the console).
    threading.Thread(target=start_sound, kwargs=dict(
        device=device, amplitude=amplitude, frequency=frequency)).start()
    t = threading.Thread(target=start_input_fetcher)
    t.start()
    start_app()  # This blocks until the app is closed.
    t.join()  # If we land here, app is closed, so wait for this thread.


def start_input_fetcher(timeout=1):
    print("Started. Type `h` to get started")
    time.sleep(3)
    # Whether we should print the `Command:` prompt.
    print_command = True
    while 1:
        if _QUIT_EVENT.is_set():
            print("Stopping input fetcher...")
            break
        try:
            if print_command:
                print("Command:")
                print_command = False
            # We cannot use input() here, as that blocks the thread, which would mean
            # we would not get quit events. Instead, we rely on signals from stdin,
            # but we should improve this implementation, as it now has a delay.
            # TODO(fab-jul): Should rewrite this to be a keyhandler.
            i_signal, _, _ = select.select([sys.stdin], [], [], timeout)
            if not i_signal:  # No input on this cycle, move on...
                continue
            i = sys.stdin.readline().strip()

            if i not in _COMMANDS:
                print(f"Invalid command: `{i}`. Type `h` for help.")
                continue

            print_command = True

            if i == "h":
                _print_help()
                continue
            if i == "q":
                print("Sending quit event...")
                _QUIT_EVENT.set()
                break
            _COMMAND_QUEUE.put(i, block=False)
        except KeyboardInterrupt:
            _QUIT_EVENT.set()
            break


def start_app():
    live_graph_modern_gl.RandomPlot.QUIT_EVENT = _QUIT_EVENT
    try:
        timer = moderngl_window.Timer()
        moderngl_window.run_window_config(
            live_graph_modern_gl.RandomPlot, timer=timer)
    except live_graph_modern_gl.QuitException:
        # Raised when the window catches the quit event.
        pass
    except KeyboardInterrupt:
        pass
    _QUIT_EVENT.set()
    print("Window did close.")


def start_sound(*, device, amplitude, frequency):
    if device is None:
        #TODOdevice = 3
        pass
    sample_rate = sd.query_devices(device, 'output')['default_samplerate']
    #print(sd.query_devices(device, 'output'))
    sin = MakeSignal(sample_rate)

    with sd.OutputStream(
            device=device, blocksize=block_size,
            channels=1, callback=sin.callback, samplerate=sample_rate):
        while 1:
            if _QUIT_EVENT.is_set():
                break

            try:
                command = _COMMAND_QUEUE.get(block=False)
            except queue.Empty:
                time.sleep(0.01)
                continue

            if command == "u":
                pass
            elif command == "d":
                pass
            else:
                print("Warning, unknown command:", command)


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
        'frequency', nargs='?', metavar='FREQUENCY', type=float, default=240,
        help='frequency in Hz (default: %(default)s)')
    parser.add_argument(
        '-d', '--device', type=_int_or_str,
        help='output device (numeric ID or substring)')
    parser.add_argument(
        '-a', '--amplitude', type=float, default=0.4,
        help='amplitude (default: %(default)s)')
    args = parser.parse_args(remaining)

    gimme_sound(args.device, amplitude=args.amplitude, frequency=args.frequency)


if __name__ == "__main__":
    main()