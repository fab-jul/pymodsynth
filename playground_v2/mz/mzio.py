"""ModernGL based live graph implementation."""
import collections
import typing
from typing import Iterable, Sequence, Tuple

import moderngl
import moderngl_window as mglw
import numpy as np

from mz import base


# Signal array are arrays of shape (num_samples, 5), where the 5 channels
# represent (x coordinates, y coor)
_SignalArray = np.ndarray
_SIGNAL_ARRAY_X = 0
_SIGNAL_ARRAY_Y = 1


def _get_windows(count):
    if count == 1:
        return [Subwindow()]
    if count <= 2:
        return [Subwindow(bottom=0.5), Subwindow(top=0.5)]
    if count <= 4:
        return [Subwindow(right=0.5, bottom=0.5), 
                Subwindow(left=0.5, bottom=0.5),
                Subwindow(right=0.5, top=0.5),
                Subwindow(left=0.5, top=0.5),
                ]
    raise NotImplementedError()


class Subwindow(typing.NamedTuple):
    # Relative coordinates
    top: float = 0.
    left: float = 0.
    right: float = 1.
    bottom: float = 1.

    def transform(self, sig: _SignalArray) -> None:
        rel_height = self.bottom - self.top
        rel_width = self.right - self.left
        sig[:, 1] *= rel_height
        sig[:, 1] -= (1 - rel_height)
        # Now the signal is at the top, i.e., y \in (-1, -1 + 2*rel_height) ,
        # and we shift it by `2*top`, butting it in (-1 + 2*top)
        sig[:, 1] += 2*self.top

        sig[:, 0] *= rel_width
        sig[:, 0] -= (1 - rel_width)
        sig[:, 0] += 2*self.left


class KeyAndMouseEvent(typing.NamedTuple):
    dx: float
    dy: float
    # The keys that were pressed, e.g., ("A", "F")
    keys: Tuple[str, ...]
    # TODO: support more modifiers.
    # Whether SHIFT is pressed.
    shift_is_on: bool


class RecordKeyPressedEvent:
    pass


class SignalWindow(mglw.WindowConfig):
    gl_version = (3, 3)

    def __init__(self,
                 ctx,
                 wnd,
                 timer,
                 event_queue: collections.deque,
                 num_samples: int,
                 ):
        super().__init__(ctx=ctx, wnd=wnd, timer=timer)

        # Map BaseKey -> ASCII, e.g., {wnd.keys.A: "A"}
        self._interesting_keys = {}
        self._modifiers = {}
        self._current_keys = []  # They are sorted.
        self._event_queue = event_queue

        self._signal = np.zeros((num_samples,), base.OUT_DTPYE)
        data_samples = 2 * num_samples
        # Static vectors needed to draw signal.
        self.x = np.linspace(-1.0, 1.0, data_samples)
        r = np.ones(data_samples)
        g = 1 / data_samples * np.arange(data_samples)
        b = 1 - (1 / data_samples * np.arange(data_samples))

        # Shape: (num_samples, 5), where the signal lives at [:, 1].
        self.vertices: _SignalArray = np.stack(
            (self.x, np.zeros_like(self.x), r, g, b), axis=-1)

        self.record_key = self._ascii_to_key("0")

        self._suppl = []

    def set_signal(self, signal, suppl):
        self._signal[:] = signal
        self._suppl = suppl#[s[:len(self._signal)] for s in suppl]

    def _ascii_to_key(self, key_as_ascii: str):
        all_keys = vars(self.wnd.keys)
        if key_as_ascii.isdigit():
            return all_keys["NUMBER_" + key_as_ascii]

        try:
            return all_keys[key_as_ascii.upper()]
        except KeyError:
            pass

        manual_assignements = {
            "/": self.wnd.keys.SLASH,
            # TODO: Add more if needed.
        }
        if key_as_ascii in manual_assignements:
            return manual_assignements[key_as_ascii]
        raise ValueError(f"Invalid key: {key_as_ascii}")

    def set_interesting_keys(self, keys_as_ascii: typing.Iterable[str]):
        self._interesting_keys = {self._ascii_to_key(k): k for k in keys_as_ascii}

    def key_event(self, key, action, modifiers):
        if key == self.record_key and action == self.wnd.keys.ACTION_RELEASE:
            self._event_queue.append(RecordKeyPressedEvent())
        if key not in self._interesting_keys:
            return
        # Key presses
        if action == self.wnd.keys.ACTION_PRESS:
            self._current_keys.append(key)
            self._modifiers["shift"] = modifiers.shift
        # Key releases
        elif action == self.wnd.keys.ACTION_RELEASE:
            self._current_keys.remove(key)

    def mouse_position_event(self, x, y, dx, dy):
        if self._current_keys:
            keys_ascii = tuple(self._interesting_keys[k] for k in self._current_keys)
            self._event_queue.append(
                KeyAndMouseEvent(dx, dy, keys_ascii,
                                 shift_is_on=self._modifiers["shift"]))

    def iter_with_subwindow(self, sigs: Sequence[_SignalArray]) -> Iterable[Tuple[Subwindow, np.ndarray]]:
        windows = _get_windows(count=len(sigs))
        yield from zip(windows, sigs)

    def render(self, time, frametime):
        prog = self.ctx.program(
            vertex_shader='''
                #version 330

                in vec2 in_vert;
                in vec3 in_color;

                out vec3 v_color;

                void main() {
                    v_color = in_color;
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330

                in vec3 v_color;

                out vec3 f_color;

                void main() {
                    f_color = v_color;
                }
            ''',
        )

        # Only visualize first channel.
        signal = self._signal[:]  # Shape: (num_samples,)
        num_samples = len(signal)
        for subwindow, sig in self.iter_with_subwindow(
            # TODO: Could draw {1, ..., 4} signals here!
            # TODO: Order is opposite of what I want somehow...
            [signal, 
            *self._suppl,
             #np.linspace(.8, .2, 2048),
             #np.linspace(.5, .1, 2048),
             #np.linspace(.2, .8, 2048)
             ]):
            if len(sig.shape) == 2:
                sig = sig[:, 0]
            if len(sig) > num_samples:
                sig = sig[:num_samples]
            elif len(sig) < num_samples:
                sig = np.concatenate((sig, np.zeros(num_samples - len(sig))), 0)

            # Reset vertices array's X axis.
            self.vertices[:, _SIGNAL_ARRAY_X] = self.x
            # Turn signal into a solid line by:
            # - setting every even element to signal + 0.1
            # - setting every odd element to signal - 0.1
            self.vertices[::2, _SIGNAL_ARRAY_Y] = sig + 0.1
            self.vertices[1::2, _SIGNAL_ARRAY_Y] = sig - 0.1

            subwindow.transform(self.vertices)

            vbo = self.ctx.buffer(self.vertices.astype('f4').tobytes())
            vao = self.ctx.vertex_array(prog, vbo, 'in_vert', 'in_color')
            vao.render(moderngl.TRIANGLE_STRIP)


def prepare_window(event_queue: collections.deque,
                   num_samples: int,
                   ) -> Tuple[mglw.BaseWindow, mglw.Timer, SignalWindow]:
    """This is a copy of moderngl_window.run_window_config.

    We add some features:
    - do not care about sys.argv
    - pass args to the config_cls instead
    """
    mglw.setup_basic_logging(SignalWindow.log_level)
    window_cls = mglw.get_local_window_cls(None)

    h, w = map(int, SignalWindow.window_size)
    size = h, w
    show_cursor = SignalWindow.cursor

    window = window_cls(
        title=SignalWindow.title,
        size=size,
        fullscreen=SignalWindow.fullscreen,
        resizable=SignalWindow.resizable,
        gl_version=SignalWindow.gl_version,
        aspect_ratio=SignalWindow.aspect_ratio,
        vsync=SignalWindow.vsync,
        samples=SignalWindow.samples,
        cursor=show_cursor if show_cursor is not None else True,
    )
    window.print_context_info()
    mglw.activate_context(window=window)
    timer = mglw.Timer()
    signal_window = SignalWindow(
        ctx=window.ctx, wnd=window, timer=timer, event_queue=event_queue,
        num_samples=num_samples)
    window.config = signal_window
    return window, timer, signal_window


def run_window_loop(window: mglw.BaseWindow, timer: mglw.Timer):
    """Run window loop."""

    # Swap buffers once before staring the main loop.
    # This can trigged additional resize events reporting
    # a more accurate buffer size
    window.swap_buffers()
    window.set_default_viewport()

    timer.start()

    try:
        while not window.is_closing:
            current_time, delta = timer.next_frame()
            if window.config.clear_color is not None:
                window.clear(*window.config.clear_color)
            # Always bind the window framebuffer before calling render
            window.use()
            window.render(current_time, delta)
            if not window.is_closing:
                window.swap_buffers()
    except KeyboardInterrupt:
        pass

    _, duration = timer.stop()
    window.destroy()
    if duration > 0:
        mglw.logger.info(
            "Duration: {0:.2f}s @ {1:.2f} FPS".format(
                duration, window.frames / duration
            )
        )


def _test():
    import queue
    q = queue.Queue()
    window, timer, signal_window = prepare_window(
        q, num_samples=2048)
    s = np.concatenate((np.linspace(0, 1, 512),
                        np.linspace(1, 0, 2048-512)), axis=0)
    signal_window.set_signal(s.reshape(-1, 1))
    run_window_loop(window, timer)


if __name__ == "__main__":
    _test()
