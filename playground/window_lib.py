"""ModernGL based live graph implementation."""
import collections
import typing
from typing import Tuple

import moderngl
import moderngl_window as mglw
import numpy as np


class KeyAndMouseEvent(typing.NamedTuple):
    dx: float
    dy: float
    # The keys that were pressed, e.g., ("A", "F")
    keys: Tuple[str, ...]
    # TODO: support more modifiers.
    # Whether SHIFT is pressed.
    shift_is_on: bool


class SignalWindow(mglw.WindowConfig):
    gl_version = (3, 3)

    def __init__(self,
                 ctx,
                 wnd,
                 timer,
                 event_queue: collections.deque,
                 num_samples: int,
                 num_channels: int,
                 ):
        super().__init__(ctx=ctx, wnd=wnd, timer=timer)

        # Map BaseKey -> ASCII, e.g., {wnd.keys.A: "A"}
        self._interesting_keys = {}
        self._modifiers = {}
        self._current_keys = []  # They are sorted.
        self._event_queue = event_queue
        self.echoes = collections.deque()

        self._signal = np.zeros((num_samples, num_channels), np.float32)
        data_samples = 2 * num_samples
        # Static vectors needed to draw signal.
        x = np.linspace(-1.0, 1.0, data_samples)
        r = np.ones(data_samples)
        g = 1 / data_samples * np.arange(data_samples)
        b = 1 - (1 / data_samples * np.arange(data_samples))

        # Shape: (num_samples, 5), where the signal lives at [:, 1].
        self.vertices = np.stack((x, np.zeros_like(x), r, g, b), axis=-1)

    def set_signal(self, signal):
        self._signal[:] = signal

    def set_interesting_keys(self, keys_as_ascii: typing.Iterable[str]):
        self._interesting_keys = {vars(self.wnd.keys)[k.upper()]: k for k in keys_as_ascii}

    def key_event(self, key, action, modifiers):
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
        signal = self._signal[:, 0]  # Shape: (num_samples,)

        FFT = False
        if FFT:
            fft = np.fft.fft(signal, n=signal.shape[0])
            fft.resize((512,))
            #print("fft", fft.shape)
            fft_abs = np.abs(fft)
            signal = fft_abs

        # self.vertices = np.stack((x, np.zeros_like(x), r, g, b), axis=-1)
        # Turn signal into a solid line by:
        # - setting every even element to signal + 0.1
        # - setting every odd element to signal - 0.1
        self.vertices[::2, 1] = signal + 0.1
        self.vertices[1::2, 1] = signal - 0.1

        self.echoes.append(self.vertices[:])
        if len(self.echoes) > 1:
            self.echoes.popleft()
        for i, sig in enumerate(self.echoes):
            inv_i = len(self.echoes) - i
            fade = pow(1.2, inv_i)
            # TODO: add in some dimensions, divide in some others
            #vertices = np.dstack([_X, sig + inv_i*0.07, _R/fade, _G/fade, _B/fade])
            vbo = self.ctx.buffer(sig.astype('f4').tobytes())
            vao = self.ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_color')
            vao.render(moderngl.TRIANGLE_STRIP)


def prepare_window(event_queue: collections.deque,
                   num_samples: int,
                   num_channels: int) -> Tuple[mglw.BaseWindow, mglw.Timer, SignalWindow]:
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
        num_samples=num_samples, num_channels=num_channels)
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

