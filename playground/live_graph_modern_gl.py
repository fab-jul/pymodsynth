"""ModernGL based live graph implementation."""
import collections
import queue
import threading
import typing
from typing import Tuple

import moderngl
import moderngl_window as mglw
import numpy as np


# TODO: make independent of num_samples and num_channels.
NUM_SAMPLES = 512
# Shared among users. Overwrite to update view.
# NOTE: the 1 is because we have a one channel output.
SIGNAL = np.zeros((512, 1), np.float32)

DATA_SAMPLES = 2 * NUM_SAMPLES

# Static vectors needed to draw signal.
_X = np.linspace(-1.0, 1.0, DATA_SAMPLES)
_R = np.ones(DATA_SAMPLES)
_G = 1/DATA_SAMPLES * np.arange(DATA_SAMPLES)
_B = 1 - (1/DATA_SAMPLES * np.arange(DATA_SAMPLES))


# TODO: wire up properly.
# ASCII version of keys we are interested.
INTERESTING_KEYS = []


class QuitException(Exception):
    """Raised to exit the render loop. Needs to be caught."""
    pass


class KeyAndMouseEvent(typing.NamedTuple):
    dx: float
    dy: float
    # The keys that were pressed, e.g., ("A", "F")
    keys: Tuple[str, ...]
    # TODO: support more modifiers.
    # Whether SHIFT is pressed.
    shift_is_on: bool


class RandomPlot(mglw.WindowConfig):
    gl_version = (3, 3)

    def __init__(self,
                 ctx,
                 wnd,
                 timer,
                 event_queue: collections.deque,
                 ):
        super().__init__(ctx=ctx, wnd=wnd, timer=timer)

        # Map BaseKey -> ASCII, e.g., {wnd.keys.A: "A"}
        #self._interesting_keys = {}
        self._interesting_keys = {vars(self.wnd.keys)[k.upper()]: k
                                  for k in INTERESTING_KEYS}
        self._modifiers = {}
        self._current_keys = []  # They are sorted.
        self._event_queue = event_queue

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

        signal = SIGNAL[:, 0]

        # mwe
        FFT = True
        if FFT:

            fft = np.fft.fft(signal, n=signal.shape[0])
            fft.resize((512,))
            #print("fft", fft.shape)
            fft_abs = np.abs(fft)
            signal = fft_abs
        #

        signal = np.stack((signal + 0.1, signal - 0.1), axis=-1)
        signal = signal.flatten()
        vertices = np.dstack([_X, signal, _R, _G, _B])
        vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        vao = self.ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_color')
        vao.render(moderngl.TRIANGLE_STRIP)


# This is a copy of moderngl_window.run_window_config that adds some features:
# - do not care about sys.argv
# - pass args to the config_cls instead
def run_window_config(config_cls: mglw.WindowConfig,
                      event_queue: collections.deque,
                      **config_cls_kwargs) -> None:
    mglw.setup_basic_logging(config_cls.log_level)
    window_cls = mglw.get_local_window_cls(None)

    h, w = map(int, config_cls.window_size)
    size = h, w
    show_cursor = config_cls.cursor

    window = window_cls(
        title=config_cls.title,
        size=size,
        fullscreen=config_cls.fullscreen,
        resizable=config_cls.resizable,
        gl_version=config_cls.gl_version,
        aspect_ratio=config_cls.aspect_ratio,
        vsync=config_cls.vsync,
        samples=config_cls.samples,
        cursor=show_cursor if show_cursor is not None else True,
    )
    window.print_context_info()
    mglw.activate_context(window=window)
    timer = mglw.Timer()
    window.config = config_cls(ctx=window.ctx, wnd=window, timer=timer, event_queue=event_queue)

    # Swap buffers once before staring the main loop.
    # This can trigged additional resize events reporting
    # a more accurate buffer size
    window.swap_buffers()
    window.set_default_viewport()

    timer.start()

    while not window.is_closing:
        current_time, delta = timer.next_frame()

        if window.config.clear_color is not None:
            window.clear(*window.config.clear_color)

        # Always bind the window framebuffer before calling render
        window.use()

        window.render(current_time, delta)
        if not window.is_closing:
            window.swap_buffers()

    _, duration = timer.stop()
    window.destroy()
    if duration > 0:
        mglw.logger.info(
            "Duration: {0:.2f}s @ {1:.2f} FPS".format(
                duration, window.frames / duration
            )
        )

