"""ModernGL based live graph implementation."""

import threading

import moderngl
import moderngl_window as mglw
import numpy as np


NUM_SAMPLES = 512
# Shared among threads. Overwrite to update view.
# NOTE: the 1 is because we have a one channel output.
SIGNAL = np.zeros((512, 1), np.float32)

# Static vectors needed to draw signal.
_X = np.linspace(-1.0, 1.0, NUM_SAMPLES)
_R = np.ones(NUM_SAMPLES)
_G = np.zeros(NUM_SAMPLES)
_B = np.zeros(NUM_SAMPLES)


class QuitException(Exception):
    """Raised to exit the render loop. Needs to be caught."""
    pass


class RandomPlot(mglw.WindowConfig):
    gl_version = (3, 3)

    QUIT_EVENT: threading.Event = None

    def render(self, time, frametime):
        if self.QUIT_EVENT and self.QUIT_EVENT.is_set():
            print("Window received quit event, closing...")
            raise QuitException()

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

        vertices = np.dstack([_X, SIGNAL[:, 0], _R, _G, _B])
        vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        vao = self.ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_color')
        vao.render(moderngl.LINE_STRIP)
