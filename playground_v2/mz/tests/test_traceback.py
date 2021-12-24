from mz import base
from mz import sources
import numpy as np


class Buggy(base.Module):

    def out(self, _):
        return np.ones((4, 1))


class Example(base.Module):

    def setup(self):
        f = base.Constant(220)# Buggy()
        a = sources.SineSource(f)
        b = a * 2 + 3
        c = sources.TimeIndependentSineSource(b)
        self.out = c


if __name__ == "__main__":
    a = Example()
    clock = base.Clock()
    _ = a(clock())
