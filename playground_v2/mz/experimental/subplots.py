import matplotlib.pyplot as plt
import numpy as np


class Subplots:

    def __init__(self, nrows=1, ncols=1, width=5, height=5):
        self.fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(width * ncols, height * nrows))
        self.axs = np.array(axs).reshape((nrows*ncols))
        self.i = -1

    @property
    def current_ax(self):
        if self.i == -1:
            self.i = 0
        return self.axs[self.i]

    def next_ax(self):
        self.i += 1
        return self.current_ax
