# NEEDS
# pip install torch
# pip install ninja

import functools
import os
import numpy as np
from torch.utils.cpp_extension import load

from mz.experimental import subplots
from mz import base
from mz import helpers


# We load the CPP only on demand, as it involves a JIT ninja compilation step.
@functools.lru_cache()
def _cpp_backend():
  print("Loading CPP backend...")
  pybind_backend_dir = os.path.dirname(os.path.realpath(__file__))
  src_dir = os.path.join(pybind_backend_dir, 'src')
  return load(
      name="cpp_backend",
      sources=[os.path.join(src_dir, "backend.cpp")],
      verbose=True)


# ------------------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------------------


def _quantize(a: np.ndarray, lo: float, hi: float, num: int):
    a = np.clip(a, lo, hi)
    return np.round((a - lo) / (hi - lo) * (num - 1)).astype(np.int32)


class Butterworth(base.Module):

  src: base.Module
  cutoffs: base.Module = base.Constant(200.)

  # This is only used for bp, in which case it's the high cutoff,
  # and `cutoffs` is used as the low cutoff.
  cutoffs_bp_hi: base.Module = base.Constant(1000.)

  mode: str = "lp"  # One of lp, hp, bp.
  order: int = 4

  def setup(self):

      if self.mode == "bp":
        raise NotImplementedError("Need High Pass for bandpass!")
      elif self.mode == "hp":
        raise NotImplementedError("High Pass not implemented yet!")
      # TODO: HP shoul use same number of coeffs,
      # but backend not implemented yet.
      elif self.mode in ("lp", "hp"):  
        if self.order % 2 != 0:
          raise ValueError
        n = self.order // 2
        self._w0 = np.zeros((n,), np.float32)
        self._w1 = np.zeros((n,), np.float32)
        self._w2 = np.zeros((n,), np.float32)
      else:
        raise ValueError(f"Invalid mode == {self.mode}!")

      # Will be created whenever num_samples changes.
      self._result_buf = None

      # TODO: Only for book keeping, disable
      self._timer = helpers.Timer()

  def out(self, clock_signal: base.ClockSignal):
      src = self.src(clock_signal)
      cutoffs = self.cutoffs(clock_signal)
      # Validation.
      if cutoffs.shape != src.shape:
          raise ValueError("Cutoffs and source must have the same shape!")
      if len(src.shape) != 1:
          raise ValueError(f"Invalid src shape = {src.shape}!")
      # Create output buffer if needed.
      if self._result_buf is None or self._result_buf.shape != src.shape:
        print("Creating result_buf...")
        self._result_buf = np.empty(src.shape, np.float32)
      # Run filter.
      if self.mode == "bp":
        # Fetch highs.
        cutoffs_hi = self.cutoffs_bp_hi(clock_signal)
        raise NotImplementedError  # Swee above.
      else:
        with self._timer("Time in C++"):
          _cpp_backend().butterworth_lowpass_or_highpass(
            src, cutoffs, self._result_buf,
            self._w0, self._w1, self._w2,
            clock_signal.sample_rate, self.mode == "lp")
      return self._result_buf
