# NEEDS
# pip install torch
# pip install ninja

import os
from torch.utils.cpp_extension import load

pybind_backend_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.join(pybind_backend_dir, 'src')
src = load(
  name="src",
  sources=[os.path.join(src_dir, "test.cpp")],
  verbose=True)


import numpy as np
a = np.ones((128,), np.float32)

b = src.add(a,a)
print(b)
