import numpy as np
from mz import base
from mz import sources


def test_hold():
    for inp, expected_out in (
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]),
        ([3, 0, 0, 4, 0, 0, 7, 0, 0, 10],
         [3, 3, 3, 4, 4, 4, 7, 7, 7, 10]),
        ([0, 0, 0, 4, 0, 0, 7, 0, 0, 10],
         [5, 5, 5, 4, 4, 4, 7, 7, 7, 10]),
        ([3, 0, 0, 4, 0, 0, 7, 0, 0, 0],
         [3, 3, 3, 4, 4, 4, 7, 7, 7, 7]),
    ):
        hold = sources.Hold(base.Constant(1))
        hold.prev_value = 5.
        out = hold.out_given_inputs(base.ClockSignal.test_signal(),
                                    src=np.array(inp).reshape(-1, 1))
        assert out[:, 0].tolist() ==  expected_out
                        