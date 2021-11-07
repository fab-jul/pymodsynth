import inspect

import numpy as np


def parse(prg, definitions="v2"):
    locals_def = {}
    exec(f"import {definitions}", {}, locals_def)
    module = locals_def[definitions]

    # Extract all classes with __call__.
    classes = {}
    for k in dir(module):
        v = getattr(module, k)
        if isinstance(v, type) and hasattr(v, "__call__"):
            classes[k] = v
            print(k)
            sig = inspect.signature(v.__call__)
            for name, param in sig.parameters.items():
                if param.annotation == np.ndarray:
                    print(name, "SIGNAL")
                else:
                    print(name, param.annotation)

            print(sig)

    # Turn program into callable
    def program(clock):
        locals_prg = {"clock": clock}
        exec(prg, classes, locals_prg)
        return locals_prg["o"]

    return program


class Parameter:

    def __get__(self):
        return self.val


def parameter():
    pass


_TEST_PRG = """
def program():
    sin = SignalGenerator()
    sin_freq = parameter()
    o = sin(clock, freq=sin_freq)

    sig = f1(s1(clock))
    carrier_freq = parameter()
    o = m1(sin, carrier=carrier_freq)
"""

programm()

def test():
    cmd = "r"
    while 1:
        if cmd == "r":
            program = parse(_TEST_PRG)
            print(program(np.arange(10)))
        cmd = input()


if __name__ == '__main__':
    test()

