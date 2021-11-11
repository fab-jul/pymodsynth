"""Write and reads params files.

Example file:

([^\s]+)
lfo.frequency step=0.001 range=[-inf, 10.] key=f
lfo.amplitude step=0.01 range=[1., 10.]

Syntax:

<name> step=<step> range=[<range_lo>, <range_hi>] [key=<key>]
- <range_lo> can be a float or "-inf"
- <range_hi> can be a float or "inf"
- key= is optional.

"""
import os
import re
import tempfile
import typing
from typing import Optional, NamedTuple, Iterable

_RE = re.compile(
    r"([^\s]+)\s+step=([\d\.]+)\s+range=\[([\d\.]+|-inf),\s*([\d\.]+|inf)\](\s+key=(.+))?")


# TODO: Params should expose their default spec.

class ParamSpec(NamedTuple):
    param_name: str
    step: float = 1/10.
    # Limits of the parameter.
    lo: float = float('-inf')
    hi: float = float('inf')
    key: Optional[str] = None

    def to_text(self):
        output = [
            self.param_name, f"step={self.step}",
            f"range=[{self.lo},{self.hi}]",
        ]
        if self.key:
            output.append(f"key={self.key}")
        return " ".join(output)

    @classmethod
    def from_text(cls, line: str):
        m = _RE.search(line)
        if not m:
            raise ValueError("Invalid line: {line}")
        return cls(
            param_name=m.group(1),
            step=float(m.group(2)),
            lo=float(m.group(3)),
            hi=float(m.group(4)),
            key=m.group(6))


# TODO: params should expose their range!
def write_params(params: Iterable[ParamSpec], file_path):
    with open(file_path, "w") as f:
        for p in params:
            f.write(p.to_text() + "\n")


def parse_params(file_path) -> typing.Sequence[ParamSpec]:
    params = []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            params.append(ParamSpec.from_text(line))
    return params


def _test():
    p1 = ParamSpec.from_text(
        "lfo.frequency step=0.001 range=[-inf, 10.] key=f")
    p2 = ParamSpec.from_text(
        "lfo.amplitude step=0.01 range=[1., 10.]")

    tmp = tempfile.mktemp()
    write_params([p1, p2], tmp)
    with open(tmp, "r") as f:
        print(f.read())
    p = parse_params(tmp)
    assert len(p) == 2
    os.remove(tmp)


if __name__ == '__main__':
    _test()