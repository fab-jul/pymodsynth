
# TODO: autoformat
import pprint
from typing import Iterable, Sequence

from playground.crazy import Module

EXAMPLE_STEP_SEQ = """

# Melody: values in 1-12
melody: 1 2 8 12 8 12 1

# Step count: values in 1-infty 
steps:  1 1 1 1  1 1  1

# Gate mode:
# - H: hold for the step count,    XXXXXX
# - E: Sound for each step count   X X X
# - F: Sound for first step count  X
# - S: Skip, no souond
gate: F F F F F F F F 

# Slide / Skip
# TODO

"""


def step_seq_parser(lines: Lines):
    groups_parser = (
            stripped | without_line_comments | grouped)
    groups = groups_parser(lines)
    for g in groups:
        assert len(g) == 1, g
        g, = g
        label, values = g.split(":")
        values = values.strip().split()


def ascii_configurable(from_ascii):
    pass


@ascii_configurable
class StepSequencer(Module):
    melody: Sequence[str]
    steps: Sequence[str]
    gate: Sequence[str] = []


env = """
x
 x
   x
       x
       x
       x
       x
       x
       x
x 
"""

def test():
    # TODO: can we just do it inline actually?!?
    s = StepSequencer.tracking_file()
    s = Envelope(env)


Lines = Iterable[str]


class _Parser:

    def __init__(self, f):
        self.f = f

    def __or__(self, other_parser):
        if not isinstance(other_parser, _Parser):
            raise TypeError
        return _Parser(lambda xs: other_parser.f(self.f(xs)))

    def __call__(self, xs):
        return self.f(xs)


def parser(f):
    return _Parser(f)


# Helper parsers
@parser
def stripped(lines: Lines) -> Lines:
    return (l.strip() for l in lines)


@parser
def without_line_comments(lines: Lines, comment_sym="#") -> Lines:
    for l in lines:
        if not l.startswith(comment_sym):
            yield l


@parser
def grouped(lines: Lines) -> Iterable[Lines]:
    current_group = []
    for l in lines:
        if not l:  # Empty line:
            if current_group:
                yield current_group
                current_group = []
        else:
            current_group.append(l)
    if current_group:
        yield current_group


as_list = parser(list)





def main():
    lines = EXAMPLE_STEP_SEQ.split("\n")
    step_seq_parser(lines)


if __name__ == '__main__':
    main()