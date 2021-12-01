
class Module:

    def __call__(self, clock_signal) -> np.ndarray:
        return self.out(clock_signal)

    def out(self, clock_signal) -> np.ndarray:
        raise NotImplementedError


class LambdaModule(Module):

    def __init__(self, out):
        self.out = out


MultiOutput = dict


class MultiOutputModule(Module):

    def outputs(self, key) -> Module:
        return LambdaModule(out=lambda clock_signal: self.multi_out(clock_signal)[key])

    def out(self, clock_signal) -> np.ndarray:
        raise NotImplementedError

    def multi_out(self, clock_signal) -> MultiOutput:
        return ...


class StepSequencer(MultiOutputModule):

    def __init__(self,
                 bmp: Module = Constant(130),
                 ):
        pass

    def multi_out(self, clock_signal):
        triggers = ...
        frequencies = ...
        return MultiOutput(frequencies=frequencies, triggers=triggers)


class Root(Module):

    def __init__(self):
        self.step = StepSequencer(...)
        self.envelopes = ...
        self.sine = ...
        self.sine_bass = ...
        envelopes = self.envelopes(self.step.outputs("triggers"))
        tone = self.sine(self.step.outputs("frequencies"))
        self.out = tone * envelopes + self.sine_bass







