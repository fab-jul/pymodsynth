import mz


class SimpleSine(mz.Module):

    def setup(self):
        self.out = mz.SineSource(frequency=mz.Parameter(144, key="f"))


class _SimpleNoteSampler(mz.Module):

    src: mz.Module
    env: mz.Module

    def out_given_inputs(self, _, src, env):
        return src * env


array = ...


class SineSourceSample(mz.BaseModule):

    length: mz.Parameter = mz.Constant(512)

    def setup(self):
        self.sine = mz.SineSource(frequency=mz.Constant(220))

    def out_given_inputs(self, clock_signal, length: float):
        # Clock signal starts at 0 and is `length` long.
        clock_signal_for_sine = clock_signal.of_length(length)
        return self.sine(clock_signal_for_sine)


class TriggerModulator(...):

    def out(self, clock_signal):
        pass


class Testing(mz.Module):

    def setup(self):
        self.seq = mz.MelodySequencer(
            bpm=mz.Constant(120),
            pattern=mz.Pattern([0, 0, 0, 1], note_value=1/16.)
        )

        self.sampler = _SimpleNoteSampler(
            src=mz.SineSource(frequency=mz.Constant(220)),# ** (mz.FreqFactors.STEP.value * self.seq.output("note"))),
            env=mz.ADSREnvelopeGenerator(hold=self.seq.output("hold"))
        )

        self.out = mz.TriggerModulator(sampler=self.sampler, triggers=self.seq.output("triggers"))



class SimpleSynthVoice(mz.Module):

    def setup(self):
        # CV and gate output
        self.seq = mz.StepSequencer()
        self.voice = mz.SineSource(frequency=220 ** (mz.FreqFactors.STEP.value * self.seq.output("cv")))
        self.voice = mz.ButterworthFilter(self.voide)



def _run():
    t = Testing()
    clock = mz.Clock()
    signal = clock()
    t(signal)


if __name__ == "__main__":
    _run()