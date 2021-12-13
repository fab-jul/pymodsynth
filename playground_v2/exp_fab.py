
import mz

class SimpleSine(mz.Module):

    def setup(self):
        self.out = mz.SineSource(frequency=mz.Parameter(144, key="f"))


class _SimpleNoteSampler(mz.Module):

    src: mz.Module
    env: mz.Module

    def out_given_inputs(self, _, src, env):
        return src * env


class Testing(mz.Module):

    def setup(self):
        self.seq = mz.MelodySequencer(
            bpm=mz.Constant(120),
            pattern=mz.Pattern([1, 0, 1, 0])
        )

        self.sampler = _SimpleNoteSampler(
            src=mz.SineSource(frequency=220 ** (mz.FreqFactors.STEP.value * self.seq.output("note"))),
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