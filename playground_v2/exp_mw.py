
import mz
P = mz.Parameter



class SignalWithEnvelope(mz.BaseModule):

    src: mz.BaseModule
    env: mz.BaseModule

    def out(self, clock_signal: mz.ClockSignal):
        # This defines the length!
        env = self.env(clock_signal)
        fake_clock_signal = clock_signal.change_length(env.shape[0])
        src = self.src(fake_clock_signal)
        return env * src


class BabiesFirstSynthie(mz.Module):
    def setup(self):
        base_frequency = mz.Parameter(220, key='f')
        lfo = mz.SineSource(frequency=mz.Parameter(0.66, key='l', lo=0.1, hi=60, clip=True))
        dancing_triangle = mz.SkewedTriangleSource(frequency=base_frequency,
                                                        alpha=mz.lift(lfo))
        low_hum = mz.SineSource(frequency=mz.Parameter(66, key='b'))
        self.out = dancing_triangle + low_hum * 0.5

class Test(mz.Module):
    def setup(self):
        # self.lfo = mz.SineSource(frequency=(mz.Parameter(3) + 1)/2)
        # self.src = mz.SineSource(frequency=self.lfo * 440)
        # self.out = self.src
        self.sick = mz.SkewedTriangleSource(frequency=P(220), alpha=mz.lift(mz.SineSource(P(1))))
        self.out = self.sick
        # self.out = mz.ButterworthFilter(self.sick,
        #                                 f_low=P(10, key='k', lo=1, hi=40000, clip=True),
        #                                 f_high=P(10000, key='l', lo=1, hi=40000, clip=True),
        #                                 mode="bp")


