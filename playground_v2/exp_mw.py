
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


class Test(mz.Module):
    def setup(self):
        # self.lfo = mz.SineSource(frequency=(mz.Parameter(3) + 1)/2)
        # self.src = mz.SineSource(frequency=self.lfo * 440)
        # self.out = self.src
        self.sick = mz.SkewedTriangleSource(frequency=P(220), alpha=mz.lift(mz.SineSource(P(1))))
        self.out = mz.ButterworthFilter(self.sick,
                                        f_low=P(10, key='k', lo=1, hi=40000, clip=True),
                                        f_high=P(10000, key='l', lo=1, hi=40000, clip=True),
                                        mode="bp")

