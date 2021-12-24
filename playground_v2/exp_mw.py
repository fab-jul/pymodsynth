
import mz


class Test(mz.Module):
    def setup(self):
        self.ulfo = mz.SineSource(frequency=(mz.Parameter(0.01)+1)*0.5)
        self.lfo = mz.SineSource(frequency= mz.Parameter(0.1) + self.ulfo * 0.9)
        self.src = mz.SineSource(frequency=(self.lfo + 1)/2 * 600)
        self.out = self.src




