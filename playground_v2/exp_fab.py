
import mz

class SimpleSine(mz.Module):

    def setup(self):
        self.out = mz.SineSource(frequency=mz.Parameter(144, key="f"))

