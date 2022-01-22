import playground_v2.mz as mz


class BabiesFirstSynthie(mz.Module):
    def setup(self):
        self.base_frequency = mz.Parameter(220, key='f')
        self.lfo = mz.SineSource(frequency=mz.Constant(0.66))
        self.dancing_triangle = mz.SkewedTriangleSource(frequency=self.base_frequency,
                                                        alpha=mz.lift(self.lfo))
        self.low_hum = mz.SineSource(frequency=mz.Parameter(44), key='b')
        self.out = self.dancing_triangle + self.low_hum * 0.5


