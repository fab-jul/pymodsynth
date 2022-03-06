
from mz import base
from mz import envelopes
from mz import filters
from mz import sources


class Kick(base.Module):

    triggers: base.Module

    def setup(self):
        length = base.Constant(5000.)
        base_freq = 320.
        freq = base_freq * sources.PiecewiseLinearEnvelope(
            xs=[0., 0.2, 0.4, 0.7],
            ys=[0.9, 0.5, 0.3, 0.1],
            length=length) #+ mz.LFO(mz.Constant(0.1), lo=100, hi=400)
        src = sources.SignalWithEnvelope(
            src=sources.TimeIndependentSineSource(frequency=freq),
            env=envelopes.ADSREnvelopeGenerator(total_length=length*2))
        kick = sources.TriggerModulator(src, triggers=self.triggers)

        de = filters.DelayElement(time=2, feedback=0.9, hi_cut=0.7)
        kick = filters.SimpleDelay(kick, de, mix=0.3)
        de = filters.DelayElement(time=10, feedback=0.6, hi_cut=0.4)
        kick = filters.SimpleDelay(kick, de, mix=0.2)
        
        self.out = kick
