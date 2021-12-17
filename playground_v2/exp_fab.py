from typing import Sequence, Tuple
import collections
import dataclasses
import scipy.signal
import numpy as np
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


# class TriggerModulator(...):
# 
#     def out(self, clock_signal):
#         pass


class SignalWithEnvelope(mz.BaseModule):

    src: mz.BaseModule
    env: mz.BaseModule

    def out(self, clock_signal: mz.ClockSignal):
        # This defines the length!
        env = self.env(clock_signal)
        fake_clock_signal = clock_signal.change_length(env.shape[0])
        src = self.src(fake_clock_signal)
        return env * src


class PiecewiseLinearEnvelope(mz.Module):

    xs: Sequence[float]
    ys: Sequence[float]
    length: mz.SingleValueModule = mz.Constant(500.)

    def setup(self):
        assert len(self.xs) == len(self.ys)
        assert max(self.xs) <= 1.
        assert min(self.xs) >= 0.
        if self.xs[-1] < 1.:
            self.xs = (*self.xs, 1.)
            self.ys = (*self.ys, self.ys[-1])

    def out_given_inputs(self, clock_signal: mz.ClockSignal, length: float):
        length: int = round(length)
        prev_x_abs = 0
        prev_y = 1.
        pieces = []
        for x, y in zip(self.xs, self.ys):
            x_abs = round(x * length)
            if x_abs - prev_x_abs <= 0:
                prev_y = y
                continue
            pieces.append(np.linspace(prev_y, y, x_abs - prev_x_abs))
            prev_x_abs = x_abs
            prev_y = y
        env = np.concatenate(pieces, 0)
        env = clock_signal.add_channel_dim(env)
        env = clock_signal.pad_or_truncate(env, pad=env[-1, 0])
        return env

# TODO: why does stuff need to be hashable for math

class Testing(mz.Module):

    def setup(self):
        length = mz.Constant(2000.)
        base_freq = 520.
        freq = base_freq * PiecewiseLinearEnvelope(
            xs=[0., 0.2, 0.4, 0.7],
            ys=[1., 0.7, 0.4, 0.1],
            length=length)
        src = SignalWithEnvelope(
            src=mz.TimeIndependentSineSource(frequency=freq),
            env=mz.ADSREnvelopeGenerator(total_length=length*2))
        bpm = mz.Constant(130)
        trigger = mz.PeriodicTriggerSource(bpm)
        kick = mz.TriggerModulator(src, triggers=trigger)
        de = mz.DelayElement(time=4, feedback=0.6, hi_cut=0.4)
        kick = mz.SimpleDelay(kick, de, mix=0.2)

        melody_cycler = 150. * mz.FreqFactors.STEP.value ** mz.Cycler((0, 4, 7, 3))
        melody = mz.TriggerModulator(melody_cycler, trigger)
        freq = mz.Hold(melody)

        voice = mz.SkewedTriangleSource(
            alpha=mz.Parameter(0.9, lo=0., hi=1., key="f",knob="r_mixer_hi"),
            frequency=freq,
        )

        voice = mz.ButterworthFilter(
            voice,
            f_low=mz.Parameter(100., lo=10, hi=4000., knob="r_mixer_mi"),
            mode="lp")

        melody_trigger = mz.PeriodicTriggerSource(bpm*2)
        src = mz.ADSREnvelopeGenerator(attack=mz.Constant(3),
                                       hold=mz.Constant(1.),
                                       release=mz.Constant(5),
                                       total_length=mz.Constant(10000))
        voice_env = mz.TriggerModulator(src, triggers=melody_trigger)

        voice_with_env = voice * voice_env
        voice_with_env = mz.SimpleDelay(voice_with_env, de, mix=0.2)

        self.out = voice_with_env + kick*0.9

        return

        self.out = kick

        #self.sine = mz.SineSource(frequency=mz.Parameter(100)) * 0.25
        #self.out = self.out + self.sine
        return

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
