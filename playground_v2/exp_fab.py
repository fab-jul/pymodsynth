from typing import Sequence, Tuple
import soundfile as sf
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

class SamplePlayer(mz.BaseModule):

    wav_file_p: str

    def setup(self):
        self.data, samplerate = sf.read(self.wav_file_p)

    def out(self, clock_signal):
        return self.data[:, :clock_signal.num_channels]


class Testing(mz.Module):
    def setup(self):

        lfo_freq = mz.LFO(mz.Constant(0.5), lo=220, hi=220*mz.FreqFactors.STEP.value**4)
        lfo_freq = mz.Print(lfo_freq)
        self.out = mz.SineSource(lfo_freq)

class NeatTunes(mz.Module):
    def setup(self):
        length = mz.Constant(5000.)
        base_freq = 420.
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
        de = mz.DelayElement(time=2, feedback=0.9, hi_cut=0.7)
        kick = mz.SimpleDelay(kick, de, mix=0.4)
        de = mz.DelayElement(time=5, feedback=0.6, hi_cut=0.7)
        kick = mz.SimpleDelay(kick, de, mix=0.2)

        hi_sample = SamplePlayer("samples/HatO_SP_08.wav")
        hi_sample = SamplePlayer("samples/HatC_SP_20.wav")
        trigger_hi = mz.PeriodicTriggerSource(bpm*2)
        hi = mz.TriggerModulator(hi_sample, triggers=trigger_hi)
        #kick = mz.Reverb(kick, p=mz.Constant(0.1))
        drums = kick + hi*mz.Parameter(0.5, 0., 1., knob="fx2_2")
#        drums = mz.ButterworthFilter(
#            drums, f_low=mz.Constant(1),
#            f_high=mz.Parameter(5000, lo=1., hi=10000, key="f", clip=True), mode="hp")

        melody_cycler = 120. * mz.FreqFactors.STEP.value ** mz.Cycler(
            #(12,)*8 + (8,)*8 + (5,)*8 + (0, )*4 + (1, 0, 1, 5))
            (12,)*8 + (8,)*8 + (5,)*8 + (0, )*4 + (5,)*8 + (8,)*8 + (12,)*8
        )
        melody = mz.TriggerModulator(melody_cycler, trigger)
        freq = mz.Hold(melody)
        voice = mz.SkewedTriangleSource(
            alpha=mz.Parameter(0.9, lo=0., hi=1., knob="r_mixer_hi"),
            frequency=freq,
        )
        melody_trigger = trigger_hi
        src = mz.ADSREnvelopeGenerator(attack=mz.Cycler((3, 8)),#mz.Constant(3),
                                       hold=mz.Cycler((1., 4., 8., 20)),
                                       release=mz.Constant(5),
                                       total_length=mz.Cycler((20000, 26000, 20000)))
        voice_env = mz.TriggerModulator(src, triggers=melody_trigger)
        voice_with_env = voice * voice_env
        voice_with_env = mz.SimpleDelay(voice_with_env, de, mix=0.2)
        voice = voice_with_env

        voice = mz.ButterworthFilter(
            voice, 
            f_low=mz.Parameter(5000, lo=1., hi=24000, knob="r_mixer_lo", clip=True), mode="lp")

        # Bassline

        trigger = mz.PeriodicTriggerSource(bpm)
        bassline_cycler = 220./2 * mz.FreqFactors.STEP.value ** mz.Cycler(
            (0, 0, 0, 0, 0, 0, 0, 12,))
        melody = mz.TriggerModulator(bassline_cycler, trigger)
        freq = mz.Hold(melody)
        bassline = mz.SkewedTriangleSource(
            alpha=mz.Parameter(0.5, lo=0., hi=1., knob="r_mixer_mi"),
            frequency=freq,
        )
        bassline = mz.ButterworthFilter(
            bassline, 
            f_low=mz.Parameter(5000, lo=1., hi=24000, knob="l_mixer_mi", clip=True), mode="lp")
        #melody_trigger = trigger_hi
        src = mz.ADSREnvelopeGenerator(attack=mz.Cycler((1, 1)),#mz.Constant(3),
                                       hold=mz.Cycler((20., 24.)),
                                       release=mz.Cycler((40, 20, 25)),
                                       total_length=mz.Cycler((#15000,
                                                               #25000,
                                                               60000,
                                                               45000,
                                                               50000,)))
        bassline_env = mz.TriggerModulator(src, triggers=trigger)
        bassline_with_env = bassline * bassline_env
        #bassline_with_env = mz.SimpleDelay(voice_with_env, de, mix=0.2)
        bassline = bassline_with_env
        #bassline = mz.SimpleDelay(bassline, de, mix=0.4)
        #bassline = mz.Reverb(bassline)

        self.out = (drums + 
                    voice*mz.Parameter(0.4, 0., 1., knob="fx2_1") +
                    bassline*0.2
                    )



class Testing1(mz.Module):

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

        melody_trigger = mz.PeriodicTriggerSource(bpm*2)
        melody_cycler = 150. * mz.FreqFactors.STEP.value ** mz.Cycler((0, 0, 0, 1))
        melody = mz.TriggerModulator(melody_cycler, trigger)
        freq = mz.Hold(melody)

        voice = mz.SkewedTriangleSource(
            alpha=mz.Parameter(0.9, lo=0., hi=1., key="f",knob="r_mixer_hi"),
            frequency=freq,
        )

        # TODO: some thing like sine but linear

        voice_a = mz.ButterworthFilter(
            voice,
            f_low=mz.Constant(2000),
            mode="lp")

        voice_b = mz.ButterworthFilter(
            voice,
            f_low=mz.Constant(6500),
            mode="lp")

        mask = (1 + mz.SineSource(frequency=mz.Constant(0.1)))/2
        voice = voice_a * mask + voice_b * (1-mask)

        src = mz.ADSREnvelopeGenerator(attack=mz.Cycler((3, 8)),#mz.Constant(3),
                                       hold=mz.Cycler((1., 4., 8., 20)),
                                       release=mz.Constant(5),
                                       total_length=mz.Cycler((10000, 16000, 10000)))
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
