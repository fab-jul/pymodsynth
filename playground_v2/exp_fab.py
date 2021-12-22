from typing import Sequence, Tuple
from numpy.core.fromnumeric import shape
import random
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


class TODOBUg(mz.Module):
    def setup(self):

        lfo_freq = mz.LFO(mz.Constant(0.5), lo=220, hi=220*mz.FreqFactors.STEP.value**4)
        lfo_freq = mz.Print(lfo_freq)
        self.out = mz.SineSource(lfo_freq)



class Testing(mz.Module):

    def setup(self):
        src = mz.SkewedTriangleSource(
            alpha=mz.Parameter(0.5, lo=0., hi=1., knob="r_mixer_mi"),
            frequency=mz.Parameter(300., key="f",
                                   knob="r_mixer_hi"))

        lp = mz.ButterworthFilter(
            src, f_low=mz.Parameter(5000, lo=1., hi=20000, knob="r_mixer_lo"),
            mode="lp", order=4)

        src = lp

        triggers = mz.PeriodicTriggerSource(bpm=mz.Constant(130))

        env = mz.ADSREnvelopeGenerator(
            total_length=mz.Parameter(5000, key="g"),
        )
        env_at_triggers = mz.TriggerModulator(env, triggers)

        self.out = env_at_triggers * src



class Overdrive(mz.BaseModule):
    
    src: mz.BaseModule
    clip_factor: float = 0.5

    def out_given_inputs(self, clock_signal, src: np.ndarray):
        return np.clip(src, -self.clip_factor, self.clip_factor)/self.clip_factor

class Noise(mz.BaseModule):

    src: mz.BaseModule
    eps: float = 0.1

    def out_given_inputs(self, clock_signal, src):
        return src + np.random.standard_normal(size=src.shape) * self.eps


class Ornstein(mz.BaseModule):

    theta: float = 10.
    eps: float = 0.03

    def setup(self):
        self._last_value = mz.Stateful(0.)

    def out(self, clock_signal: mz.ClockSignal):
        y0 = self._last_value
        output = np.random.standard_normal(size=clock_signal.shape) * self.eps
        output[0] += y0
        dt = clock_signal.ts[1, 0] - clock_signal.ts[0, 0]
        exp_decay = (1-self.theta * dt) ** np.arange(clock_signal.num_samples-1, -1, -1)
        output = np.cumsum(output * (1-self.theta*dt), axis=0) #* exp_decay[:, np.newaxis]
#        for i in range(1, clock_signal.num_samples):
#            output[i, :] += (1-self.theta * dt) * output[i-1, :]
        self._last_value = output[-1, :]
        return output * 0.1

class NicStein(mz.BaseModule):

    src: mz.BaseModule
    eps: float = 0.01

    def out_given_inputs(self, clock_signal: mz.ClockSignal, src):
        rans = np.random.standard_normal(size=clock_signal.shape) * self.eps
        rans = np.cumsum(rans, axis=0)
        rans = rans - np.linspace(0, rans[-1, 0], num=clock_signal.num_samples)[:, np.newaxis]
        return rans + src
                                

class Nick(mz.Module):

    bpm: int = 130

    def setup(self):
        
        bpm = mz.Constant(self.bpm)

        length = mz.Constant(5000.)
        base_freq = 420.
        freq = base_freq * PiecewiseLinearEnvelope(
            xs=[0., 0.2, 0.4, 0.7],
            ys=[1., 0.7, 0.4, 0.1],
            length=length)
        src = SignalWithEnvelope(
            src=mz.TimeIndependentSineSource(frequency=freq),
            env=mz.ADSREnvelopeGenerator(total_length=length*2))
        trigger = mz.PeriodicTriggerSource(bpm)
        kick_trigger = trigger
        kick = mz.TriggerModulator(src, triggers=trigger)
        de = mz.DelayElement(time=2, feedback=0.9, hi_cut=0.7)
        kick = mz.SimpleDelay(kick, de, mix=0.4)
        de = mz.DelayElement(time=5, feedback=0.6, hi_cut=0.7)
        kick = mz.SimpleDelay(kick, de, mix=0.2)

        # sNaRe

        freq = 320 * PiecewiseLinearEnvelope(
            xs=[0.,  0.1, 0.3, 0.5, 0.9],
            ys=[0.3, 0.4, 0.9, 0.4, 0.3],
            length=length)

        src = mz.TimeIndependentSineSource(frequency=freq)
        src = Noise(mz.Constant(0.), eps=0.3)
        src = Overdrive(src, 0.8)
        src = Noise(src, eps=0.1)
        src = Overdrive(src, 0.8)
        src = mz.ButterworthFilter(src, f_low=mz.Parameter(13000, key="f"),
                                   mode="lp")
        

        src = SignalWithEnvelope(
            src=src,
            env=mz.ADSREnvelopeGenerator(
                attack=mz.Constant(0.1),
                release=mz.Constant(6),
                hold=mz.Constant(.1),
                total_length=mz.Constant(8000))
            )
        
        trigger = mz.PeriodicTriggerSource(bpm, rel_offset=0.5)

        snare = mz.TriggerModulator(src, triggers=trigger)

        # hI hAt

        freq = 320 * PiecewiseLinearEnvelope(
            xs=[0.,  0.1, 0.3, 0.5, 0.9],
            ys=[0.3, 0.4, 0.9, 0.4, 0.3],
            length=length) 
        src = mz.TimeIndependentSineSource(frequency=freq)
        src = NicStein(src, eps = 0.1)
        src = NicStein(src)

        # src = Overdrive(src, 0.8)

        src = mz.ButterworthFilter(src, f_low = mz.Constant(1000),f_high=mz.Constant(5000),
                                   mode="bp")

        src = SignalWithEnvelope(
            src=src,
            env=mz.ADSREnvelopeGenerator(
                attack=mz.Constant(0.5),
                release=mz.Constant(6),
                sustain=mz.Constant(1),
                hold=mz.Constant(.1),
                total_length=mz.Constant(2000))
            )

        trigger = mz.PeriodicTriggerSource(4 * bpm)

        hi_hat = mz.TriggerModulator(src, triggers=trigger)

        melody_cycler = 120. * mz.FreqFactors.STEP.value ** mz.Cycler(
            #(1, 1, 2, 2, 8, 8,) +
            (1, 1, 0, 0, 0, 0, 1, 1,
             6, 6, 6, 8, 8, 9, 12, 12)
        )
        melody = mz.TriggerModulator(melody_cycler, trigger)
        freq = mz.Hold(melody)
        voice = mz.SkewedTriangleSource(
            alpha=mz.Parameter(0.9, lo=0., hi=1., knob="r_mixer_hi"),
            frequency=freq,
        )
        melody_trigger = trigger
        src = mz.ADSREnvelopeGenerator(attack=mz.Cycler((3, 8)),#mz.Constant(3),
                                       hold=mz.Cycler((1., 4., 8.)),
                                       release=mz.Constant(5),
                                       total_length=mz.LFO(
                                           frequency=mz.Constant(0.01),
                                           lo=1000,hi=20000),
        )
                                       #total_length=mz.Cycler((
                                       #                        4000,
                                       #                        10000,
                                       #                        6000,
                                       #                        8000,
                                       #                        10000,
                                       #                        10000,
                                       #                        8000,
                                       #                        6000,
                                       #                        10000,
                                       #                        4000,
                                       #                        )))
        voice_env = mz.TriggerModulator(src, triggers=melody_trigger)
        voice_with_env = voice * voice_env
        voice_with_env = mz.SimpleDelay(voice_with_env, de, mix=0.2)
        voice = voice_with_env

        voice = mz.ButterworthFilter(
            voice, 
            f_low=mz.Parameter(5000, lo=1., hi=24000, knob="r_mixer_lo", clip=True), mode="lp")

        self.out = hi_hat + kick + snare# + voice


class RandomDropper(mz.BaseModule):

    src: mz.BaseModule
    p: float = 0.25
    drop_to: float = 0.

    def out_given_inputs(self, clock_signal: mz.ClockSignal, src):
        return src * max((np.random.random() > self.p), self.drop_to)


class RandomCycler(mz.BaseModule):

    choices: Sequence[int] = (0, 5, 8)

    def out(self, clock_signal):
        note = random.choice(self.choices)

        output = np.array([note], dtype=clock_signal.get_out_dtype())
        return clock_signal.add_channel_dim(output)


class JustVoice(mz.Module):

    def setup(self):

        bpm = 100

        mz.DrumMachine("""
            kick:   X....X....
            snare:  ..X....X..
            high:   X.X.X..X..
        """,
        kick = )

        melody_cycler = 150. * mz.FreqFactors.STEP.value ** mz.Cycler(
            (0, 0, 0, 3, 6, 8, 4, 3),
        )
        #melody_cycler = mz.Cycler((0.1, 0.5, 0.7))
        trigger = self.monitor << mz.PeriodicTriggerSource(mz.Constant(bpm))
        #melody_cycler = RandomDropper(melody_cycler)
        melody = mz.TriggerModulator(melody_cycler, trigger)
        freq = mz.Hold(melody)
        sine = mz.SineSource(frequency=freq)

        trigger = mz.PeriodicTriggerSource(mz.Constant(bpm*2))
        env = mz.ADSREnvelopeGenerator(attack=mz.Cycler((3, 8)),#mz.Constant(3),
                                       hold=mz.Cycler((1., 4., 8., 20)),
                                       release=mz.Constant(5),
                                       total_length=mz.Constant(9000))
        env = RandomDropper(env, p=0.1, drop_to = 0.2)
        melody = mz.TriggerModulator(env, trigger)
        melody = melody * sine
        melody = Overdrive(melody * 2)/1.3

        freq = NicStein(mz.Constant(0.), eps = 0.3)*80 + 250

        werid_thing = mz.SineSource(frequency=freq)

        werid_thing = Overdrive(werid_thing)

        trigger = mz.PeriodicTriggerSource(mz.Constant(bpm))
        env = mz.ADSREnvelopeGenerator(attack=mz.Cycler((3, 8)),#mz.Constant(3),
                                       hold=mz.Cycler((1., 4., 8., 20)),
                                       release=mz.Constant(5),
                                       total_length=mz.Constant(9000))

        bla = mz.TriggerModulator(env, trigger)

        werid_thing = werid_thing * bla

        weird_thing = mz.ButterworthFilter(werid_thing, f_low=mz.Parameter(1000, knob = "r_mixer_hi"), mode="lp")

        self.out = weird_thing + melody + Nick(bpm)
        return
        melody = mz.TriggerModulator(melody_cycler, trigger)
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



class NeatTunes(mz.Module):
    def setup(self):
        length = mz.constant(5000.)
        base_freq = 420.
        freq = base_freq * piecewiselinearenvelope(
            xs=[0., 0.2, 0.4, 0.7],
            ys=[1., 0.7, 0.4, 0.1],
            length=length)
        src = signalwithenvelope(
            src=mz.timeindependentsinesource(frequency=freq),
            env=mz.adsrenvelopegenerator(total_length=length*2))
        bpm = mz.constant(130)
        trigger = mz.periodictriggersource(bpm)
        kick = mz.triggermodulator(src, triggers=trigger)
        de = mz.delayelement(time=2, feedback=0.9, hi_cut=0.7)
        kick = mz.simpledelay(kick, de, mix=0.4)
        de = mz.delayelement(time=5, feedback=0.6, hi_cut=0.7)
        kick = mz.simpledelay(kick, de, mix=0.2)


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
            (0, 0, 0, 0, 0, 0, 0, 0,))
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
