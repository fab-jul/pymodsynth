
import mz




class Test(mz.Module):
    def setup(self):
        bpm = mz.Constant(160)
        kick_pattern = mz.Pattern(pattern=(1, 1, 1, 1), note_values=1/4)
        kick_env = mz.KickEnvelope()
        kick_track = mz.Track(bpm=bpm,
                              pattern=kick_pattern,
                              env_gen=kick_env) >> mz.Collect("kick track")

        note_base_freq = mz.Constant(220)
        NL = 1/8
        NV = 1/4

        note_pattern1 = mz.NotePattern(pattern=(1, 2, 3), note_lengths=(NL, NL*3, NL/3), note_values=NV)
        note_track1 = mz.NoteTrack(bpm=bpm, note_pattern=note_pattern1, carrier_base_frequency=note_base_freq * 1) >> mz.Collect("nt1")
        note_pattern2 = mz.NotePattern(pattern=(1, 0, 0, 0, 0), note_lengths=(NL, NL, NL, NL, NL), note_values=NV)
        note_track2 = mz.NoteTrack(bpm=bpm, note_pattern=note_pattern2, carrier_base_frequency=note_base_freq * 1.333)
        note_pattern3 = mz.NotePattern(pattern=(1, 0, 0, 0, 0, 0, 0), note_lengths=(NL, NL, NL, NL, NL, NL, NL), note_values=NV)
        note_track3 = mz.NoteTrack(bpm=bpm, note_pattern=note_pattern3, carrier_base_frequency=note_base_freq * 1.999)

        notes = mz.ButterworthFilter((note_track1 )/3, f_low=mz.Constant(1), f_high=mz.Constant(1000), mode="bp") >> mz.Collect("filtered")
        self.out = kick_track + notes




if __name__ == "__main__":
    mz.plot_module(Test, num_frames=10)



