
import mz




class Test(mz.Module):
    def setup(self):
        bpm = mz.Constant(2000)
        kick_pattern = mz.Pattern(pattern=(1, 2, 1, 3), note_values=1/4)
        kick_env = mz.KickEnvelope()
        kick_track = mz.Track(bpm=bpm,
                              pattern=kick_pattern,
                              env_gen=kick_env) #>> mz.Collect("kick track")

        note_base_freq = mz.Constant(220)
        NL = 1/8
        NV = 1/4

        note_pattern1 = mz.NotePattern(pattern=(1, 2, 3), note_lengths=(NL, NL*3, NL*2), note_values=1/3)
        note_track1 = mz.NoteTrack(bpm=bpm, note_pattern=note_pattern1, carrier_base_frequency=note_base_freq * 1) #>> mz.Collect("nt1", "notes")
        note_pattern2 = mz.NotePattern(pattern=(1, 0, 0, 0, 0), note_lengths=(NL, NL, NL, NL, NL), note_values=NV)
        note_track2 = mz.NoteTrack(bpm=bpm, note_pattern=note_pattern2, carrier_base_frequency=note_base_freq * 1.333) #>> mz.Collect("nt2", "notes")
        note_pattern3 = mz.NotePattern(pattern=(1, 0, 0, 0, 0, 0, 0), note_lengths=(NL, NL, NL, NL, NL, NL, NL), note_values=NV)
        note_track3 = mz.NoteTrack(bpm=bpm, note_pattern=note_pattern3, carrier_base_frequency=note_base_freq * 1.999)

        note_sum = (note_track1 + 0)/2
        #notes = mz.ButterworthFilter(note_sum, f_low=mz.Constant(1), f_high=mz.Constant(1000), mode="bp")
        self.out = kick_track + note_sum




if __name__ == "__main__":
    mz.plot_module(Test, start_frame=0, num_frames=5, cluster=False)



