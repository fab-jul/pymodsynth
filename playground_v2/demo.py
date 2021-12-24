import mz

class DemoModule(mz.Module):
    
    def setup(self):
        self.out = mz.SineSource()