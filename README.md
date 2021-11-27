# pymodsynth

Modular Synthesizer in Python

Core Idea: Modular Synthesizers are "just" function generators with a nice user interface. Can we build something similar in Python?


## Next Steps

- Kick and Snare (drum machine)
- Filters (low/mi/high - buttersworth)
- Filters (decay / reverb etc.)
- Record with microphone
- Input samples
- Record our sound
- Visual feedback in the GUI / explore keyboard and mouse again


## API: Thoughts

Terms:
- `sample`: A single time-step in the output signal.
- `frame`: A collection of e.g. 2048 samples. We generate frames in each event loop step.

### Parameters v2


Should be pluggable anywhere in the module hiararchy:


```python


class Root(Module):
  
  def __init__(self):
    self.tmp = ...


```






### Ports
  
We should have a module with multiple outputs:

```python
class Module:

  def __call__(self, clock_signal) -> np.ndarray:
    return self.out(clock_signal)

  def out(self, clock_signal) -> np.ndarray:
    raise NotImplementedError


class LambdaModule(Module):

  def __init__(self, out):
    self.out = out


MultiOutput = dict


class MultiOutputModule(Module):

  def outputs(self, key) -> Module:
    return LambdaModule(out=lambda clock_signal: self.multi_out(clock_signal)[key])

  def out(self, clock_signal) -> np.ndarray:
    raise NotImplementedError

  def multi_out(self, clock_signal) -> MultiOutput:
    return ...


class StepSequencer(MultiOutputModule):

  def __init__(self,
               bmp: Module = Constant(130),
               ):
    pass

  def multi_out(self, clock_signal):
    triggers = ...
    frequencies = ...
    return MultiOutput(frequencies=frequencies, triggers=triggers)


class Root(Module):

  def __init__(self):
    self.step = StepSequencer(...)
    self.envelopes = ...
    self.sine = ...
    self.sine_bass = ...
    envelopes = self.envelopes(self.step.outputs("triggers"))
    tone = self.sine(self.step.outputs("frequencies"))
    self.out = tone * envelopes + self.sine_bass
```

### Time-Varying vs Frame-Constant

Most modules will take `Module` instances in the constructor, that allows for time-varying variables.
An example would be a SineGenerator, where the frequency can vary at each _sample_.

```python
class SineGenerator(Module):
    def __init__(self, frequency: Module):
        self.frequency = frequency

    def __call__(self, clock_signal: ClockSignal):
        frequency = self.frequency(clock_signal)
        ...
```

However, there are other modules, where it is difficult, impossible, or unnecessary to implement time-varying variables.
An example would be an EnvelopeGenerator, where we ask it once per frame to generate, and
where we don't need to change the envelope while it's playing (it's fine to
change the next note (IMO)), or the BPM of a trigger generator, where a fixed BPM per frame is much simpler to 
implement. That's where the following helper methods come in:
- `out_mean(clock_signal) -> float` gives the mean of the output
- `out_mean_int(clock_signal) -> int` gives the mean of the output, as an int, by rounding `out_mean`.

Additionally, we annotate the constructor with `SingleValuedModule`. Example usage:
```python
class EnvelopeGenerator(Module):
    def __init__(self,
                 attack: SingleValuedModule, 
                 release: SingleValuedModule):
        self.attack = attack
        self.release = release
        
    def out(self, clock_signal: ClockSignal):
        attack = self.attack.out_mean_int(clock_signal)
        attack = self.attack.out_mean_int(clock_signal)
        ...
```





## WIP

```sh
python gimme_sound.py -d -1 --output_gen_class=FooBar
```

## Random Ideas

![Figure](https://github.com/fab-jul/pymodsynth/raw/main/fig.png)

- Windows: 
  - Code: Where we write our functions
  - Patch matrix: Matrix where we can type `x` to connect outputs to inputs (might grow too large though)
  - Terminal window: Reload code, play, etc, possibly with a REPL like interface
  - Rolling output of current function f(t)

- Functions:
  - Everything is based on functions that take in `time`, and `*inputs`, and produce one or multiple outputs
  - To get memory, you can make a stateful function (e.g., instance function of some class)
  - There will be a decorator probably

- Interactive inputs
  -  we will need some kind of interactive inputs that simulates knobs. maybe the multi touch trackpad? a mouse? the iPad?

- We form a *graph* via the patch matrix
  - unclear how to create the graph of callables but should be easy
  - should have multiple threads, reload of graph happens in the background: **music should always continue**

- Some form of live input would be cool, e.g., tracking the mouse?

## Function examples

```
def saw_tooth(t, duration=1):
  return t % duration
```

## Links

- [pyo](http://ajaxsoundstudio.com/software/pyo/) for audio?



## Considerations
- some modules have inner state (modulator has its own clock to create a sine). these need to know the ts created by the callback. 
-> pass [ts] as the first channel[0], and have signal_out / signal_in in channel[1]


- callback:
create ts :: [sampling_time]

generator :: (sampling_times, _) -> (sampling_times, signal_values)

filter :: (sampling_times, input_values) -> (sampling_times, output_values)

but how to pass changing parameters? every timestep there could be new params... but different filters have different numbers of parameters... so channels are not the way
-> this is also solved by having access to ts, because paramset can be sampled with ts. 

TODO:
- ability to plug window into any output
- visual knob/slider indicators which move when using mouse/kb to adjust parameters. should show hi/low/current





