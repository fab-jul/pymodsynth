# API: Thoughts / Ideas

Terms:
- `sample`: A single time-step in the output signal.
- `frame`: A collection of e.g. 2048 samples. We generate frames in each event loop step.

## Parameters v2


We need to fix implementation, Parameters should be pluggable anywhere in the module hierarchy:


```python


class Root(Module):
  
  def __init__(self):
    # Unnamed paramters.
    self.sine = SineGenerator(frequency=Parameter(...))
    
    # Using the same parameter multiple times.
    self.bpm = Parameter(...)
    self.step_seq_melody = StepSequencer(bpm=self.bpm)
    self.step_seq_melody_highs = StepSequencer(bpm=self.bpm)

```


## MultiOutputModule

We should have a module with multiple outputs:

```python
import typing 
import numpy as np

# Normal Module, as we have it atm.
class Module:

  def __call__(self, clock_signal) -> np.ndarray:
    return self.out(clock_signal)

  def out(self, clock_signal) -> np.ndarray:
    raise NotImplementedError
  
  
class OutputFn(typing.Protocol):
  
  def __call__(self, clock_signal) -> np.ndarray:
    """Output function"""


class LambdaModule(Module):
  """Helper module to create a module on the fly given an output function."""

  def __init__(self, out: OutputFn):
    self.out = out


MultiOutput = dict


class MultiOutputModule(Module):
  """Module with a two special functions:`multi_out` and `outputs`."""

  def outputs(self, output_name) -> Module:
    """Get the output with name `output_name`."""
    return LambdaModule(
      out=lambda clock_signal: self.multi_out(clock_signal)[output_name])

  def out(self, clock_signal) -> np.ndarray:
    raise NotImplementedError("MultiOutputModule does not support direct out calling!")

  def multi_out(self, clock_signal) -> MultiOutput:
    return ...


# Example usage:


class StepSequencer(MultiOutputModule):

  def multi_out(self, clock_signal) -> MultiOutput:
    triggers = ...
    frequencies = ...
    return MultiOutput(frequencies=frequencies, triggers=triggers)


class Root(Module):

  def __init__(self):
    self.step = StepSequencer(...)
    self.envelopes = ...
    self.sine = ...
    self.sine_bass = ...
    # Here we use the `outputs` function to get outputs by name.
    envelopes = self.envelopes(self.step.outputs("triggers"))
    tone = self.sine(self.step.outputs("frequencies"))
    self.out = tone * envelopes + self.sine_bass
```

## Time-Varying vs Frame-Constant

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


