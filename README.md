# pymodsynth

Modular Synthesizer in Python

Core Idea: Modular Synthesizers are "just" function generators with a nice user interface. Can we build something similar in Python?

## Code Health - Fix this asap

- [ ] Make sure everything we have in main works on macOS and windows
- [ ] Add some basic unit tests for the sound loop and for main modules

## Milestones

- Framework:
  - [ ] Visualize arbitrary signals.
  - [ ] Automatic testing (github integration?)
  - [ ] Cache outputs.
- Modules:
  - [ ] Kick and Snare (drum machine).
  - [ ] Filters (low/mi/high - buttersworth).
  - [ ] Filters (decay / reverb etc.)
  - [ ] Record with microphone.
  - [ ] Input samples.
  - [ ] Record output of the root module into a wav file for sharing/re-use.
  - [ ] Visual feedback in the GUI / explore keyboard and mouse again.


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





