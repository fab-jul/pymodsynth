# pymodsynth

Modular Synthesizer in Python

Core Idea: Modular Synthesizers are "just" function generators with a nice user interface. Can we build something similar in Python?

# ⚠  Experimental Project  ⚠

This project is highly experimental still.
To explore the current version, see [playground_v2/README.md](https://github.com/fab-jul/pymodsynth/tree/main/playground_v2).

---

## Contribute

### Guidelines

#### Remove merged feature branches

```sh
BRANCH=<branch>
git branch -d "$BRANCH"       # Delete locally
git push origin -d "$BRANCH"  # Delete remotely
```

## Milestones

- Framework:
  - [ ] Make sure everything we have in main works on macOS and windows
  - [ ] Visualize arbitrary signals.
  - [ ] Automatic testing (github integration?)
  - [ ] Cache outputs.
- Modules:
  - [ ] Kick and Snare (drum machine).
  - [x] Filters (low/mi/high - buttersworth).
  - [ ] Filters (decay / reverb etc.)
  - [ ] Record with microphone.
  - [ ] Input samples.
  - [ ] Record output of the root module into a wav file for sharing/re-use.
  - [ ] Visual feedback in the GUI / explore keyboard and mouse again.



---

# Old stuff.

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


## Next Steps

- Kick and Snare (drum machine)
- Filters (low/mi/high - buttersworth)
- Filters (decay / reverb etc.)
- Record with microphone
- Input samples
- Record our sound
- Visual feedback in the GUI / explore keyboard and mouse again
- ability to plug window into any output
- visual knob/slider indicators which move when using mouse/kb to adjust parameters. should show hi/low/current





