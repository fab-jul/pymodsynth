# pymodsynth
Modular Synthesizer in Python

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

- We form a *graph* via the patch matrix
  - unclear how to create the graph of callables but should be easy
  - should have multiple threads, reload of graph happens in the background: **music should always continue**

- Some form of live input would be cool, e.g., tracking the mouse?

```
def saw_tooth(t, duration=1):
  return t % duration
```
