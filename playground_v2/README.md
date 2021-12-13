## WIP Guide

### Annotated examples

#### Sine Source

```python
import mz

class SineSource(mz.Module):
    # 1
    frequency: mz.Module = mz.Constant(440.)
    amplitude: mz.Module = mz.Constant(1.0)
    phase: mz.Module = mz.Constant(0.0)

    # 2
    def setup(self):
        # 3
        self._last_cumsum_value = mz.State(0.)

    # 4
    def out_given_inputs(self, 
                         clock_signal: mz.ClockSignal, 
                         frequency: np.ndarray, 
                         amplitude: np.ndarray,
                         phase: np.ndarray):
        dt = np.mean(clock_signal.ts[1:] - clock_signal.ts[:-1])
        cumsum = np.cumsum(frequency * dt, axis=0) + self._last_cumsum_value.get()
        self._last_cumsum_value.set(cumsum[-1, :])
        out = amplitude * np.sin((2 * np.pi * cumsum) + phase)
        return out

```

1. Specify module members like in dataclasses, via class-level _type-annotated_ attributes
2. Specify a `setup` function (**optional**) if you would like to perform further initialization given 
   the module attributes.
3. Any variables that change in `out` must be `mz.State` variables! These are just very light wrappers around arbirary python types, so you may
   use whatever you like, e.g., lists.
4. To define an `out` implementation, the simplest is to define `out_given_inputs`, which takes a `clock_signal` and each of the _`Modules`_ variables


#### Reverb: `SingleValueModule` and `prepend_past`

```py
class Reverb(mz.Module):

    src: mz.Module
    # 1
    delay: mz.SingleValueModule = mz.Constant(3000)
    echo: mz.SingleValueModule = mz.Constant(10000)
    p: mz.SingleValueModule = mz.Constant(0.05)

    def out_given_inputs(self,
                         clock_signal: mz.ClockSignal,
                         src: np.ndarray,
                         # 2
                         delay: float,
                         echo: float,
                         p: float):

        # 3
        past_context = math.ceil(delay + echo)
        num_frames = int(math.ceil(past_context / clock_signal.num_samples)) + 1
        # 4
        src = self.prepend_past("src", current=src, num_frames=num_frames)

        h = basic_reverb_ir(delay, echo, p)

        convolved = scipy.signal.convolve(src, h, mode="valid")
        return convolved[-clock_signal.num_samples:, :]

```

1. You may annotate variables with `mz.SingleValueModule` to indicate that the implementation of the module
   _only uses a single value of the output of that module_
2. In `out_given_inputs`, you then receive a float rather than a full np.ndarray. Note how `src` is annotated
with `mz.Module` above, and thus you receive a `np.ndarray` in `out_given_inputs`
3. You can then implement your module using the floats directly
4. An important function is `prepend_past`, which **automatically buffers inputs for you**. In this example
   we get `src` as well as the last `num_frames-1` versions of `src` in one array.

### `mz.BaseModule` vs other base classes.

We provide a few base classes:

- `mz.BaseModule`: The top base class. Implements a lot of useful functions such as recursive submodule finding, sampling, obtaining
  a cache key, support for math, automatic input buffering.
- `mz.Module`: A very light-weight subclass of `mz.BaseModule` that introduces an **output constraint**: All `out` functions
  of `mz.Module` subclasses are expected to return arrays of shape `(num_samples, num_channels)`. Note that `mz.BaseModule` subclasses
  may return _arbitrary_ stuff from `out`, including, e.g., dictionaries, lists, strings, etc.
- `mz.CacheableBaseModule`: Adds `cached_out`.  # TODO: Maybe fold into base.
- `mz.MultiOutputModule`: Use this if your module returns multiple arrays in a dictionary. See `sources.MelodySequencer` for an example.

### State and Parameters

TODO

### Gotchas

#### "Locked" Modules

After `setup`, a module is - by default - "locked" and you may not change any variables _assignements_, i.e.:

```python
class Foo(mz.Module):

    bar: Module

    def out_given_inputs(self, clock_signal: mz.ClockSignal, bar: np.ndarray):
        self.foo = np.mean(bar)  # Will raise an exception because we assign to self.
```

There are two alternatives:

_Option 1_: (preferred): Use `mz.State` for variables that need to change, and create a placeholder in `setup`. The above example would be:

```python
class Foo(mz.Module):

    bar: Module

    def setup(self):
        self.foo = mz.State()

    def out_given_inputs(self, clock_signal: mz.ClockSignal, bar: np.ndarray):
        self.foo.set(np.mean(bar))  # Ok - we do not create new variables.
```

_Option 2_: (discouraged, but useful for tests etc): Use the `unlocked` context:


```python
class Foo(mz.Module):

    bar: Module

    def out_given_inputs(self, clock_signal: mz.ClockSignal, bar: np.ndarray):
        with self.unlocked():
            self.foo = np.mean(bar)  # Ok - but DISCOURAGED! Use mz.State instead.
```

### Implementation Details

TODO