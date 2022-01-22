

# Quickstart

If you want to synthesize music immediately without studying the concepts, clone the project and start here.
### IDE Setup

[IDE setup goes here]

### A Toy Example
*Everything is a Module*

In order to create music from nothing, we have to derive from `Module`:
```python
import modular_synthesizer as mz

class BabiesFirstSynthie(mz.Module):
    def setup(self):
        self.base_frequency = mz.Parameter(220, key='f')
        self.lfo = mz.SineSource(frequency=mz.Constant(0.66))
        self.dancing_triangle = mz.SkewedTriangleSource(frequency=self.base_frequency, 
                                                        alpha=mz.lift(self.lfo))
        self.low_hum = mz.SineSource(frequency=mz.Parameter(44), key='b')
        self.out = self.dancing_triangle + self.low_hum * 0.5
```
In `setup()`, we
- create some sources, all of which are `Modules`.
- define how our new `Module` should process them.
- assign the result to `self.out` so it can be passed to the sound buffer of our system - or elsewhere.

BabiesFirstSynthie has three basic sources - `Modules` that do not process any inputs (but may very well be parametrized by other `Modules`): 
- `base_frequency` is a `Parameter`: A constant `Module` whose value you can control by pressing the given key and moving your mouse left and right. Try it, it's fun!
- `lfo`, a low frequency oscillator, in this case a simple sine wave with constant frequency 0.66.
- `low_hum`, another sine wave whose frequency is another `Parameter` you can control by key or midi knob.

You can already start to see our principle: Nothing is just a number. Everything is a `Module`. We do not pass `int` or `float` as frequency arguments, we pass `Modules`. Why? Because our `Modules` are callables, and callables *compose*. 

The mighty consequence is that instead of passing your `Modules'` `self.out` to your sound buffer, you can make it the input of any other `Module`! 

BabiesFirstSynthie has a fourth source:
```python
self.dancing_triangle = mz.SkewedTriangleSource(frequency=self.base_frequency, 
                                                alpha=mz.lift(self.lfo))
```
In addition to a frequency argument, this source takes a skew angle alpha. For both arguments, we pass the `Modules` defined before, so they will change over time. This creates a dynamic sound and showcases one of the principles of modular synthesizing: *Any signal can be a control signal!* 

Many periodic signals have codomain [-1, 1], but for many parameters, only positive values make sense. This has to be accounted for when composing `Modules`. In the case of `alpha`, `mz.lift` suffices, which simply lifts a signal from [-1, 1] to [0, 1].

Finally, the output of this Module is a weighted sum - expressed with overloaded arithmetic operators. Depending on the downstream consumers, this might need to be compressed, lifted or otherwise processed. 

### Hot Reloading
Have you listened to the toy example? Did you find the skew angle changes too slowly? It's a shame we made its frequency, `lfo` a `Constant` instead of a `Parameter`, which would be changeable at runtime. But it is not too late. Leave the program running, and edit the code to:

```python
self.lfo = mz.SineSource(frequency=mz.Parameter(0.66, key='l', lo=0.1, hi=60, clip=True))
```
Save the file and highlight the output window. Now (if the program did not crash) you should be able to press 'l' and change the lfo frequency in the optionally given lo-hi range. 

We call this **hot reloading** and it turns out to be extremely useful during exploration. While there are limitations, you can hot reload in many situations. For example, try to add another source to `setup()` and don't forget to add it to `self.out` at the end (or directly add it like in the snippet below). If you save the file before it can parse successfully, the program should not crash, but cannot reload. Fix the code, and try again. 

```python
self.out = self.dancing_triangle + self.low_hum * 0.5 + mz.SineSource(frequency=mz.Constant(880))
```


# Motivation and Basic Idea

## Signal Path

# Types

## Basic Types

## Module

## Constant, Parameter

## Module-Like

## 



