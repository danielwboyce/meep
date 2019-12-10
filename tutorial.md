---
# Perturbation theory with resonant modes of a ring resonator.
---

[Perturbation theory](https://en.wikipedia.org/wiki/Perturbation_theory) is a mathematical method commonly used to find 
an approximate solution to a problem by starting with the exact solution of a related problem and then by solving a 
small “perturbation part” that has been added to known problem. This method is a familiar tool when solving problems in 
quantum mechanics, but can also be beneficial when solving problems in classical electrodynamics, as we will see.

In [Tutorial/Ring Resonator in Cylindrical Coordinates](Ring_Resonator_in_Cylindrical_Coordinates.md) we found the 
resonance modes of a ring resonator in two-dimensional cylindrical coordinates. We will expand this problem using 
perturbation theory to show how performing one simulation can easily allow us to find the resonance states of ring 
resonators with slightly different shapes without performing additional simulations.

[TOC]

The Python Script
-----------------
We begin by defining a cylindrical space and resonator, as performed in [Tutorial/Ring Resonator in Cylindrical 
Coordinates](Ring_Resonator_in_Cylindrical_Coordinates.md):
```python
import meep as mp
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt


def main():
    n = 3.4                 # index of waveguide
    r = 1
    a = r                   # inner radius of ring
    w = 1                   # width of waveguide
    b = a + w               # outer radius of ring
    pad = 4                 # padding between waveguide and edge of PML

    dpml = 2                # thickness of PML
    pml_layers = [mp.PML(dpml)]

    resolution = 100

    sr = b + pad + dpml            # radial size (cell is from 0 to sr)
    dimensions = mp.CYLINDRICAL    # coordinate system is (r,phi,z) instead of (x,y,z)
    cell = mp.Vector3(sr, 0, 0)

    m = 4

    geometry = [mp.Block(center=mp.Vector3(a + (w / 2)),
                         size=mp.Vector3(w, 1e20, 1e20),
                         material=mp.Medium(index=n))]
```
Be sure, as before, to set the `dimensions` parameter to `CYLINDRICAL`. Also note that unlike the previous tutorial, 
`m` has been given a hard value and is no longer a command-line argument. The resolution has also been increased to 100
in order to reduce discretization error. This increase in resolution is only strictly necessary while calculating errors
in the perturbed states, but we increased it throughout the whole script for neatness.

Next, we use Harminv to find a resonant frequency:
```python
    fcen = 0.15         # pulse center frequency
    df = 0.1            # pulse width (in frequency)

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r+0.1))]

    sim = mp.Simulation(cell_size=cell,
                        geometry=geometry,
                        boundary_layers=pml_layers,
                        resolution=resolution,
                        sources=sources,
                        dimensions=dimensions,
                        m=m)

    h = mp.Harminv(mp.Ez, mp.Vector3(r+0.1), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=200)

    Q_values = [mode.Q for mode in h.modes]
    max_Q_index = np.argmax(Q_values)
    Harminv_freq_at_R = h.modes[max_Q_index].freq

    sim.reset_meep()
```

We can use the calculated resonant frequency to run the simulation again, this time where our Gaussian pulse is centered
at the resonant frequency and has an extremely narrow band (so that hopefully only one resonant mode is excited).

```python
    fcen = Harminv_freq_at_R
    df = 0.01

    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r + 0.1))]

    sim = mp.Simulation(cell_size=cell,
                        geometry=geometry,
                        boundary_layers=pml_layers,
                        resolution=resolution,
                        sources=sources,
                        dimensions=dimensions,
                        m=m)

    sim.run(until_after_sources=200)
```

Now things get a bit different. 