NpMolDescriptors
==============================

Vectorized Numpy calculations of certain molecular descriptors. These include:

- (Sorted) Coulomb Matrix

- Atom-atom distances

- Behler Parinello ACSFs (G1, G2, and G4)

This package is intended for fast transformation from Cartesians to these various descriptors using NumPy.  In the future, we may migrate these over to tensorflow vectors that can be evaluated on a GPU to see if timing would increase.

### Requirements

- NumPy

- PyVibDMC

### Copyright

Copyright (c) 2020, Ryan DiRisio

