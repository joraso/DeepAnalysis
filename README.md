# DeepAnalysis
An Implementation of the Boltzmann Generator methodology as developed by Noe et.al. (DOI:10.1126/science.aaw1147)

The method aims to sample complex physical systems with many degrees of freedom by constructing a reverseable deep neural network that learns a mapping between from the coordinate space to an easily sampled, iso-probablistic, prior.

Included so far are:
- Objects and functions for assembling a reverseable DNN.
- A Map object that assembles a Boltzmann Generator to specification, with built in methods for:
  - ML (max likelyhood, or by example) training.
  - KL (by energy) training.
- Implementations of several toy models found in the original paper:
  - A double well.
  - A Muller-Brown potential.
  - A 2D particle dimer system.
- Metropolis Monte Carlo sampling of model objects (for comparison/testing and ML training).

Possible additions for later:
- Extension of model object functionality to objects that are opaque to keras. Current objects rely on their internal calculations being entirely in tensors so that the DNN can compile, but this limits the extensability of the code.
- Testing on additional, more complex models, such as folding protiens, or atomistic fluids.
