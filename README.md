# PyMCMC #

Simple implementation of the Metropolis-Hastings algorithm for Markov Chain Monte Carlo sampling of multidimensional spaces.

The implementation is minimalistic. All that is required is a function which can set the point in space to probe, and one which can return the log likelihood at that point. The output can be stored either in memory by means of a numpy array, or in a ROOT::TTree.

Burning and thining is left to the user. For example `data[1000::2]` will burn 1000 samples and thin by a factor of two.

The loop is implemented purely in python. Consequently, it is useful when the likelihood evaluation is relatively slow such that the bottleneck isn't the python overhead.

The chain is *not adaptive*. It is best to learn appropriate scales in a separate run, and use these scales in future runs. This workflow is useful when the same space is sampled many times, with only small variations; for example when evaluating on multiple pseudo-data sets.

## Example ##

Given an object `space` with attribute `ndims` and methods `setpars`, `loglikelihood`.

```python
import numpy as np
import json
from mcmc import MCMC

initial_scales  # initial guess the scale for reach parameter
ndims  # number of parameters
setpars  # function which sets the parameter values
loglikelihood  # function which evaluates at the current parameters

# Configure the MCMC to sample the space
mcmc = MCMC(ndims)
mcmc.rescale = 2
mcmc.set_scales(initial_scales)

# Sample and retrieve the data, burning and thinning as desired
mcmc.run(100000, loglikelihood, setpars)
data = mcmc.data[1000::2]

# Use this data to estimate better scales
covm = np.cov(data.T)  # get the observed covariance
scales, transform = np.linalg.eigh(covm)  # principal component analysis
scales **= 0.5  # get standard deviations from variances

# Store the trained results
with open('scales.json', 'w') as fout:
    config = {
        'scales': scales.tolist(),
        'transform': transform.tolist()
    }
    json.dump(config, fout)

# Run a spearate chain, using stored config
with open('scales.json', 'r') as fin:
    config = json.load(fin)
mcmc = MCMC(ndims)
mcmc.rescale = 2
mcmc.set_transform(np.array(config['transform']))
mcmc.set_scales(np.array(config['scales']))
```

