# PyMCMC #

Simple implementation of the Metropolis-Hastings algorithm for Markov Chain Monte Carlo sampling of multidimensional spaces.

The implementation is minimalistic. All that is required is a funtion which accepts an iterable of parameter values, and returns the positive log likelihood at that point. The output can be stored either in memory by means of a numpy array, or in a ROOT::TTree.

Burning and thining is left to the user. For example `data[1000::2]` will burn 1000 samples and thin by a factor of two.

The loop is implemented purely in python. Consequently, it is useful when the likelihood evaluation is relatively slow such that the bottleneck isn't the python overhead.

The chain is *not adaptive*. It is best to learn appropriate scales in a separate run, and use these scales in future runs. This workflow is useful when the same space is sampled many times, with only small variations; for example when evaluating multiple pseudo-data sets.

## Example ##

Following is an example of how a full work flow can be achieved using standard python tools.

```python
import numpy as np
import json
from mcmc import MCMC

initial_values  # initial guess for parameter values
initial_scales  # initial guess the scale for each parameter
ndims  # number of parameters
loglikelihood  # function accepts parameter values, return ll

# Configure the MCMC to sample the space
mcmc = MCMC(ndims)
mcmc.rescale = 2  # typically good for ndims > 5
mcmc.set_values(initial_values)
mcmc.set_scales(initial_scales)

# If you think you're initial scales aren't too far off, you can estiamte a
# better rescale value for optimal convergence
mcmc.learn_scale(
    loglikelihood,
    niter=1000,  # optional, increase if you initial values are far off
    nmax=100)  # optional, incrase if initial scales are far off

# Sample and retrieve the data, burning and thinning as desired
mcmc.run(loglikelihood, 1000000)
data = mcmc.data[1000::2]

# Use this data to estimate better scales and starting parameters
covm = np.cov(data.T)  # get the observed covariance
scales, transform = np.linalg.eigh(covm)  # principal component analysis
scales **= 0.5  # get standard deviations from variances
# The mean of each parameter distribution should be close to its MLE
values = np.mean(data, axis=0)

# Store the trained results
with open('mcmc.json', 'w') as fout:
    config = {
        'values': values.tolist(),
        'scales': scales.tolist(),
        'transform': transform.tolist()
    }
    json.dump(config, fout)

# Prepare a new run using stored config
with open('mcmc.json', 'r') as fin:
    config = json.load(fin)
mcmc = MCMC(ndims)
mcmc.rescale = 2
mcmc.set_scales(np.array(config['values']))
mcmc.set_transform(np.array(config['transform']))
mcmc.set_scales(np.array(config['scales']))
# Learn a better rescale value given the new scales
mcmc.learn_scale(loglikelihood)
```

