#!/usr/bin/env python

from __future__ import division
import math
import numpy as np
from matplotlib import pyplot as plt
_vmpl = list(map(int, plt.matplotlib.__version__.split('.')))
if _vmpl[0] > 1 or (_vmpl[0] == 1 and _vmpl[1] >= 5):
    plt.style.use('ggplot')

from pymcmc import MCMC


class OnlineStats(object):
    """Running computation of mean and variance."""
    def __init__(self):
        self._n = 0
        self._mean = 0
        self._squares = 0

    def update(self, x):
        self._n += 1
        delta = x - self._mean
        self._mean += delta/self._n
        self._squares += delta * (x-self._mean)

    def var(self):
        return self._squares/(self._n-1) if self._n >= 2 else float('nan')

    def mean(self):
        return self._mean


class MultiNorm(object):
    """Multivariate normal distribution"""

    def __init__(self, ndims, mus=list(), sigs=list(), cut=(0, 0)):
        """
        Initialize with some dimensionality, given or random centre and scales,
        and a random covariance.

        :param ndims: int
            dimensionality of the parameter space
        :param mus: iterable
            coordinates for the distribution center
        :param sigs: iterable
            scale parameter for each axis
        :param cut: (float, float)
            make the likelihood zero in this range on the first axis (e.g. to
            turn it into a multi-modal distribution)
        """
        self._ndims = ndims

        # Default scales to order 10 to ensure no accidental unity
        if not mus:
            self._mus = np.random.randn(ndims) * 10
        else:
            self._mus = np.array(mus)

        if not sigs:
            self._sigs = (np.random.rand(ndims)+0.5) * 10
        else:
            self._sigs = np.array(sigs)

        self._cut = cut

        # Build a random correlation matrix
        self._corr = np.random.randn(ndims*ndims).reshape((ndims, ndims))
        # Make positive semi-definite
        self._corr = np.dot(self._corr, self._corr.T)
        # Ensure no correlations above 1
        self._corr /= np.max(self._corr)
        # Set the diagonals to 1
        self._corr[np.diag_indices(ndims)] = 1

        # Build the full covariance matrix, which is now positive definite
        self._cov = np.outer(self._sigs, self._sigs) * self._corr

        # The computation needs the inverse and determinant, compute now
        self._icov = np.linalg.inv(self._cov)
        self._det = np.linalg.det(self._cov)

    def likelihood(self, x):
        """Return the likelihood at the point x"""
        return \
            1/((2*math.pi)**self._ndims * self._det)**0.5 * \
            math.exp(self.loglikelihood(x))

    def loglikelihood(self, x):
        """Return the log likelihood at point x, with some constant offset"""
        if self._cut[0] < x[0] < self._cut[1]:
            return float('-inf')
        diff = x-self._mus
        return -0.5*(np.dot(diff.T, np.dot(self._icov, diff)))

    def sample(self, nsamples):
        """Obtain a true sampling of the space"""
        samples = np.random.multivariate_normal(
            self._mus, self._cov, nsamples)
        mask = \
            (samples[:, 0] < self._cut[0]) + \
            (samples[:, 0] > self._cut[1])
        return samples[mask]


def draw_axis(norm, axis, data):
    rng = (-5*norm._sigs[axis], 5*norm._sigs[axis])

    truth = norm.sample(1000000)
    truth = truth[:, axis]
    truth -= norm._mus[axis]
    truth_vals, bins = np.histogram(truth, 40, rng, normed=True)

    binw = bins[1]-bins[0]
    binc = bins[:-1] + binw/2.

    data = data[:, axis]

    plt.xlabel(r'$x - \mu$')
    plt.ylabel(r'$P(x)$')
    plt.hist(data - norm._mus[axis], 40, rng, normed=True)
    plt.plot(binc, truth_vals, 'o')
    plt.gca().set_xlim(rng)


def trace_axis(norm, axis, data):
    print("Trace for %s, %s" % (norm._mus[axis], norm._sigs[axis]))
    data = data[:, axis]

    stats = OnlineStats()
    mean = list()
    var = list()
    for x in data:
        stats.update(x)
        mean.append(stats.mean())
        var.append(stats.var())

    mean = np.array(mean)
    var = np.array(var)
    var[:2] = 0

    plt.xlabel(r'sample')
    plt.ylabel(r'$x$, $\sigma$')
    plt.plot(data)
    plt.plot(mean)
    plt.plot(var**0.5)


def performance(norm, data):
    """Evaluate the performance of a MCMC's data"""
    rmss = np.std(data, axis=0)
    trmss = norm._sigs
    # Return the average fractional error
    return np.mean(np.fabs(rmss-trmss)/trmss)


def probe_results(scales, ntrials, ndims, mode):
    """
    Probe a grid of results at various scales.

    :param scales: iterable
        rescale values at which to run MCMCs
    :param ntrials: int
        number of trials at each scale
    :param ndims: int
        dimension of space to probe
    :param mode: str
        {var, over, pca}
        var: scale by the known variance of each parameter
        over: over-estimate the variance of each parameter
        pca: tranform to the PCA space
    """
    if mode not in {'var', 'over', 'pca'}:
        raise ValueError("Unknown probe mode: %s" % mode)

    results = list()

    for scale in scales:
        for _ in range(ntrials):
            norm = MultiNorm(ndims)
            mcmc = MCMC(norm._ndims)
            mcmc.verbose = False
            mcmc.rescale = scale

            if mode == 'var':
                # Use the know variance of each paramete as the proposal scale
                mcmc.set_scales(norm._sigs)

            elif mode == 'over':
                # Overestimate the variances by up to 50% (typical)
                mcmc.set_scales(
                    norm._sigs * 
                    (np.random.rand(norm._ndims)/2.+1))

            elif mode == 'pca':
                # Transform the proposal to a space of independent variables
                mcmc.set_covm(norm._cov)

            mcmc.run(norm.loglikelihood, 10000)
            data = mcmc.data[0::1]
            results.append((scale, performance(norm, data), mcmc.getrate()))

            print('%.1f: %.3f, %.2f' % results[-1])

    results = np.array(results)

    plt.xlabel(r'scale')
    plt.ylabel(r'$\langle |\Delta\sigma| / \sigma \rangle$')
    plt.plot(results[:, 0], results[:, 1], 'o')
    plt.savefig('%d_%s_scale.pdf' % (ndims, mode), format='pdf')
    plt.clf()

    plt.xlabel(r'acceptance rate')
    plt.ylabel(r'$\langle |\Delta\sigma| / \sigma \rangle$')
    plt.plot(results[:, 2], results[:, 1], 'o')
    plt.savefig('%d_%s_accept.pdf' % (ndims, mode), format='pdf')
    plt.clf()


# Set some formatting options for numpy and pyplot
np.set_printoptions(precision=2, suppress=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

rescale = 2.5

# 1D simple calse
norm = MultiNorm(1)
mcmc = MCMC(norm._ndims)
mcmc.rescale = rescale
mcmc.set_scales(norm._sigs)
mcmc.run(norm.loglikelihood, 10000)
data = mcmc.data[0::1]
draw_axis(norm, 0, data)
plt.savefig('axis_1D.pdf', format='pdf')
plt.clf()
trace_axis(norm, 0, data[:1000])
plt.savefig('trace_1D.pdf', format='pdf')
plt.clf()

# 1D asymmetric case
norm = MultiNorm(1, mus=[0], sigs=[1], cut=(float('-inf'), 0))
mcmc = MCMC(norm._ndims)
mcmc.rescale = rescale
mcmc.set_scales(norm._sigs)
mcmc.run(norm.loglikelihood, 100000)
data = mcmc.data[0:10000:1]
draw_axis(norm, 0, data)
plt.savefig('axis_asym_10k.pdf', format='pdf')
plt.clf()
data = mcmc.data[0::1]
draw_axis(norm, 0, data)
plt.savefig('axis_asym_100k.pdf', format='pdf')
plt.clf()

# 1D bi-modal case
norm = MultiNorm(1, mus=[0], sigs=[1], cut=(-0.5, 0))
mcmc = MCMC(norm._ndims)
mcmc.rescale = rescale
mcmc.set_scales(norm._sigs)
mcmc.run(norm.loglikelihood, 100000)
data = mcmc.data[0:10000:1]
draw_axis(norm, 0, data)
plt.savefig('axis_bimod_10k.pdf', format='pdf')
plt.clf()
data = mcmc.data[0::1]
draw_axis(norm, 0, data)
plt.savefig('axis_bimod_100k.pdf', format='pdf')
plt.clf()

# 2D case
norm = MultiNorm(2)
print(norm._cov)
mcmc = MCMC(norm._ndims)
mcmc.rescale = rescale
mcmc.set_scales(norm._sigs)
mcmc.run(norm.loglikelihood, 10000)
data = mcmc.data[1000::1]
print(np.cov(data.T))
draw_axis(norm, 0, data)
plt.savefig('axis_2D_0.pdf', format='pdf')
plt.clf()
draw_axis(norm, 1, data)
plt.savefig('axis_2D_1.pdf', format='pdf')
plt.clf()

# Plot the result for one axis, given a small and large scale
norm = MultiNorm(50)
mcmc = MCMC(norm._ndims)
mcmc.rescale = rescale
mcmc.set_covm(norm._cov)
mcmc.run(norm.loglikelihood, 100000)
data = mcmc.data[10000::1]
draw_axis(norm, 0, data)
plt.savefig('axis_50D.pdf', format='pdf')
plt.clf()
trace_axis(norm, 0, data)
plt.savefig('trace_50D.png', format='png')
plt.clf()

# Plot the RMS error as a function of scale for various cases
scales = np.linspace(1, 4, 16)
ntrials = 10
probe_results(scales, ntrials, 50, 'var')
probe_results(scales, ntrials, 50, 'over')
probe_results(scales, ntrials, 50, 'pca')

# 1D traces
norm = MultiNorm(1)
mcmc = MCMC(norm._ndims)
mcmc.rescale = 0.1
mcmc.set_scales(norm._sigs)
mcmc.run(norm.loglikelihood, 1000)
data = mcmc.data[0::1]
trace_axis(norm, 0, data)
plt.savefig('trace_small.pdf', format='pdf')
plt.clf()
mcmc = MCMC(norm._ndims)
mcmc.rescale = 10
mcmc.set_scales(norm._sigs)
mcmc.run(norm.loglikelihood, 1000)
data = mcmc.data[0::1]
trace_axis(norm, 0, data)
plt.savefig('trace_large.pdf', format='pdf')
plt.clf()

