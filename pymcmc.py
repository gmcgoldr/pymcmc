from __future__ import division
import sys
import math
import time
import warnings

import numpy as np


class MCMC(object):
    """
    Samples a space of a given dimensionality using a Markov Chain Monte Carlo.
    Stores the sampled points in a ROOT.TTree, or in an np.ndarray. If the
    points data will exceed the available memory, use a TTree associated with 
    a ROOT.TFile object, which will get buffered to disk.
    """

    def __init__(self, npars):
        """
        Minimal initialization information is the number of parameters.

        :param npars: int
            dimensionality of likelihood sapce
        """
        np.random.seed((int(time.time())+id(self))%4294967295)

        self._npars = npars  # dimensions of the MCMC space

        self.nprint = 1000  # print at this interval of evaluations
        self.exclude = list()  # List of parameters to exclude (fixed)
        self.include = list()  # List of parametesr to include (excludes others)
        self.verbose = True  # Display progress
        self.rescale = 2  # global scaling for parameters

        self.datapars = list()  # List of parameters to store in data output

        # The parameter values
        self._values = np.zeros(self._npars, dtype=np.float64)
        # Scale the proposal by these factors for each parameter
        self._scales = np.ones(self._npars, dtype=np.float64)

        self._transform = None  # Transformation from proposal to likelihood
        self._excluded = None  # List of excluded parameters

        # Mutable state during running (clear before running)

        self.data = None  # the np.ndarray of results if not writing to a tree
        self.tree = None  # ROOT.TTree to write if not writing to array

        self._ntarget = 0  # Target number of evaluations
        self._nevaluated = 0  # The number of evaluated points
        self._naccepted = 0  # The number of points accepted
        self._proctime = 0;  # total processing time in ms
        self._lasttime = 0;  # processing time at last print

    def configure_tree(self, tree=None, names=list()):
        """
        Configure a tree for storing the points. Points won't be stored to the
        data array.

        :param tree: ROOT.TTree
            use this tree object (if None, makes a new tree)
        :param names: iterable
            list of names for the tree branches
        """
        # Only import if needed, as it is time consuming and might not be
        # available to every user
        from ROOT import TTree

        if not tree:
            self.tree = TTree("Evaluations", "MCMC Evaluations")
        else:
            self.tree = tree

        # If no branch names are provided, generate them
        if not names:
            # Pad the parameter number with 0s for correct alphanumeric sorting
            nord = math.log(self._npars-1, 10)
            names = [('Par%0'+nord+'d') % i for i in range(self._npars)]

        # Generate the branches in the tree
        for ipar, name in enumerate(names):
            # ROOT knows how to get the address of a np.ndarray. By slicing to
            # an index, ROOT will get the address at that position in the array
            self.tree.Branch(name, self._values[ipar:], '%s/D'%name)

    def set_scales(self, scales):
        """
        Set scaling factors to apply to the proposed points.

        :param scales: iterable
            list of scales to apply to each parameter
        """
        if len(scales) != self._npars:
            raise ValueError("Must provide one scale for each parameter")
        self._scales = np.array(scales, dtype=np.float64)

    def set_values(self, values):
        """
        Set parameter values.

        :param scales: iterable
            list of values for each parameter
        """
        if len(values) != self._npars:
            raise ValueError("Must provide one value for each parameter")
        self._values = np.array(values, dtype=np.float64)

    def set_transform(self, transform):
        """
        Set a mapping from the space in which to propose new points, to the
        space of the likelihood function.

        In the following example, the data with shape (nobs, npars) from a
        prior MCMC run is used to do PCA. The PCA matrix, wherein each column
        is a normalized eigenvector of the data's space, can be used as a
        transform.

        >>> covm = np.cov(data.T)
        >>> scales, transform = np.linalg.eigh(covm)
        >>> mcmc.set_transform(transform)
        >>> mcmc.set_scales(scales**0.5)

        The proposal function can then propose shifts along independent axes.
        The proposal space is correctly scaled, accounting for correlations.

        :param transform: array_like
            transformation matrix M_{i,j} mapping the space in which to propose
            new points to the likelihood space. Each row i is a parameter in
            the likelihood space, each column j is a parameter in the space of
            the proposal function.
        """
        self._transform = np.array(transform, dtype=np.float64)
        if self._transform.shape != (self._npars, self._npars):
            self._transform = None
            raise ValueError("Transformation matrix must have shape (npars, npars)")

    def getrate(self):
        """Get the acceptance rate from the last run"""
        return self._naccepted / self._nevaluated

    def proposal(self):
        """
        Propose a new point to move to in the likelihood space.

        :return: np.ndarray
            shifts to apply to each parameter for the next evaluation
        """

        shifts = np.random.randn(self._npars)
        shifts *= self._scales / self._npars**0.5 * self.rescale

        if self._transform is not None:
            shifts = np.dot(self._transform, shifts)

        # Remove the shifts from parameters excluded from the MCMC
        if self._excluded:
            shifts[self._excluded] = 0

        return shifts

    def progress_bar(self):
        """
        Print a progress bar with bandwith, eta, acceptance rate.
        """
        prog = self._nevaluated / self._ntarget

        if self._nevaluated > 0:
            bandmean = self._proctime / 1000. / self._nevaluated
            eta = (self._ntarget-self._nevaluated) * bandmean
            etam = int(eta/60)  # minutes
            etas = int(eta-etam*60)  # seconds
        else:
            eta = float("inf")
            bandwidth = float("inf")

        instband = (self._proctime-self._lasttime) / self.nprint

        rate = 100*self._naccepted/self._nevaluated

        bar = '=' * int(prog*50)  # string of = chars for progress
        fill = ' ' * (50-len(bar))  # pad with spaces

        sys.stdout.write('\r[%s%s] %3d%% %5.2fms %02d:%02d min. (%6.2f%%)' % (
            bar, fill,  # progress bar
            int(prog*100),  # percentage completion
            instband,  # instantaneous bandwidth in ms
            etam, etas,  # time to completion in minutes
            rate))  # acceptance rate
        sys.stdout.flush()

        # Update instantaneous state
        self._lasttime = self._proctime
        self._last_naccpeted = self._naccepted

    def learn_scale(
            self, 
            loglikelihood, 
            ntarget=1000, 
            nmax=100,
            ratemin=0.22,
            ratemax=0.24):
        """
        Adjust the `rescale` parameter until the acceptance rate is in the
        desired range.

        :param loglikelihood: func([float]) -> float
            returns the log likelihood for the passed parameter values
        :param ntarget: int
            number of points to evaluate
        :param nmax: int
            maximum steps to take trying to find optimal `rescale` value
        :param ratemin: float
            minimum acceptance rate
        :param ratemax: float
            maximum acceptance rate
        """

        # Modify internal state to avoid storing these chains, and to not
        # show many progress bars
        prev_tree = self.tree
        self.tree = None
        prev_verbose = self.verbose
        self.verbose = False

        niters = 0   # number of attempts to find optimal scale
        step = 1  # step size to follow for next scale
        side = 0  # -1 if below range, +1 if above range

        # Continually change scale parameter until acceptance is in range
        while True:
            # Cap maximum number of attempts
            niters += 1
            if niters > nmax:
                warnings.warn("Failed to learn scale", RuntimeWarning)
                return False

            # Run the MCMC with the current scale
            self.run(loglikelihood, ntarget)
            rate = self._naccepted/self._nevaluated

            if rate < ratemin:
                # Rate is too low, decrease scale to stay closer to high
                # likelihood point resuling in more acceptance
                self.rescale *= 1/(1+step)
                # Acceptance was too large before, overshot optimal rate, so
                # decrease step size to look between the two values next
                if side == 1:
                    step /= 2
                # Indicate that last run had too low acceptance
                side = -1

            elif rate > ratemax:
                # Rate is too high, increase scale to probe points further from
                # the high likleihood region, resulting in less acceptance but
                # better probing of the entire space
                self.rescale *= (1+step)
                if side == -1:
                    step /= 2
                side = 1

            else:
                # Rate is in the given range, stop adjusting it
                break

        # Restore the original state of verbosity and output 
        self.tree = prev_tree
        self.verbose = prev_verbose
        self.data = None

        # Succeeded at finding a parameter
        return True

    def run(self, loglikelihood, ntarget):
        """
        Perform the MCMC computation. 

        :param loglikelihood: func([float]) -> float
            returns the log likelihood for the passed parameter values
        :param ntarget: int
            number of points to evaluate
        """

        if self.include and self.exclude:
            raise ValueError("Can't specify both included and excluded parameters")

        # Setup mutable state for this run
        self._ntarget = ntarget
        self._nevaluated = 0
        self._naccepted = 0
        self._proctime = 0
        self._lasttime = 0

        # If no tree is provided, allocate memory to write out results
        if not self.tree and not self.datapars:
            # For all parameters
            self.data = np.zeros(
                (self._ntarget, self._npars), 
                dtype=np.float64)
        elif not self.tree and self.datapars:
            # For only the given parameters
            self.data = np.zeros(
                (self._ntarget, len(self.datapars)), 
                dtype=np.float64)
        else:
            # Don't store in memory (use tree)
            self.data = None

        # Build a list of parameter indices to exclude (don't shift those)
        if self.include:
            self._excluded = np.array(
                set(range(self._npars)) - set(self.include))
        elif self.exclude:
            self._excluded = np.array(self.exclude)

        # Evaluate and accept the starting point
        last_prob = loglikelihood(self._values)
        self._nevaluated += 1
        self._naccepted += 1
        if self.tree:
            self.tree.Fill()
        elif self.datapars:
            self.data[self._nevaluated-1] = self._values[self.datapars]
        else:
            self.data[self._nevaluated-1] = self._values

        # Loop ends when the correct number of points have been accepted
        while self._nevaluated < self._ntarget:
            start = time.time()  # start time of this iteration

            # Propose a new point
            shifts = self.proposal()
            self._values += shifts

            # Evaluate the likelihood at that point
            prob = loglikelihood(self._values)
            self._nevaluated += 1

            # MCMC: if new likelihood > old, keep the point. Otherwise if
            # new/old > random uniform, keep
            if prob >= last_prob or \
                    math.exp(prob-last_prob) > np.random.rand():
                self._naccepted += 1
                last_prob = prob

            # Case where MCMC doesn't accept the point, reset to last point
            else:
                self._values -= shifts

            # Store the evaluted point
            if self.tree:
                self.tree.Fill()
            elif self.datapars:
                self.data[self._nevaluated-1] = self._values[self.datapars]
            else:
                self.data[self._nevaluated-1] = self._values

            # Keep track of time per evaluation in ms
            self._proctime += (time.time()-start)*1E3

            # Print a progress bar
            if self.verbose and self._nevaluated % self.nprint == 0:
                self.progress_bar()

        # Cap off the progress bar and finish the line
        if self.verbose:
            self.progress_bar()
            sys.stdout.write('\n')
            sys.stdout.flush()

