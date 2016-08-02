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

        self.nprint = 10000  # print at this interval of evaluations
        self.exclude = list()  # List of parameters to exclude (fixed)
        self.include = list()  # List of parametesr to include (excludes others)
        self.verbose = True  # Display progress
        self.rescale = 2.5  # global scaling for parameters

        self.datapars = list()  # List of parameters to store in data output

        # The parameter values
        self._values = np.zeros(self._npars, dtype=float)
        # Scale the proposal by these factors for each parameter
        self._scales = np.ones(self._npars, dtype=float)

        self._covm = np.zeros((self._npars, self._npars), dtype=float)
        np.fill_diagonal(self._covm, 1)
        self._mean = np.zeros(self._npars, dtype=float)

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
        Set standard deviation of each parameter for the proposal function.

        :param scales: iterable
            list of scales to apply to each parameter
        """
        if len(scales) != self._npars:
            raise ValueError("Must provide one scale for each parameter")
        self._covm.fill(0)
        self._covm[np.diag_indices(self._npars)] = scales**2

    def set_covm(self, covm):
        """
        Set the covariance matrix of the multivariate Normal from which to
        draw proposed transitions.
        """
        covm = np.array(covm, dtype=float)
        if covm.shape != self._covm.shape:
            raise ValueError("Covariance matrix must have shape (npars, npars)")
        else:
            self._covm = covm

    def set_values(self, values):
        """
        Set parameter values.

        :param scales: iterable
            list of values for each parameter
        """
        if len(values) != self._npars:
            raise ValueError("Must provide one value for each parameter")
        self._values = np.array(values, dtype=float)

    def getrate(self):
        """Get the acceptance rate from the last run"""
        return self._naccepted / self._nevaluated

    def proposal(self, n=None):
        """
        Propose a transition to a new point to probe, or a list transitions.

        :param n: int
            number of transitions to propose
        :return: np.array
            proposed transitions with shape (self._npars,) or (n, self._npars)
        """

        shifts = np.random.multivariate_normal(self._mean, self._covm, n)
        shifts *= self.rescale * self._npars**-0.5

        if self._excluded:
            shifts.T[self._excluded] = 0

        return shifts

    def progress_bar(self):
        """
        Print a progress bar with bandwith, eta, acceptance rate.
        """
        prog = self._nevaluated / self._ntarget

        proctime = time.time() - self._starttime
        proctime *= 1e3

        if self._nevaluated > 0:
            bandmean = proctime / 1000. / self._nevaluated
            eta = (self._ntarget-self._nevaluated) * bandmean
            etam = int(eta/60)  # minutes
            etas = int(eta-etam*60)  # seconds
        else:
            eta = float("inf")
            bandwidth = float("inf")

        instband = (proctime-self._lasttime) / self.nprint

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
        self._lasttime = proctime
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
        self._ntarget = int(ntarget)
        self._nevaluated = 0
        self._naccepted = 0
        self._starttime = 0
        self._lasttime = 0

        # If no tree is provided, allocate memory to write out results
        if not self.tree and not self.datapars:
            # For all parameters
            self.data = np.zeros(
                (self._ntarget, self._npars), 
                dtype=float)
        elif not self.tree and self.datapars:
            # For only the given parameters
            self.data = np.zeros(
                (self._ntarget, len(self.datapars)), 
                dtype=float)
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

        shifts = self.proposal(self._ntarget-1)

        self._starttime = time.time()  # start time of this iteration

        # Loop ends when the correct number of points have been accepted
        for shift in shifts:
            # Propose a new point
            self._values += shift

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
                self._values -= shift

            # Store the evaluted point
            if self.tree:
                self.tree.Fill()
            elif self.datapars:
                self.data[self._nevaluated-1] = self._values[self.datapars]
            else:
                self.data[self._nevaluated-1] = self._values

            # Print a progress bar
            if self.verbose and self._nevaluated % self.nprint == 0:
                self.progress_bar()

        # Cap off the progress bar and finish the line
        if self.verbose:
            self.progress_bar()
            sys.stdout.write('\n')
            sys.stdout.flush()

