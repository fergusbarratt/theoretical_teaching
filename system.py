from __future__ import division
import numpy as np
from numpy import sqrt
from scipy.stats import multivariate_normal, norm
from numpy.random import choice, rand
from numpy import tanh
from copy import copy
import pandas as pd


class system(object):
    '''system class: contains generators for initial state, matrices,
        etc and timestepping procedure. default eta is complete asymmetry
        for given J and initial state set the initial_state_type and J_type to
        None and supply J and initial_state arguments
        noise_dist should be a scipy distribution'''

    def __init__(self,
                 coupling_strength,
                 temperature,
                 dim,
                 initial_state_type='Gaussian',
                 J_type='Gaussian',
                 symmetric=True,
                 eta=0,
                 hopfield_patterns=10,
                 er_p=1,
                 squash_func=tanh,
                 noise_dist=None,
                 J=None,
                 initial_state=None,
                 mean=0,
                 cov=1,
                 squeeze=1,
                 center=0):
        '''eta: symmetry, hopfield patterns, number of patterns used for gener
        ation of hopfield matrices, er_p: erdos renyi acceptance threshold,
        squash_func is threshold function in update'''
        self.p = hopfield_patterns
        self.dim = dim
        self.squash_func = squash_func
        self.temperature = temperature
        self.coupling_strength = coupling_strength
        self.initial_state_type = initial_state_type
        self.J_type = J_type
        self.eta = eta
        self.er_p = er_p

        self.delta = 1/100  # default timestep

        if J_type is 'Gaussian':
            self.J = self._gen_gaussian(coupling_strength, symmetric, eta)
        elif J_type is 'Hopfield':
            self.J = self._gen_hopfield(coupling_strength, self.p)
        elif J_type is 'ER':
            self.J = self._gen_ER(self.er_p, coupling_strength, symmetric, eta)
        elif J_type is 'WU':
            self.J = self._weight_transform(
                self._gen_uniform(coupling_strength, squeeze, center),
                inh_exc=True)
        elif J_type is None and J is not None:
            self.J = J
        else:
            raise NotImplementedError('No J')

        if initial_state_type is 'Gaussian':
            self.initial_state = self._gen_gaussian_initial_state(mean, cov)
        elif initial_state_type is None and initial_state is not None:
            self.initial_state = initial_state
        else:
            raise NotImplementedError('No IS')

        # cache initial state
        self.state = self.initial_state

        if noise_dist is None:
            self.noise_dist = self._gen_noise_dist()

    def _gen_noise_dist(self):
        self.noise_dist = norm(0, sqrt(2*self.temperature*self.delta))

    def _gen_gaussian_initial_state(self, mean=0, cov=1):
        return norm.rvs(size=self.dim, loc=mean, scale=cov)

    def _gen_gaussian(self,
                      J,
                      symmetric,
                      eta=None,
                      mean=0,
                      variance=1,
                      symmetry_threshold=0.99):

        # eta = 1 doesn't work. Anything above symmetry_threshold (0.99)
        # generate a symmetric matrix
        # for setting covariance of elements use J.

        if eta is not None:
            self.eta = eta

        if eta is None or abs(self.eta) > symmetry_threshold:
            self.symmetric = True
            if self.eta != 0:
                self.eta = abs(self.eta)/self.eta  # set eta to +-1
        else:
            # user provided
            self.symmetric = symmetric
            if self.symmetric:
                # make sure eta is not None
                self.eta = 1

        if not self.symmetric:
            # require asymmetric
            gaussian = np.zeros([self.dim, self.dim])
            # joint distribution of pairs on either side of diagonal (eta is the
            # covariance)
            cov = J**2/self.dim*np.array([[1, self.eta],
                                          [self.eta, 1]])
            pair_gen_dist = multivariate_normal([0, 0], cov)
            # generate correlated pairs
            for i, j in [(i, j) for i in range(self.dim)
                         for j in range(self.dim)]:
                if i > j:
                    gaussian[i][j], gaussian[j][i] = pair_gen_dist.rvs()
        else:
            # require symmetric
            non_asymm = norm.rvs(
                loc=0,
                scale=J**2 / self.dim,
                size=(self.dim, self.dim))
            gaussian = (non_asymm + self.eta * non_asymm.T)/2  # symmetry

        return gaussian

    def _gen_ER(self,
                p,
                J,
                symmetric,
                eta=None,
                mean=0,
                variance=1,
                symmetry_threshold=0.99):
        # generate erdos renyi graph with gaussian edge weights.
        # based on _gen_gaussian

        gaussian = self._gen_gaussian(J,
                                      symmetric,
                                      eta,
                                      mean,
                                      variance,
                                      symmetry_threshold)

        return self._er_transform(gaussian, p)

    def _gen_hopfield(self, J, p, er_sparse=False, sparsity=0.1):
        '''ER sparse sets elements to zero columnwise with probability equal
        to the sparsity'''
        self.p = p
        self.patterns = [choice([1.0, -1.0], size=self.dim) for _ in range(p)]
        hopfield = (1/self.dim)*np.sum([np.outer(xi_i, xi_i)
                                        for xi_i in self.patterns], axis=0)
        for i in range(self.dim):
            hopfield[i][i] = 0
        if er_sparse:
            ER = np.zeros_like(hopfield.T)
            for col, col_data in enumerate(hopfield.T):
                for row, elem in enumerate(col_data):
                    if rand() > p:
                        ER[row, col] = elem
            hopfield = ER
        return J*hopfield

    def _gen_uniform(self, J, squeeze=1, center=0):
        '''random iid [0, 1] entries'''
        return J * (rand(self.dim, self.dim) / squeeze + center)

    def _er_transform(self, matrix, p):
        '''delete edges with probability p'''
        ER = np.zeros_like(matrix.T)
        for col, col_data in enumerate(matrix.T):
            for row, elem in enumerate(col_data):
                if rand() > p:
                    ER[row, col] = elem
        return ER

    def _weight_transform(self, matrix, weights=None, inh_exc=False):
        '''weight each x differently in coupling. inh_exc generates random -1s
        +1'''
        if inh_exc:
            weights = choice([1.0, -1.0], size=self.dim)
        if weights is None:
            raise Exception('Need some weights')
        return matrix.dot(np.diag(weights))

    def reset_state(self):
        self.state = self.initial_state

    def regen_state(self, mean=0, cov=1):
        self.state = self._gen_gaussian_initial_state(mean, cov)

    def regen_J(self, eta=None, J_type=None, symmetric=False):
        if J_type is not None:
            self.J_type = J_type
        if eta is not None:
            self.eta = eta
        if self.J_type is 'Gaussian':
            self.J = self._gen_gaussian(
                self.coupling_strength, symmetric=symmetric)
        elif self.J_type is 'Hopfield':
            self.J = self._gen_hopfield(self.coupling_strength, self.p)

    def regen_noise_dist(self):
        self.noise_dist = self._gen_noise_dist()

    def pattern_overlaps(self):
        return np.array([np.sum([self.state * pattern])
                         for pattern in self.patterns])

    def next_state(self, delta):
        # returns the system object after a timestep delta with updated state
        self.delta = delta
        # cache timestep

        self.state = self.state + delta * \
            (-self.state + np.dot(self.J, self.squash_func(self.state))) + \
            self.noise_dist.rvs(size=self.dim)
        return self

    def propagate(self, delta, steps):
        # propagates the system object for steps with timestep delta.
        # returns a list of system objects,
        # leaves the system object in the final state
        states = []
        self._gen_noise_dist()
        for _ in range(steps):
            states.append(copy(self))
            self.next_state(delta)
        return states

    def state_trajectory(self,
                         delta,
                         steps,
                         reset_state=False,
                         initial_state=None,
                         regen_state=False,
                         mean=0,
                         cov=1):

        # propagates the system state for steps with timestep delta
        # returns a list of system states regen_sta
        # cache timestep

        self.delta = delta

        # if initial_state is not None this will reset the state to it after
        # every run
        if initial_state is not None:
            self.state = self.initial_state = initial_state

        self._gen_noise_dist()

        traj = np.empty((steps+1, self.dim))
        traj[0] = self.initial_state

        for t in range(steps):
            traj[t+1] = np.asarray(self.next_state(delta).state)

        # commands for resetting or regenerating the intial_state

        if reset_state:
            self.reset_state()

        elif regen_state:
            self.regen_state(mean, cov)

        return traj

    def ensemble(self,
                 delta,
                 steps,
                 runs,
                 reset_state=True,
                 initial_state=None,
                 regen_state=False,
                 mean=0,
                 cov=1):
        '''run an ensemble of trajectories
         reset_state resets the machine to its initial state after every run
         regen_state regenerates a gaussian state after every run
         initial_state if passed will start the run at that state, and the
         and the machine state will be reset to it after each run
         returns a list of lists of system states, one for each run'''
        self.delta = delta
        # swap the axes here so the time is first
        data = np.swapaxes(
               np.asarray(
                [self.state_trajectory(delta,
                                       steps,
                                       reset_state=reset_state,
                                       regen_state=regen_state,
                                       initial_state=initial_state,
                                       mean=mean,
                                       cov=cov) for _ in range(runs)]), 0, 1)

        xdf = pd.Panel(data=data).to_frame()
        xdf.index.rename(['trajectory', 'spin'], inplace=True)
        return xdf
