#!/usr/bin/env python
# coding: utf-8
"""Poisson Matrix Factorization using sparse representation of input matrix."""


import sys
import numpy as np
import numpy_indexed as npi
from scipy import special
from scipy import stats

import psis



def _compute_expectations(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute E[x] and E[log x]
    '''
    # beta=beta.reshape((beta.shape[0], 1))
    return (alpha / beta, special.psi(alpha) - np.log(beta))


def _compute_entropy(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute Entropy[x]
    '''
    # beta=beta.reshape((beta.shape[0], 1))
    return alpha + (1 - alpha) * special.psi(alpha) - np.log(beta) + special.gammaln(alpha)


def _gamma_term(a, b, shape, rate, Ex, Elogx):
    return np.sum((a - shape) * Elogx - (b - rate) * Ex +
                  (special.gammaln(shape) - shape * np.log(rate)))


def _sum_product_newaxis1(auxvar, data, axis=1):
    return np.sum(auxvar * data[np.newaxis, :, :], axis=axis)



class PoissonMF():
    """
    Poisson Matrix Factorization using sparse representation of input matrix.
    Modification of a code created by: 2014-03-25 02:06:52 by Dawen Liang <dliang@ee.columbia.edu>
    """

    def __init__(self, n_components=100, max_iter=100, tol=0.0005,
                 smoothness=100, random_state=None, verbose=False, allone=False,
                 **kwargs):
        """ Poisson matrix factorization
        Arguments
        ---------
        n_components : int
            Number of latent components
        max_iter : int
            Maximal number of iterations to perform
        tol : float
            The threshold on the increase of the objective to stop the
            iteration
        smoothness : int
            Smoothness on the initialization variational parameters
        random_state : int or RandomState
            Pseudo random number generator used for sampling
        verbose : bool
            Whether to show progress during model fitting
        **kwargs: dict
            Model hyperparameters: theta_a, theta_b, beta_a, beta_b


        self.a1 = float(kwargs.get('theta_a', 0.1)) # shape 
        self.a2 = float(kwargs.get('theta_b', 0.1)) # rate
        self.b1 = float(kwargs.get('beta_a', 0.1))  # shape
        self.b2 = float(kwargs.get('beta_b', 0.1))  # rate

        """
        self.allone = allone
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose
        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)
        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.a1 = float(kwargs.get('theta_a', 0.1))
        self.a2 = float(kwargs.get('theta_b', 0.1))
        self.b1 = float(kwargs.get('beta_a', 0.1))
        self.b2 = float(kwargs.get('beta_b', 0.1))

    def _init_components(self, n_rows, n_cols):
        # variational parameters for beta
        #print("( %f, %f )" % (n_rows, n_cols))
        self.gamma_b = self.smoothness * np.random.gamma(self.smoothness, 1. / self.smoothness,
                                                         size=(n_rows, self.n_components))
        self.rho_b = self.smoothness * np.random.gamma(self.smoothness, 1. / self.smoothness,
                                                       size=(n_rows, self.n_components))
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)
        # variational parameters for theta
        self.gamma_t = self.smoothness * np.random.gamma(self.smoothness, 1. / self.smoothness,
                                                         size=(n_cols, self.n_components))
        self.rho_t = self.smoothness * np.random.gamma(self.smoothness, 1. / self.smoothness,
                                                       size=(n_cols, self.n_components))
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)

    def fit(self, X, n_rows=0, n_cols=0):
        '''Fit the model to the data in X.
        Parameters
        ----------
        X : array-like, shape (n_examples, 3)
            Training data.
        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        X_new = X.copy()
        if n_rows == 0:
            self.n_rows = np.max(X_new[:, 0]) + 1
        else:
            self.n_rows = n_rows

        if n_cols == 0:
            self.n_cols = np.max(X_new[:, 1]) + 1
        else:
            self.n_cols = n_cols

        if self.verbose:
            print("[pmf] rows=", self.n_rows)
            print("[pmf] cols=", self.n_cols)
        assert (np.max(X_new[:, 0]) < self.n_rows), "There is a row index in the data numbered "+str(np.max(X_new[:, 0]))+" that violate the dimension n_rows="+str(self.n_rows)
        self.row_index = X_new[:, 0]
        assert (np.max(X_new[:, 1]) < self.n_cols), "There is a column index in the data numbered "+str(np.max(X_new[:, 1]))+" that violate the dimension n_cols="+str(self.n_cols)
        self.cols_index = X_new[:, 1]
        self.vals_vec = X_new[:, 2]
        self._init_components(self.n_rows, self.n_cols)  # beta, theta
        return self._update(X_new)

    def transform(self, X, attr=None):
        '''Encode the data as a linear combination of the latent components.
        TODO
        '''
        return 1

    def _update_phi(self, X):
        self.phi_var = np.zeros((X.shape[0], self.n_components))
        self.phi_var = np.add(self.phi_var, np.exp(self.Elogb[self.row_index, :]))
        self.phi_var = np.add(self.phi_var, np.exp(self.Elogt[self.cols_index, :]))
        self.phi_var = np.divide(self.phi_var, np.sum(self.phi_var, axis=1)[:, np.newaxis])
        self.phi_var = self.vals_vec[:, np.newaxis] * self.phi_var

    def _update(self, X, update_beta=True):
        # alternating between update latent components and weights
        old_bd = -np.inf
        elbo_lst = []
        for i in range(self.max_iter):
            self._update_phi(X)
            self._update_theta(X)
            if update_beta:
                self._update_phi(X)
                self._update_beta(X)
            bound = self._bound(X)
            elbo_lst.append(bound)
            if (i > 0):
                improvement = abs((bound - old_bd) / (old_bd))
                if self.verbose:
                    sys.stdout.write('\r\tAfter ITERATION: %d\tObjective: %.2f\t'
                                     'Old objective: %.2f\t'
                                     'Improvement: %.5f' % (i, bound, old_bd,
                                                            improvement))
                    sys.stdout.flush()
                if improvement < self.tol:
                    break
            old_bd = bound
        if self.verbose:
            sys.stdout.write('\n')
        return elbo_lst

    def _update_theta(self, X):
        grouped = npi.group_by(self.cols_index).sum(self.phi_var)
        self.gamma_t[grouped[0]] = self.a1 + grouped[1]
        self.rho_t = self.a2 + np.sum(self.Eb, axis=0, keepdims=True)
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)

    def _update_beta(self, X):
        self.gamma_b = self.b1 + npi.group_by(self.row_index).sum(self.phi_var)[1]
        self.rho_b = self.b2 + np.sum(self.Et, axis=0, keepdims=True)
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def _bound(self, X):
        bound = np.sum(self.phi_var * (self.Elogt[self.cols_index, :] + self.Elogb[self.row_index, :]))
        bound -= np.sum(self.phi_var * (np.log(self.phi_var) - np.log(X[:, 2]).reshape(X.shape[0], 1)))
        bound -= np.sum(np.inner(self.Eb, self.Et))
        bound += _gamma_term(self.a1, self.a2,
                             self.gamma_t, self.rho_t,
                             self.Et, self.Elogt)
        bound += _gamma_term(self.b1, self.b2, self.gamma_b, self.rho_b,
                             self.Eb, self.Elogb)
        return bound

    def samplePosterior(self):
        latent_a = np.random.gamma(self.gamma_t, 1. / self.rho_t)
        latent_b = np.random.gamma(self.gamma_b, 1. / self.rho_b)
        return np.random.poisson(np.inner(latent_a, latent_b))

    def samplePrior(self):
        latent_a = np.random.gamma(self.a1, 1. / self.a2, (self.n_cols, self.n_components))
        latent_b = np.random.gamma(self.b1, 1. / self.b2, (self.n_rows, self.n_components))
        return np.random.poisson(np.inner(latent_a, latent_b))

    def psis(self, test_data, NSAMPLES=1):
        log_lik = np.zeros((NSAMPLES, test_data.shape[0]))
        for n in range(NSAMPLES):
            t = np.random.gamma(self.gamma_t, 1. / self.rho_t)  # gamma_t is shape and rho_t is rate
            b = np.random.gamma(self.gamma_b, 1.0 / self.rho_b)  # gamma_b is shape and rho_b is rate
            lambdas = np.inner(t, b).T
            log_lik[n, :] = stats.poisson(lambdas[test_data[:, 0], test_data[:, 1]]).logpmf(test_data[:, 2]).reshape(-1)
        loo, loos, ks = psis.psisloo(log_lik)
        return loo



