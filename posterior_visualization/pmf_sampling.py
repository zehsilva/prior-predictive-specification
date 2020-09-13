#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import scipy as sp
import numpy_indexed as npi
import psis


# In[4]:


def full_biased_corr(data_mat):
    temp = np.tril(np.corrcoef(data_mat), -1).flatten()
    temp = temp[np.nonzero(temp)]
    corr1 = np.mean(temp)
    temp = np.tril(np.corrcoef(data_mat, rowvar=False), -1).flatten()
    temp = temp[np.nonzero(temp)]
    corr2 = np.mean(temp)
    return corr1, corr2


def sampling_corr(x, ns=10000):
    N, D = x.shape
    vals = np.zeros(shape=(ns, 2))
    # vals2 = np.zeros(shape=(ns,2))
    for i in range(ns):
        sam1 = np.random.choice(N, 1)
        sam2 = np.random.choice(D, 2)
        vals[i, 0] = x[sam1, sam2[0]]
        vals[i, 1] = x[sam1, sam2[1]]
    corr2 = np.corrcoef(vals, rowvar=False)[0, 1]
    for i in range(ns):
        sam1 = np.random.choice(N, 2)
        sam2 = np.random.choice(D, 1)
        vals[i, 0] = x[sam1[0], sam2]
        vals[i, 1] = x[sam1[1], sam2]
    corr1 = np.corrcoef(vals, rowvar=False)[0, 1]
    return (corr1, corr2)


def sampling_corr_cov_empirical_adjust(x, ns=10000):
    N, D = x.shape
    vals = np.zeros(shape=(ns, 2))
    # vals2 = np.zeros(shape=(ns,2))
    mean = np.mean(x)
    v_bias = np.var(x)
    for i in range(ns):
        sam1 = np.random.choice(N, 1)
        sam2 = np.random.choice(D, 2)
        vals[i, 0] = x[sam1, sam2[0]]
        vals[i, 1] = x[sam1, sam2[1]]
    corr2 = np.corrcoef(vals, rowvar=False)[0, 1]
    cov2 = np.cov(vals, rowvar=False)[0, 1]
    for i in range(ns):
        sam1 = np.random.choice(N, 2)
        sam2 = np.random.choice(D, 1)
        vals[i, 0] = x[sam1[0], sam2]
        vals[i, 1] = x[sam1[1], sam2]
    corr1 = np.corrcoef(vals, rowvar=False)[0, 1]
    cov1 = np.cov(vals, rowvar=False)[0, 1]
    v_unbias = 2 * v_bias + (D - 1) / (N * D - 1) * cov1 + (N - 1) / (N * D - 1) * cov2


def recover_k_empirical_simple(e_corr1_m, e_corr2_m, e_mean, e_var):
    res = ((e_mean / e_var) ** 2) * (
            (1 - (e_corr1_m + e_corr2_m)) * e_var / (e_corr1_m * e_corr2_m) - e_mean / (e_corr1_m * e_corr2_m))
    return (res)


def recover_all_gamma(e_corr1_m, e_corr2_m, e_mean, e_var, mean_a=1):
    K = np.ceil(recover_k_empirical_simple(e_corr1_m, e_corr2_m, e_mean, e_var))
    res1 = (1 - (e_corr1_m + e_corr2_m)) / e_corr1_m - e_mean / (e_corr1_m * e_var)
    res2 = (1 - (e_corr1_m + e_corr2_m)) / e_corr2_m - e_mean / (e_corr2_m * e_var)
    a1 = 1. / res1
    b1 = 1. / res2
    a2 = a1 / mean_a
    b2 = K * (a1 / a2) * (b1 / e_mean)  # e_mean = K*(a1/a2)*(b1/b2)
    return (int(K), a1, a2, b1, b2)


def empirical(data_mat, ns=50000):
    corr1, corr2 = sampling_corr(data_mat, ns)
    e_mean = np.mean(data_mat)
    e_var = np.var(data_mat)
    e_k = recover_k_empirical_simple(corr1, corr2, e_mean, e_var)
    return {"e_mean": e_mean, "e_var": e_var, "e_corr_vals": np.array([corr1, corr2]), "e_latent_dim": e_k}


def empirical_v2(data_mat):
    corr1, corr2 = full_biased_corr(data_mat)
    e_mean = np.mean(data_mat)
    e_var = np.var(data_mat)
    e_k = recover_k_empirical_simple(corr1, corr2, e_mean, e_var)
    return {"e_mean": e_mean, "e_var": e_var, "e_corr_vals": np.array([corr1, corr2]), "e_latent_dim": e_k}


def theoretical(shape_a, scale_a, shape_b, scale_b, latent):
    mean_a = shape_a * scale_a
    mean_b = shape_b * scale_b
    std_a = np.sqrt(shape_a) * (scale_a)
    std_b = np.sqrt(shape_b) * (scale_b)

    v0 = (mean_a * mean_b)
    v1 = (mean_a * std_b) ** 2
    v2 = (mean_b * std_a) ** 2
    v3 = (std_a * std_b) ** 2
    varr = v0 + v1 + v2 + v3
    return {"mean": latent * mean_a * mean_b, "var": (latent) * varr, "corr_vals": np.array([v1 / varr, v2 / varr]),
            "cov_vals": (latent) * np.array([varr - v1, varr - v2, varr])}


def findhyper(target_mean, target_var, target_corr1, target_corr2):
    e_k = recover_k_empirical_simple(target_corr1, target_corr2, target_mean, target_var)

    def optim_fun(x):
        theo = theoretical(x[0], x[1], x[2], x[3], e_k)
        return (target_mean - theo['mean']) ** 4 + (target_var - theo['var']) ** 2 - 100000 * theo['mean'] - theo['var']
        +np.abs(target_corr1 - theo['corr_vals'][0]) / target_corr1 + np.abs(
            target_corr2 - theo['corr_vals'][1]) / target_corr2

    x = sp.optimize.minimize(optim_fun, np.array([10, 10, 10, 10]), method='Nelder-Mead', tol=1e-6,
                             options={"maxtiter": 20000})
    print("theoretical achieved: ", theoretical(x.x[0], x.x[1], x.x[2], x.x[3], e_k))
    print("target values: ", target_mean, target_var, target_corr1, target_corr2)
    return x




# # Prior sampling

# In[7]:


class poisson_gamma():
    def __init__(self, shape_a, rate_a, shape_b, rate_b, latent=11):
        self.shape_a = float(shape_a)
        self.shape_b = float(shape_b)
        self.scale_a = 1.0 / rate_a
        self.scale_b = 1.0 / rate_b
        self.latent = latent

    def sample(self, shape=(1, 1, 100)):
        latent_a = np.random.gamma(self.shape_a, self.scale_a, (self.latent, shape[0], shape[-1]))
        latent_b = np.random.gamma(self.shape_b, self.scale_b, (self.latent, shape[1], shape[-1]))
        self.rmat = np.random.poisson(np.einsum('kj...,ki...', latent_a, latent_b).T)

    def sample2(self, shape=(1, 1, 100)):
        latent_a = np.random.gamma(self.shape_a, self.scale_a, (self.latent, shape[0], shape[-1]))
        latent_b = np.random.gamma(self.shape_b, self.scale_b, (self.latent, shape[1], shape[-1]))
        self.rmat = np.random.poisson(np.einsum('kj...,ki...', latent_a, latent_b).T)

    def theoretical(self):
        mean_a = self.shape_a * self.scale_a
        mean_b = self.shape_b * self.scale_b
        std_a = np.sqrt(self.shape_a) * (self.scale_a)
        std_b = np.sqrt(self.shape_b) * (self.scale_b)

        v0 = (mean_a * mean_b)
        v1 = (mean_a * std_b) ** 2
        v2 = (mean_b * std_a) ** 2
        v3 = (std_a * std_b) ** 2
        varr = v0 + v1 + v2 + v3
        self.mean = self.latent * mean_a * mean_b
        self.var = (self.latent) * varr
        self.corr1 = v1 / varr
        self.corr2 = v2 / varr
        return {"t_mean": self.mean,
                "t_var": self.var,
                "t_corr1": self.corr1,
                "t_corr2": self.corr2}



