# coding: utf-8

# In[28]:
import matplotlib

matplotlib.use('Agg')

import jax.numpy as npj
from jax import grad
import time
import numpy as np
import scipy.stats as stats
import scipy.special as sspecial
import matplotlib.pyplot as plt
import pandas as pd

exp_print_count = 0

# from google.colab import files
plt.style.use("ggplot")


def savedownload_fig(filen):
    plt.savefig(filen)


#  files.download(filen)

def savedownload_csv(filen, df):
    df.to_csv(filen, sep="\t")


#  files.download(filen)


# In[29]:


### auxiliary functions

def recover_k_empirical_simple(BaseEDM, e_corr1_m, e_corr2_m, e_mean, e_var):
    emean_changed = e_mean
    if (BaseEDM is not None):
        emean_changed = BaseEDM.especial_factor() * e_mean
    res = ((e_mean / e_var) ** 2) * ((1 - (e_corr1_m + e_corr2_m)) * e_var / (e_corr1_m * e_corr2_m) - emean_changed / (
            e_corr1_m * e_corr2_m))
    return (res)


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


def sampling_corr_empirical_others(x, ns=10000):
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
    return (corr1, corr2, cov1, cov2, mean, v_unbias, v_bias)


def full_biased_corr(data_mat):
    temp = np.tril(np.corrcoef(data_mat), -1).flatten()
    temp = temp[np.nonzero(temp)]
    corr1 = np.mean(temp)
    temp = np.tril(np.corrcoef(data_mat, rowvar=False), -1).flatten()
    temp = temp[np.nonzero(temp)]
    corr2 = np.mean(temp)
    return corr1, corr2


def empirical(data_mat, BaseEDM, ns=50000):
    corr1, corr2 = sampling_corr(data_mat, ns)
    e_mean = np.mean(data_mat)
    e_var = np.var(data_mat)
    e_k = recover_k_empirical_simple(BaseEDM, corr1, corr2, e_mean, e_var)
    return {"e_mean": e_mean, "e_var": e_var, "e_corr1": corr1, "e_corr2": corr2, "e_latent_dim": e_k}


def empirical_v2(data_mat, BaseEDM):
    corr1, corr2 = full_biased_corr(data_mat)
    e_mean = np.mean(data_mat)
    e_var = np.var(data_mat)
    e_k = recover_k_empirical_simple(BaseEDM, corr1, corr2, e_mean, e_var)
    return {"e_mean": e_mean, "e_var": e_var, "e_corr1": corr1, "e_corr2": corr2, "e_latent_dim": e_k}


# In[35]:


class Abstract_EDM(object):
    def __init__(self, theta, kappa, phi_fun):
        # super().__init__()
        self.theta = float(theta)
        self.kappa = float(kappa)
        self.phi_fun = phi_fun
        self.dphi = grad(self.phi_fun)
        self.ddphi = grad(self.dphi)

    def mean(self):
        return self.kappa * float(self.dphi(self.theta))

    def var(self):
        return self.kappa * float(self.ddphi(self.theta))

    def especial_factor(self):
        # factor k*phi'(theta)+phi''(theta)/phi'(theta)
        # appears in some theoretical formulas, and in particular to the one with number of latent dimension
        return self.kappa * float(self.dphi(self.theta)) + float(self.ddphi(self.theta)) / float(self.dphi(self.theta))

    def sample(self, nsamples):
        pass

    def update(self, theta, kappa):
        pass


class Gaussian_EDM(Abstract_EDM):
    def __init__(self, mu, sigma):
        theta = mu / (sigma ** 2)
        kappa = sigma ** 2
        phi_fun = lambda x: npj.power(x, 2) / 2
        super(Gaussian_EDM, self).__init__(theta, kappa, phi_fun)
        self.mu = mu
        self.sigma = sigma

    def sample(self, shape=None):
        return np.random.normal(self.mu, self.sigma, shape)

    def update(self, theta, kappa):
        self.mu = theta * kappa
        self.sigma = np.sqrt(kappa)
        self.theta = theta
        self.kappa = kappa


class Gamma_EDM(Abstract_EDM):
    def __init__(self, shape, rate):
        theta = -rate
        kappa = shape
        phi_fun = lambda x: -npj.log(-x)
        super(Gamma_EDM, self).__init__(theta, kappa, phi_fun)
        self.shape = shape
        self.rate = rate

    def sample(self, shape=None):
        return np.random.gamma(self.shape, 1. / self.rate, shape)

    def update(self, theta, kappa):
        self.shape = kappa
        self.rate = -theta
        self.theta = theta
        self.kappa = kappa


class CompoundPoisson():
    def __init__(self, BaseEDM):
        self.base = BaseEDM

    def sample(self, rate):
        ns = np.random.poisson(rate)
        theta = self.base.theta
        kappa = self.base.kappa
        self.base.update(self.base.theta, self.base.kappa * ns)  ## update parameters to sample
        samples = self.base.sample()
        self.base.update(theta, kappa)  ## update back to original number
        return samples


# In[36]:


class exp_dispersion_gamma():
    def __init__(self, BaseEDM, shape_a, rate_a, shape_b, rate_b, latent=11):
        self.shape_a = float(shape_a)
        self.shape_b = float(shape_b)
        self.scale_a = 1.0 / rate_a
        self.scale_b = 1.0 / rate_b
        self.latent = latent
        self.compound_poisson = CompoundPoisson(BaseEDM)

    def sample(self, shape=(1, 1, 100)):
        latent_a = np.random.gamma(self.shape_a, self.scale_a, (self.latent, shape[0], shape[-1]))
        latent_b = np.random.gamma(self.shape_b, self.scale_b, (self.latent, shape[1], shape[-1]))
        self.rmat = self.compound_poisson.sample(np.einsum('kj...,ki...', latent_a, latent_b).T)

    def sample2(self, shape=(1, 1, 100)):
        latent_a = np.random.gamma(self.shape_a, self.scale_a, (self.latent, shape[0], shape[-1]))
        latent_b = np.random.gamma(self.shape_b, self.scale_b, (self.latent, shape[1], shape[-1]))
        self.rmat = self.compound_poisson.sample(np.einsum('kj...,ki...', latent_a, latent_b).T)

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
        self.mean = self.compound_poisson.base.mean() * self.latent * mean_a * mean_b
        self.var = (self.latent) * (
                v0 * self.compound_poisson.base.var() + (self.compound_poisson.base.mean() ** 2) * varr)
        self.corr1 = (self.latent * (self.compound_poisson.base.mean() ** 2) * v1) / self.var
        self.corr2 = (self.latent * (self.compound_poisson.base.mean() ** 2) * v2) / self.var
        return {"t_mean": self.mean,
                "t_var": self.var,
                "t_corr1": self.corr1,
                "t_corr2": self.corr2}


# In[37]:


dat = """a=0.1000   b=1.0000   c=0.1000   d=1.0000    => E=0.25     V=0.55
a=0.0010    b=0.0100    c=0.0100    d=0.1000    => E=0.25     V=253.00
a=0.0001    b=0.0010    c=0.0100    d=0.1000    => E=0.25     V=2525.50
a=0.1000    b=0.1000    c=0.1000    d=0.1000    => E=25.00     V=3025.00
a=1.0000    b=1.0000    c=0.1000    d=0.1000    => E=25.00     V=550.00
a=1000.0000    b=1000.0000    c=1000.0000    d=1000.0000    => E=25.00     V=25.05
a=10.0000    b=2.0000    c=10.0000    d=2.0000    => E=625.00     V=3906.25
a=10.0000    b=1.0000    c=10.0000    d=1.0000    => E=2500.00     V=55000.00""".split("\n")
dat = """a=0.1000   b=1.0000   c=0.1000   d=1.0000    => E=0.25     V=0.55
a=0.1000    b=0.1000    c=0.1000    d=0.1000    => E=25.00     V=3025.00
a=1.0000    b=1.0000    c=0.1000    d=0.1000    => E=25.00     V=550.00
a=1000.0000    b=1000.0000    c=1000.0000    d=1000.0000    => E=25.00     V=25.05
a=10.0000    b=2.0000    c=10.0000    d=2.0000    => E=625.00     V=3906.25
a=10.0000    b=1.0000    c=10.0000    d=1.0000    => E=2500.00     V=55000.00""".split("\n")

dat = """a=0.1000   b=1.0000   c=0.1000   d=1.0000    => E=0.25     V=0.55
a=10.0    b=1.0    c=10.0    d=1.0    => E=110.00     V=1300.0""".split("\n")
exp_vals = list(map(lambda x: [float(y.split("=")[1]) for y in x.split()], map(lambda x: x.split("=>")[0], dat)))

CFGS = [(10, 1, 10, 1), (0.1, 1, 0.1, 1),
         (1, 1, 0.1, 0.1) ]

exp_vals = CFGS

# In[ ]:


class poisson_gamma():
    def __init__(self, shape_a, rate_a, shape_b, rate_b, latent=11, p_obs=0.9):
        self.shape_a = float(shape_a)
        self.shape_b = float(shape_b)
        self.scale_a = 1.0 / rate_a
        self.scale_b = 1.0 / rate_b
        self.latent = latent
        self.p_obs = p_obs

    def sample(self, shape=(1, 1, 100)):
        latent_a = np.random.gamma(self.shape_a, self.scale_a, (self.latent, shape[0], shape[-1]))
        latent_b = np.random.gamma(self.shape_b, self.scale_b, (self.latent, shape[1], shape[-1]))
        self.rmat = np.random.poisson(np.einsum('kj...,ki...', latent_a, latent_b).T)
        return self.rmat

    def sample2(self, shape=(1, 1, 100)):
        latent_a = np.random.gamma(self.shape_a, self.scale_a, (self.latent, shape[0], shape[-1]))
        latent_b = np.random.gamma(self.shape_b, self.scale_b, (self.latent, shape[1], shape[-1]))
        self.rmat = np.random.poisson(np.einsum('kj...,ki...', latent_a, latent_b).T)
        return self.rmat

    def sample_p_obs(self, shape=(1, 1, 100)):
        rmat = self.sample(shape)
        mask = np.random.binomial(1, self.p_obs, shape)
        self.rmat = rmat * mask
        return self.rmat

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


# In[ ]:


def run_exp2(exp, k_latent=25, n_repeats=5, shape=(1000, 1000), sampling_percentage=0.15, p_obs=0.5):
    res = []
    for x in range(n_repeats):
        model = poisson_gamma(exp[0], exp[1], exp[2], exp[3], k_latent, p_obs)
        model.sample((shape[0], shape[1], 1))
        res_d = {"p_a": exp[0], "p_b": exp[1], "p_c": exp[2], "p_d": exp[3], "t_latent_dim": k_latent, "p_obs": p_obs}
        res_d.update(model.theoretical())
        res_d.update(empirical(model.rmat[:, :, 0], None, ns=int(sampling_percentage * shape[0] * shape[1])))
        res.append(res_d)
    return pd.DataFrame(res)


def run_exp3(exp, k_latent=25, n_repeats=5, shape=(1000, 1000), sampling_percentage=0.15, p_obs=0.5):
    res = []
    for x in range(n_repeats):
        model = poisson_gamma(exp[0], exp[1], exp[2], exp[3], k_latent, p_obs)
        model.sample_p_obs((shape[0], shape[1], 1))
        res_d = {"p_a": exp[0], "p_b": exp[1], "p_c": exp[2], "p_d": exp[3], "t_latent_dim": k_latent, "p_obs": p_obs}
        res_d.update(model.theoretical())
        res_d.update(empirical(model.rmat[:, :, 0], None, ns=int(sampling_percentage * shape[0] * shape[1])))
        res.append(res_d)
    return pd.DataFrame(res)


def exp_worker(exp, k_latent=25, shape=(1000, 1000), sampling_percentage=0.15, p_obs=0.5):
    model = poisson_gamma(exp[0], exp[1], exp[2], exp[3], k_latent, p_obs)
    model.sample_p_obs((shape[0], shape[1], 1))
    res_d = {"p_a": exp[0], "p_b": exp[1], "p_c": exp[2], "p_d": exp[3], "t_latent_dim": k_latent, "p_obs": p_obs}
    res_d.update(model.theoretical())
    res_d.update(empirical(model.rmat[:, :, 0], None, ns=int(sampling_percentage * shape[0] * shape[1])))
    return res_d


from multiprocessing.pool import ThreadPool
import concurrent.futures


def run_exp_async(exp, k_latent=25, n_repeats=5, shape=(1000, 1000), sampling_percentage=0.15, p_obs=0.5):
    workers = []
    res = []
    pool = ThreadPool(processes=1)
    for x in range(n_repeats):
        async_result = pool.apply_async(exp_worker,
                                        (exp, k_latent, shape, sampling_percentage, p_obs))  # tuple of args for foo
        workers.append(async_result)
    for async_result in workers:
        res_d = async_result.get()
        res.append(res_d)
    return pd.DataFrame(res)


def run_exp_async2(exp, k_latent=25, n_repeats=5, shape=(1000, 1000), sampling_percentage=0.15, p_obs=0.5):
    workers = []
    res = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for x in range(n_repeats):
            async_res = executor.submit(exp_worker,
                                        exp, k_latent, shape, sampling_percentage, p_obs)  # tuple of args for foo
            workers.append(async_res)
        for async_result in workers:
            res_d = async_result.result()
            res.append(res_d)
    return pd.DataFrame(res)


# In[ ]:
### Experiment 1####
N_EXP = 30  # number of experiments per hyperparameter setting
NS = 0.15
N_COLS = 1000
N_ROWS = 1000
#### Experiment 2####
N_EXP = 30  # number of experiments per hyperparameter setting
NS = 0.20
N_COLS = 1000
N_ROWS = 1000
#### Experiment 3####
N_EXP = 30  # number of experiments per hyperparameter setting
NS = 0.20
N_COLS = 1000
N_ROWS = 1000

k_latent_vals = list(range(25, 550, 50))
k_latent_vals = list(range(50, 550, 100))
k_latent_vals = [25,50,75,100,125,150]
p_obs = [1.0, 0.98, 0.96, 0.94, 0.92, 0.9]
rand_n = np.random.randint(1000)

#### Experiment test####
N_EXP = 10  # number of experiments per hyperparameter setting
NS = 0.20
N_COLS = 100
N_ROWS = 100

k_latent_vals = [25,150]
p_obs = [1.0, 0.95, 0.9, 0.85]
rand_n = np.random.randint(1000)

#### Experiment4####
N_EXP = 30  # number of experiments per hyperparameter setting
NS = 0.20
N_COLS = 1000
N_ROWS = 1000

k_latent_vals = [50,100,150]
p_obs = [1.0, 0.96, 0.92, 0.88]
rand_n = np.random.randint(1000)

#### Experiment v2 test####
N_EXP = 10  # number of experiments per hyperparameter setting
NS = 0.20
N_COLS = 100
N_ROWS = 100

k_latent_vals = [25,150]
p_obs = [1.0, 0.95, 0.9, 0.85]
rand_n = np.random.randint(1000)


#### Experiment v2 1####
N_EXP = 30  # number of experiments per hyperparameter setting
NS = 0.20
N_COLS = 1000
N_ROWS = 1000

k_latent_vals = [25,50,75,100,125,150]
p_obs = [1.0, 0.98, 0.96, 0.94, 0.92, 0.9]
rand_n = np.random.randint(1000)


def experiment_worker(k_latent_vals, p_obs, hyperparameters, N_EXP, N_COLS, N_ROWS, NS, exp_n):
    print("Experiment :" + str(exp_n))
    t1 = time.time()
    df_temp = pd.DataFrame(
        columns=['e_corr1', 'e_corr2', 'e_latent_dim', 'e_mean', 'e_var', 'p_a', 'p_b', 'p_c', 'p_d', 't_corr1',
                 't_corr2',
                 't_latent_dim', 't_mean', 't_var', 'p_obs'])
    for k_latent in k_latent_vals:
        for obs in p_obs:
            t2 = time.time()
            temp = run_exp3(hyperparameters, k_latent=k_latent, n_repeats=N_EXP, shape=(N_COLS, N_ROWS),
                            sampling_percentage=NS, p_obs=obs)
            print(" p_obs: " + str(obs) + " true_k: " + str(k_latent) + " 'mean' :" + str(
                temp['e_latent_dim'].mean()) + ", 'std':" + str(
                temp['e_latent_dim'].std()) + ", 'time' : " + str((time.time() - t2) / 60.0))
            df_temp = df_temp.append(temp)
        print("ended in " + str((time.time() - t1) / 60.0) + " minutes")
    df_temp.to_csv(
        "exp_poisson_gamma_bin_e" + str(rand_n) + "_" + str(int(NS * 100)) + "_" + str(N_EXP) + "_cols_" + str(N_COLS) + "_rows_" + str(
            N_ROWS) + "_c_" + str(exp_n) + ".csv")
    return df_temp


def experiment_worker_k(k_latent, df_temp, p_obs, hyperparameters, N_EXP, N_COLS, N_ROWS, NS):
    for obs in p_obs:
        t2 = time.time()
        temp = run_exp3(hyperparameters, k_latent=k_latent, n_repeats=N_EXP, shape=(N_COLS, N_ROWS),
                        sampling_percentage=NS, p_obs=obs)
        print(" p_obs: " + str(obs) + " true_k: " + str(k_latent) + " 'mean' :" + str(
            temp['e_latent_dim'].mean()) + ", 'std':" + str(
            temp['e_latent_dim'].std()) + ", 'time' : " + str((time.time() - t2) / 60.0))
        df_temp = df_temp.append(temp)
    return df_temp


def experiment_worker_parallel_k(k_latent_vals, p_obs, hyperparameters, N_EXP, N_COLS, N_ROWS, NS, exp_n):
    print("Experiment :" + str(exp_n))
    t1 = time.time()
    df_temp = pd.DataFrame(
        columns=['e_corr1', 'e_corr2', 'e_latent_dim', 'e_mean', 'e_var', 'p_a', 'p_b', 'p_c', 'p_d', 't_corr1',
                 't_corr2',
                 't_latent_dim', 't_mean', 't_var', 'p_obs'])
    with concurrent.futures.ThreadPoolExecutor() as executor_k:
        for temp in executor_k.map(lambda k_latent:
                                   experiment_worker_k(k_latent, df_temp, p_obs, hyperparameters, N_EXP, N_COLS, N_ROWS, NS),
                                   k_latent_vals):
            df_temp = df_temp.append(temp)
            print("ended in " + str((time.time() - t1) / 60.0) + " minutes")
    df_temp.to_csv(
        "exp_poisson_gamma_bin_e" + str(rand_n) + "_" + str(int(NS * 100)) + "_" + str(N_EXP) + "_cols_" + str(N_COLS) + "_rows_" + str(
            N_ROWS) + "_c_" + str(exp_n) + ".csv")
    return df_temp


workers = []
df = pd.DataFrame(
    columns=['e_corr1', 'e_corr2', 'e_latent_dim', 'e_mean', 'e_var', 'p_a', 'p_b', 'p_c', 'p_d', 't_corr1', 't_corr2',
             't_latent_dim', 't_mean', 't_var', 'p_obs'])
with concurrent.futures.ThreadPoolExecutor() as executor:
    #for res_df in executor.map(lambda par:
    #                           experiment_worker(k_latent_vals, p_obs, par[1], N_EXP, N_COLS, N_ROWS, NS, par[0]),
    #                           enumerate(exp_vals)):
    for res_df in executor.map(lambda par:
                               experiment_worker_parallel_k(k_latent_vals, p_obs, par[1], N_EXP, N_COLS, N_ROWS, NS, par[0]),
                               enumerate(exp_vals)):
        df = df.append(res_df)

df.to_csv(
    "final_exp_poisson_gamma_bin_v2_e" + str(rand_n) + "_" + str(int(NS * 100)) + "_" + str(N_EXP) + "_cols_" + str(N_COLS) + "_rows_" + str(
        N_ROWS) + "_c_" + ".csv")
