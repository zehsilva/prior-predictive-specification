
import pmf, hetrec_load, pmf_sampling
import numpy as np



print("[pmf_objectives] Loading user_artists.dat (default train-test split).")
DATA, DATAFULL, DATA_TRAIN, DATA_TEST = hetrec_load.load_train_test('../data/user_artists.dat', 0.9)
NROWS, NCOLS = DATAFULL.shape
EMPIRICAL_ESTIMATES = pmf_sampling.empirical_v2(DATAFULL)
K, THETA_SHAPE, THETA_RATE, BETA_SHAPE, BETA_RATE = pmf_sampling.recover_all_gamma(EMPIRICAL_ESTIMATES['e_corr_vals'][0],
                                                                                   EMPIRICAL_ESTIMATES['e_corr_vals'][1],
                                                                                   EMPIRICAL_ESTIMATES['e_mean'],
                                                                                   EMPIRICAL_ESTIMATES['e_var'], 1.0)

bd = EMPIRICAL_ESTIMATES['e_mean']/EMPIRICAL_ESTIMATES['e_var'] * np.sqrt( (EMPIRICAL_ESTIMATES['e_corr_vals'][0]*EMPIRICAL_ESTIMATES['e_corr_vals'][1]) / (THETA_SHAPE*BETA_SHAPE) )
print("[pmf_objectives] Default data: stats=%s, nrows=%s ncols=%s => K=%s, a=%s, b=%s, c=%s, d=%s, b*d=%s" % 
                       (EMPIRICAL_ESTIMATES, DATAFULL.shape[0], DATAFULL.shape[1], 
                        K,THETA_SHAPE,THETA_RATE,BETA_SHAPE,BETA_RATE,bd))


def posterior_objective_psisloo(K=K, a=THETA_SHAPE, b=THETA_RATE, c=BETA_SHAPE, d=BETA_RATE, 
                                seed=123, NSAMPLES=100, verbose=True,
                                train_data=DATA_TRAIN, psis_data=DATA_TEST,
                                n_rows=DATAFULL.shape[0], n_cols=DATAFULL.shape[1]):
    """PSIS-LOO evaluated for concentrations: of theta=a and of beta=c and rates: of theta=b and of beta=d."""
    print("[posterior_objective_psisloo] K=%s, a=%s, b=%s, c=%s, d=%s, seed=%s" % (K,a,b,c,d,seed))
    poi2 = pmf.PoissonMF(n_components=K, max_iter=1000, tol=0.0001, smoothness=100,                     
                     random_state=seed, verbose=verbose, allone=False,
                     theta_a=a, theta_b=b, beta_a=c, beta_b=d)
    poi2.fit(train_data, n_rows=n_rows, n_cols=n_cols)
    loo = poi2.psis(psis_data, NSAMPLES)
    return {"obj": loo, "model": poi2}

