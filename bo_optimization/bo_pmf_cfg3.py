"""Configuration of PMF with PSIS-LOO for BO optimization."""

import pmf_objectives as model

import numpy as np
import time

import logging
logger = logging.getLogger(__name__)


# BO cofiguration
n_init = 2
num_iterations = n_init + 1000
lower = np.array([1, -4,-4,-4,-4])
upper = np.array([100, 2,2,2,2])
X_init = None
Y_init = None



def abcd2musigma(a, b, c, d, env=np):    
    return a/b, env.sqrt(a)/b, c/d, env.sqrt(c)/d
  

def musigma2abcd(mut, sigmat, mub, sigmab):    
    a,b = (mut*mut)/(sigmat*sigmat), mut/(sigmat*sigmat)
    c,d = (mub*mub)/(sigmab*sigmab), mub/(sigmab*sigmab)
    return a, b, c, d



def objective_function(x):
    """Uses certain global variable OUTFILE."""
    start_time = time.time()

    #K, a, b, c, d = int(np.round(x[0])), 10**x[1], 10**x[2], 10**x[3], 10**x[4] # parametrization abcd
    K, mut, sigmat, mub, sigmab = int(np.round(x[0])), 10**x[1], 10**x[2], 10**x[3], 10**x[4] #parametrization mu-sigma
    a, b, c, d = musigma2abcd(mut, sigmat, mub, sigmab)

    logger.info("[objective_function] Evaluating at %s %s %s %s %s" % (K,a,b,c,d))  
    res = model.posterior_objective_psisloo(K=K, a=a, b=b, c=c, d=d, seed=123, 
                                            NSAMPLES=100, verbose=True, 
                                            psis_data=model.DATA_TRAIN)
    obj = res["obj"] # train data LOO
    fitting_time = time.time()

    loo = res["model"].psis(model.DATA_TEST, NSAMPLES=100) # test data LOO
    end_time = time.time()

    logger.info("[objective_function] writting to %s" % (OUTFILE+".csv"))
    f = open(OUTFILE+".csv", "a")
    f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (K,a,b,c,d,obj,loo,start_time,fitting_time,end_time))
    f.close()

    logger.info("[objective_function] Evaluating at %s %s %s %s %s => train:%s test:%s" % (K,a,b,c,d,obj,loo))  
    return -obj # BO minimizes its objective


