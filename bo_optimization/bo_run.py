#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pickle
import sys

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from robo.fmin import bayesian_optimization


# Global Configuration
try:
  CFG = sys.argv[1]
  CFG = CFG.replace(".py", "")
  SEED = int(sys.argv[2])
except:
  logger.info("USAGE: python run.py CONFIGURATION_FILE_PATH SEED_VALUE")
  sys.exit(-1)


PREFIX = CFG+("_%i" % SEED)
CFG = CFG+".py"
OUTFILE = "%s_bo_posterior" % PREFIX
logger.info("OPENING CODE FROM CFG=%s (SEED=%i) => OUTPUT:%s" % (CFG, SEED, OUTFILE))
exec(open(CFG).read())


# BlackBox BO over posterior
rng = np.random.RandomState(SEED)
res = bayesian_optimization(objective_function, lower, upper, num_iterations=num_iterations, X_init=X_init, Y_init=Y_init, n_init=n_init, rng=rng,) 


