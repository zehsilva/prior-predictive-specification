import multiprocessing
import multiprocessing.pool


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonProcessPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


###############################################################################

import sys
import numpy as np
import pandas as pd
import pmf_objectives as model

posterior_objective = model.posterior_objective_psisloo

POSTERIOR_OBJECTIVE_CSV = sys.argv[0]+".csv" 
COLUMNS = ["K", "a", "b", "c", "d", "seed"]
configurations = []


SEARCHED_VALS = [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0, 1000000000.0]
SEARCHED_VALS_b = SEARCHED_VALS
SEARCHED_VALS_d = SEARCHED_VALS


K, a, c = model.K, model.THETA_SHAPE, model.BETA_SHAPE
for seed in [0,1,2]:
  for b in SEARCHED_VALS_b:
    for d in SEARCHED_VALS_d:
      configurations.append( (K, a, b, c, d, seed) )


df = pd.DataFrame(configurations).rename(columns=dict(enumerate(COLUMNS)))
df.to_csv(POSTERIOR_OBJECTIVE_CSV+".cfgs", index=False, header=True)


###############################################################################

def evalobj(cfg):
    try: obj = posterior_objective(*cfg)["obj"]
    except (ValueError, RuntimeError) as e:
        sys.stderr.write("ERROR @ cfg=%s MSG: %s\n" % (cfg, e))
        obj = None
    print("[cfg=%s] => %s" % (cfg, obj))    
    return list(cfg)+[obj]


try: processes = int(sys.argv[1])
except: processes = 1
print("%i configurations to be calculated on %s cores. out: %s" % (len(configurations), processes, POSTERIOR_OBJECTIVE_CSV))
if processes>1:
    pool = NoDaemonProcessPool(processes = processes)
    results = pool.map(evalobj, configurations) 
    pool.close()
    pool.join()
else:
    results = []
    for i, c in enumerate(configurations):
        if i%100==0: print("PROGRESS: %i/%i" % (i, len(configurations)))
        results.append( evalobj(c) )


df = pd.DataFrame(results).rename(columns=dict(enumerate(COLUMNS+["obj"])))
df.to_csv(POSTERIOR_OBJECTIVE_CSV, index=False, header=True)




