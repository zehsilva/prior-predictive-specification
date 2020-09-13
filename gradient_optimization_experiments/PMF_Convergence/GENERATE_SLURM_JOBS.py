
import numpy as np



jobs = ""
def add_job(name, cmd):
    global jobs
    jobs += "\nsbatch %s.job" % name  
    f = open("%s.job" % name, "w")
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --job-name=%s\n" % name )
    f.write("#SBATCH -o %s.log\n" % name )
    f.write("#SBATCH -p short\n")
    f.write("#SBATCH -c 1\n")
    f.write("#SBATCH --mem-per-cpu=1000\n")
    f.write("#SBATCH -t 4:00:00\n")
    f.write("srun %s\n" % (cmd))
    f.close()


ABCD_CFGS = [(0.1, 1, 0.1, 1),(0.001, 0.01, 0.01, 0.1),(0.0001, 0.001, 0.01, 0.1),  
             (0.1, 0.1, 0.1, 0.1),(1, 1, 0.1, 0.1),
             (10,2,10,2),(10, 1, 10, 1), (1000, 1000, 1000, 1000),]


ix = 0 


for a,b,c,d in ABCD_CFGS:
  for S, SY in [(1000,10)]:
    for E, V in [(10.0,-1),(100.0,-1),(1000.0,-1)]:
      for LAMBDA in [0.0]:
        for SEED in range(10):
          name = "%s" % (ix); 
          add_job(name, "python pmf_sgd_optimization.py \
                K=25,D=[E],a=%s,b=%s,c=%s,d=%s,SEED=%s,S=%s,SY=%s,E=%s,V=%s,LAMBDA=%s,ID=[%s],NITER=20000" % 
                (a,b,c,d,SEED,S,SY,E,V,LAMBDA,ix))
          ix += 1;

for a,b,c,d in ABCD_CFGS:
  for S, SY in [(1000,10)]:
    for E, V in [(10.0,100.0),(100.0,10.0),(10.0,1000.0),(1000.0,10.0)]:
      for LAMBDA in [0.0]:
        for SEED in range(10):
          name = "%s" % (ix); 
          add_job(name, "python pmf_sgd_optimization.py \
                K=25,D=[EV],a=%s,b=%s,c=%s,d=%s,SEED=%s,S=%s,SY=%s,E=%s,V=%s,LAMBDA=%s,ID=[%s],NITER=20000" % 
                (a,b,c,d,SEED,S,SY,E,V,LAMBDA,ix))
          ix += 1;


open("RUN_ALL.sh","w").write(jobs)


