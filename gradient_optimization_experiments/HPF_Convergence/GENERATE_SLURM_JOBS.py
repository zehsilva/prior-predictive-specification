
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
    f.write("#SBATCH --mem-per-cpu=3000\n")
    f.write("#SBATCH -t 6:00:00\n")
    f.write("srun %s\n" % (cmd))
    f.close()


ABCD_CFGS = [(1.0, 100.0, 10.0, 1.0, 100.0, 10.0),
             (0.1, 100.0, 1.0, 1.0, 100.0, 1.0),
             (50.0, 5000.0, 10.0, 1.0 ,  5000.0 , 1.0),
             (1.0, 100.0, 1.0, 10.0, 10.0, 1.0), 
             (450.0, 4500.0, 100.0, 10.0 ,  400.0 , 1.0), 
             (50.0, 50.0, 1.0, 1.0 ,  50.0 , 1.0),]


ix = 0 


for a0, ap0, bp0, c0, cp0, dp0 in ABCD_CFGS:
  for S, SY in [(1000,10)]:
    for E, V in [(10.0,-1),(100.0,-1),(1000.0,-1)]:
      for LAMBDA in [0.0]:
        for SEED in range(5):
          name = "%s" % (ix); 
          add_job(name, "python hpf_sgd_optimization.py \
                K=25,D=[E],a=%.3f,ap=%.3f,bp=%.3f,c=%.3f,cp=%.3f,dp=%.3f,SEED=%s,S=%s,SY=%s,E=%s,V=%s,LAMBDA=%s,ID=[%s],NITER=20000" % 
                (a0, ap0, bp0, c0, cp0, dp0,SEED,S,SY,E,V,LAMBDA,ix))
          ix += 1;

for a0, ap0, bp0, c0, cp0, dp0 in ABCD_CFGS:
  for S, SY in [(1000,10)]:
    for E, V in [(10.0,100.0),(100.0,10.0),(10.0,1000.0),(1000.0,10.0)]:
      for LAMBDA in [0.0]:
        for SEED in range(5):
          name = "%s" % (ix); 
          add_job(name, "python hpf_sgd_optimization.py \
                K=25,D=[EV],a=%.3f,ap=%.3f,bp=%.3f,c=%.3f,cp=%.3f,dp=%.3f,SEED=%s,S=%s,SY=%s,E=%s,V=%s,LAMBDA=%s,ID=[%s],NITER=50000" % 
                (a0, ap0, bp0, c0, cp0, dp0,SEED,S,SY,E,V,LAMBDA,ix))
          ix += 1;






open("RUN_ALL.sh","w").write(jobs)


