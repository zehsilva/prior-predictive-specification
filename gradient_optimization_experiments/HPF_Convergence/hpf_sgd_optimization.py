#!/usr/bin/env python
# coding: utf-8

# # HPF: matching requested expectation and variance using SGD

# The notebook demonstrates how priors matching requested values of prior predictive expectation and/or variance can be found for Hierarchical Poisson Matrix Factorization model using SGD.

# ## Setup 
# 

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals


# In[2]:


import tensorflow as tf
#tf.enable_eager_execution()
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
#tfe = tf.contrib.eager


# In[3]:


tf.__version__, tfp.__version__


# In[4]:


tf.executing_eagerly()


# In[5]:


import time
import numpy as np
import pandas as pd


# In[6]:


from aux import *


# # Configuration

# In[7]:


args = parse_script_args()


# In[8]:


np.random.seed(int(time.time()))
EID = np.random.randint(10000) # experiment id


# In[9]:


SEED = args.get("SEED", 129)
np.random.seed(SEED)


# In[10]:


K = args.get("K", 25) # factorization dimensions


# In[11]:


NITER = args.get("NITER", 10000) + 1 # how many iterations
LR = args.get("LR", 0.1) # learning rate
LAMBDA = args.get("LAMBDA", 0.0) # regularization


# In[12]:


# requested values
DESIRED_EXPECTATION = args.get("E", 25)
DESIRED_VARIANCE = args.get("V", 550)

# discrepancy measure: care about E=only expectation, V=only variance, EV=both
D = args.get("D", "EV")
if D=="E": DESIRED_VARIANCE = -1
if D=="V": DESIRED_EXPECTATION = -1
    
discrepancy_expectation = lambda expectation, variance: (expectation-DESIRED_EXPECTATION)**2 
discrepancy_variance    = lambda expectation, variance: (variance-DESIRED_VARIANCE)**2 
discrepancy_exp_var     = lambda expectation, variance: discrepancy_variance(expectation, variance) +                                                         discrepancy_expectation(expectation, variance)

NAME2DISCREPANCY = {"E":  discrepancy_expectation, "EV": discrepancy_exp_var, "V": discrepancy_variance}
discrepancy = NAME2DISCREPANCY[D]


# In[13]:


# sampling strategy: independent or the same samples for E & V
SAMPLING = args.get("SAMPLING", 0)
if SAMPLING==0 and LAMBDA<=0.0:
    if D=="E": SAMPLING = 2 # if V is not used
    if D=="V": SAMPLING = 3 # if E is not used

NSAMPLES_LATENT = args.get("S", 1000) # how many samples of latent variables
NSAMPLES_OUTPUT = args.get("SY", 10)  # how many samples of outputs for each latent

LOG_SCORE_DERIVATIVES = True


# In[14]:


# optimization initial values 

a0, ap0, bp0 = args.get("a", 3.0), args.get("ap", 3.0), args.get("bp", 1.0)
c0, cp0, dp0 = args.get("c", 3.0), args.get("cp", 3.0), args.get("dp", 1.0)


# which parameter space
PARAMETRIZATION = args.get("TRANSFORMATION", "abcd") # musigma/abcd
# how to transform from unbounded space to a bounded one and vice versa
VARIABLES_TRANSFORMATION = args.get("TRANSFORMATION", "softplus") # softplus/exp/pow10
# which of the parameters to train
TRAINABLE = args.get("TRAIN", "012345") 


# In[15]:


DESCRIPTION = dict2str(globals()).replace("NSAMPLES", "S").replace("DESIRED_", "").replace("EXPECTATION", "E").replace("VARIANCE", "V").replace("TRAINABLE", "TT")
DESCRIPTION = "a=%s ap=%s bp=%s c=%s cp=%s dp=%s %s" % (a0, ap0, bp0, c0, cp0, dp0,DESCRIPTION)
print("DESCRIPTION: %s" % DESCRIPTION)

ID = str(args.get("ID", DESCRIPTION))


# # Parameters transformation

# For HPF, we consider only one parametrization (the one from the original paper),
# so the parametrization is an identity.
# However, we still need to perform a transformation from unconstrained space.

# In[16]:


def abcdef2abcdef(a, b, c, d, e, f, env=tf):    
    return a, b, c, d, e, f


# In[17]:


#from tensorflow.contrib.distributions import softplus_inverse
#softplus_inverse = lambda v: tf.log(tf.exp(v)-1.0)
softplus_inverse = tfp.math.softplus_inverse
NAME2TRANSFORMATION = {"softplus": (tf.nn.softplus, softplus_inverse), 
                       "exp": (tf.exp, tf.math.log),
                       "pow10": (lambda v: tf.math.pow(10.0, v), lambda v: tf.math.log(v)*0.434294482)}
forward_transformation, backward_transformation = NAME2TRANSFORMATION[VARIABLES_TRANSFORMATION]

print("forward_transformation:=%s backward_transformation:=%s" % 
      (forward_transformation.__name__, backward_transformation.__name__))
#print("  transformation prec:", np.array(backward_transformation(forward_transformation(a0))-a0))

NAME2PARAMETRIZATION = {"abcd": (abcdef2abcdef, abcdef2abcdef)}
forward_parametrization, backward_parametrization = NAME2PARAMETRIZATION[PARAMETRIZATION]

print("forward_parametrization:=%s backward_parametrization:=%s " % 
      (forward_parametrization.__name__, backward_parametrization.__name__))
#print("  parametrization prec:", forward_parametrization(*backward_parametrization(a0,b0,c0,d0))-np.array([a0,b0,c0,d0]))
#assert (np.round(inverse_parametrization(*parametrization(a0,b0,c0,d0))-np.array([a0,b0,c0,d0]),2) ==0).all()


# # Expectation & variance estimators

# In[18]:


from hpf_model import create_moments_estimator, empirical_Ey_and_Ey2_tf, empirical_Ey_and_Ey2_tf_logscore
theoretical_moments = create_moments_estimator(K=K, ESTIMATOR_NO=-1)
empirical_moments   = create_moments_estimator(K=K, ESTIMATOR_NO=SAMPLING, 
    empirical_Ey_and_Ey2=empirical_Ey_and_Ey2_tf_logscore if LOG_SCORE_DERIVATIVES else empirical_Ey_and_Ey2_tf)
print("empirical_moments := %s" % empirical_moments.__name__)


# In[19]:


e, v = theoretical_moments(a0, ap0, bp0, c0, cp0, dp0)
print("Initialization: a=%.3f\tap=%.3f\tbp=%.3f\ta=%.3f\tap=%.3f\tbp=%.3f\t=> E=%.4f \tV=%.4f" % 
      (a0, ap0, bp0, c0, cp0, dp0, e, v))


# # Find hyperparameters matching desired values

# In[20]:


#tf.random.set_random_seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)


# In[21]:


# init (unbounded) params
float2 = lambda v: tf.Variable(v, dtype=tf.float64)

a0tf, ap0tf, bp0tf, c0tf, cp0tf, dp0tf = float2(a0), float2(ap0), float2(bp0), float2(c0), float2(cp0), float2(dp0)
i0, i1, i2, i3, i4, i5 = backward_parametrization(a0tf, ap0tf, bp0tf, c0tf, cp0tf, dp0tf, env=tf)

p0u = tf.Variable(backward_transformation(i0), name="0", dtype=tf.float64) 
p1u = tf.Variable(backward_transformation(i1), name="1", dtype=tf.float64) 
p2u = tf.Variable(backward_transformation(i2), name="2", dtype=tf.float64) 
p3u = tf.Variable(backward_transformation(i3), name="3", dtype=tf.float64) 
p4u = tf.Variable(backward_transformation(i4), name="4", dtype=tf.float64) 
p5u = tf.Variable(backward_transformation(i5), name="5", dtype=tf.float64) 


# In[22]:


trainable_variables = [var for var in [p0u, p1u, p2u, p3u, p4u, p5u] if var.name.split(":")[0] in TRAINABLE]
#optimizer = tf.train.AdamOptimizer(learning_rate=LR)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
start = time.time()
computation_time = 0.0
results = []


# In[23]:


best_loss = float("inf")
best_hyperparameters = -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
best_E, best_V = -1.0, -1.0


# In[24]:


for iteration in range(NITER):
  
    ##########################################################################################################
    # Optimization
    iteration_start_time = time.time()
        
    with tf.GradientTape() as tape:        
        p0,p1,p2 = forward_transformation(p0u), forward_transformation(p1u), forward_transformation(p2u)  
        p3,p4,p5 = forward_transformation(p3u), forward_transformation(p4u), forward_transformation(p5u)  
        
        a, ap, bp, c, cp, dp =  forward_parametrization(p0,p1,p2,p3,p4,p5)           
        expectation, variance = empirical_moments(a, ap, bp, c, cp, dp, NSAMPLES_LATENT, NSAMPLES_OUTPUT)        
        loss = discrepancy(expectation, variance) -(LAMBDA*variance if LAMBDA>0 else 0.0) # just to be sure
    
    grads = tape.gradient(loss, trainable_variables)              
    optimizer.apply_gradients(zip(grads, trainable_variables))#, global_step=tf.train.get_or_create_global_step())
    computation_time += (time.time()-iteration_start_time)
    
    if loss<best_loss: #TODO Can do better by evaluating here the loss with more samples
        #print("%i/%i: best: old=%.2f new=%.2f" % (iteration, NITER, best_loss, loss.numpy()))
        best_loss = loss.numpy()
        best_hyperparameters = a.numpy(), ap.numpy(), bp.numpy(), c.numpy(), cp.numpy(), dp.numpy()
        
    ##########################################################################################################
    # Reporting:
    failed = not is_valid(np.array(expectation))
    elapsed = time.time() - start
    if (failed) or (iteration%10==0 and iteration<100) or (iteration%100==0): 
        eval_start = time.time()
        best_E, best_V = theoretical_moments(*best_hyperparameters)
        expectation_exact, variance_exact = theoretical_moments(a, ap, bp, c, cp, dp)        
        
        r = (ID, D, SEED,
             a0, ap0, bp0, c0, cp0, dp0, NITER, LR,
             K, NSAMPLES_LATENT, NSAMPLES_OUTPUT, 
             DESIRED_EXPECTATION, DESIRED_VARIANCE, LAMBDA,
             ((NITER-1) if failed else iteration), computation_time, 
             a.numpy(), ap.numpy(), bp.numpy(), c.numpy(), cp.numpy(), dp.numpy(),
             loss.numpy(), expectation_exact, variance_exact, expectation.numpy(), variance.numpy(),
             best_loss, *best_hyperparameters, best_E, best_V,
             failed)
        results.append(r)

    #if (failed) or (iteration<10) or (iteration<1000 and iteration%100==0)  or (iteration%1000==0):         
        print("[%.2f][%.2f] best:\tloss:%.1f\thyperparams: a=%.3f ap=%.3f bp=%.3f a=%.3f ap=%.3f bp=%.3f => E=%.3f V=%.3f" % 
              (computation_time, elapsed, best_loss, *best_hyperparameters, best_E, best_V))        
        print("[%.2f][%.2f] %i/%i,\tloss:%.1f\thyperparams: a=%.3f ap=%.3f bp=%.3f a=%.3f ap=%.3f bp=%.3f" % 
              (computation_time, elapsed, iteration, NITER, loss.numpy(), 
               a.numpy(), ap.numpy(), bp.numpy(), c.numpy(), cp.numpy(), dp.numpy()))                
        print(" empirical: E: %.2f V: %.2f" % (expectation, variance) )    
        print(" theoretic: E: %.2f V: %.2f" % (expectation_exact, variance_exact) )
        print(" desired: E: %.2f V: %.2f (eval time=%.2f)" % (DESIRED_EXPECTATION, DESIRED_VARIANCE, time.time()-eval_start) )                
        sys.stdout.flush()

    if failed:
        print("Error: ran into invalid values!")
        break

    if (iteration%1000==0 or iteration>=NITER-1):
        path = ID+".csv"
        print("Saving results to: %s" % path)
        df = pd.DataFrame(results)
        CN = ["ID", "D", "SEED", "a0", "ap0", "bp0", "c0", "cp0", "dp0", "NITER", "LR", "K", "S", "SY", "E", "V", "LAMBDA",
              "iteration", "elapsed", "a", "ap", "bp", "c", "cp", "dp", "loss", "Et", "Vt", "Ee", "Ve", 
              "best_loss", "best_a", "best_ap", "best_bp", "best_c", "best_cp", "best_dp", "best_E", "best_V", "failed"]
        df.rename(columns=dict(enumerate(CN)), inplace=True)
        df.to_csv(path, header=True, index=False);

    if (DESIRED_EXPECTATION<0 or abs(best_E-DESIRED_EXPECTATION)/DESIRED_EXPECTATION<0.05) and \
       (DESIRED_VARIANCE<0 or abs(best_V-DESIRED_VARIANCE)/DESIRED_VARIANCE<0.05):
        NITER = min(NITER, iteration+1000)
    if iteration>=NITER: break



