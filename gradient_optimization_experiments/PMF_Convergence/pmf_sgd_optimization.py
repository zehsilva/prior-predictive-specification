#!/usr/bin/env python
# coding: utf-8

# # PMF: matching requested expectation and variance using SGD

# The notebook demonstrates how priors matching requested values of prior predictive expectation and/or variance can be found for Poisson Matrix Factorization model using SGD.

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


N, M = 1, 1
K = args.get("K", 25) # factorization dimensions


# In[11]:


NITER = args.get("NITER", 20000) + 1 # how many iterations
LR = args.get("LR", 0.1) # learning rate
LAMBDA = args.get("LAMBDA", 0.0) # regularization


# In[12]:


# requested values
DESIRED_EXPECTATION = args.get("E", 50)
DESIRED_VARIANCE = args.get("V", 1000)

# discrepancy measure: care about E=only expectation, V=only variance, EV=both
D = args.get("D", "EV") 
if D=="E": DESIRED_VARIANCE = -1
if D=="V": DESIRED_EXPECTATION = -1
    
# (mean (tf.reduce_mean) should be taken if more than one cell used)
discrepancy_expectation = lambda expectation, variance: (expectation-DESIRED_EXPECTATION)**2 
discrepancy_variance    = lambda expectation, variance: (variance-DESIRED_VARIANCE)**2 
discrepancy_exp_var     = lambda expectation, variance: discrepancy_variance(expectation, variance) +                                                         discrepancy_expectation(expectation, variance) #!

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


# optimization initial values (always in terms of abcd parametrization)

a0, b0, c0, d0 = args.get("a", None), args.get("b", None), args.get("c", None), args.get("d", None) # random init

# if not set choose randomly mu-s and sigma-s from 10**e, e in [1e-3, 1e3]
if a0 is None or a0<=0:
    mut = 10**(np.random.random()*6.0-3.0) 
    sigmat = 10**(np.random.random()*6.0-3.0) 
    mub = 10**(np.random.random()*6.0-3.0) 
    sigmab = 10**(np.random.random()*6.0-3.0) 
    a0,b0 = (mut*mut)/(sigmat*sigmat), mut/(sigmat*sigmat)
    c0,d0 = (mub*mub)/(sigmab*sigmab), mub/(sigmab*sigmab)
    print("Random initialization: mut=%.3f st=%.3f mub=%.3f sb=%.3f" % (mut, sigmat, mub, sigmab))

    
# which parameter space
PARAMETRIZATION = args.get("PARAMETRIZATION", "musigma") # musigma/abcd
# how to transform from unbounded space to a bounded one and vice versa
VARIABLES_TRANSFORMATION = args.get("TRANSFORMATION", "softplus") # softplus/exp/pow10
# which of the parameters to train
TRAINABLE = args.get("TRAIN", "0123") 


# In[15]:


DESCRIPTION = dict2str(globals()).replace("NSAMPLES", "S").replace("DESIRED_", "").replace("EXPECTATION", "E").replace("VARIANCE", "V").replace("TRAINABLE", "TT")
DESCRIPTION = "a=%.3f b=%.3f c=%.3f d=%.3f %s" % (a0,b0,c0,d0,DESCRIPTION)
print("DESCRIPTION: %s" % DESCRIPTION)

ID = str(args.get("ID", DESCRIPTION))


# # Parameterization and transformation

# We consider two problem parametrizations: either abcd or mu-sigma.
# Also, we need to perform a transformation from unconstrained space.

# In[16]:


def abcd2musigma(a, b, c, d, env=tf):    
    return a/b, env.sqrt(a)/b, c/d, env.sqrt(c)/d
  

def musigma2abcd(mut, sigmat, mub, sigmab):    
    a,b = (mut*mut)/(sigmat*sigmat), mut/(sigmat*sigmat)
    c,d = (mub*mub)/(sigmab*sigmab), mub/(sigmab*sigmab)
    return a, b, c, d


def abcd2abcd(a, b, c, d, env=tf):    
    return a,b,c,d


# In[17]:


#from tensorflow.contrib.distributions import softplus_inverse
#softplus_inverse = lambda v: tf.log(tf.exp(v)-1.0)
softplus_inverse = tfp.math.softplus_inverse
NAME2TRANSFORMATION = {"softplus": (tf.nn.softplus, softplus_inverse), 
                       "exp": (tf.exp, np.log),
                       "pow10": (lambda v: tf.math.pow(10.0, v), lambda v: tf.math.log(v)*0.434294482)}
forward_transformation, backward_transformation = NAME2TRANSFORMATION[VARIABLES_TRANSFORMATION]

print("forward_transformation:=%s backward_transformation:=%s" % 
      (forward_transformation.__name__, backward_transformation.__name__))
#print("  transformation prec:", np.array(backward_transformation(forward_transformation(a0))-a0))

NAME2PARAMETRIZATION = {"abcd": (abcd2abcd, abcd2abcd), "musigma": (musigma2abcd, abcd2musigma)}
forward_parametrization, backward_parametrization = NAME2PARAMETRIZATION[PARAMETRIZATION]

print("forward_parametrization:=%s backward_parametrization:=%s " % 
      (forward_parametrization.__name__, backward_parametrization.__name__))
#print("  parametrization prec:", forward_parametrization(*backward_parametrization(a0,b0,c0,d0))-np.array([a0,b0,c0,d0]))
#assert (np.round(inverse_parametrization(*parametrization(a0,b0,c0,d0))-np.array([a0,b0,c0,d0]),2) ==0).all()


# # Choose expectation & variance estimators

# In[18]:


from pmf_model import create_moments_estimator, empirical_Ey_and_Ey2_tf, empirical_Ey_and_Ey2_tf_logscore
theoretical_moments = create_moments_estimator(K=K, ESTIMATOR_NO=-1, N=1, M=1)
empirical_moments   = create_moments_estimator(K=K, ESTIMATOR_NO=SAMPLING, N=1, M=1, 
    empirical_Ey_and_Ey2=empirical_Ey_and_Ey2_tf_logscore if LOG_SCORE_DERIVATIVES else empirical_Ey_and_Ey2_tf)
print("empirical_moments := %s" % empirical_moments.__name__)


# In[19]:


e, v = theoretical_moments(a0, b0, c0, d0)
print("Initialization: a=%.3f\tb=%.3f\tc=%.3f\td=%.3f\t=> E=%.4f \tV=%.4f" % (a0, b0, c0, d0, e, v))


# # Find hyperparameters matching desired values

# In[20]:


#tf.random.set_random_seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)


# In[21]:


# init (unbounded) params
float1 = lambda v: tf.Variable(float(v), dtype=tf.float64)
float2 = lambda v: tf.Variable(v, dtype=tf.float64)

a0tf, b0tf, c0tf, d0tf = float2(a0), float2(b0), float2(c0), float2(d0)
i0, i1, i2, i3 = backward_parametrization(a0tf, b0tf, c0tf, d0tf, env=tf)
#i0, i1, i2, i3 = float1(i0), float1(i1), float1(i2), float1(i3)

p0u = tf.Variable(backward_transformation(i0), name="0", dtype=tf.float64) 
p1u = tf.Variable(backward_transformation(i1), name="1", dtype=tf.float64) 
p2u = tf.Variable(backward_transformation(i2), name="2", dtype=tf.float64) 
p3u = tf.Variable(backward_transformation(i3), name="3", dtype=tf.float64) 

print("Initialization: i0=%.3f, i1=%.3f, i2=%.3f, i3=%.3f" % 
      (i0.numpy(), i1.numpy(), i2.numpy(), i3.numpy()))
print("Initialization: p0u=%.3f, p1u=%.3f, p2u=%.3f, p3u=%.3f" % 
      (p0u.numpy(), p1u.numpy(), p2u.numpy(), p3u.numpy()))


# In[22]:


trainable_variables = [var for var in [p0u, p1u, p2u, p3u] if var.name.split(":")[0] in TRAINABLE]
#optimizer = tf.train.AdamOptimizer(learning_rate=LR)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
start = time.time()
computation_time = 0.0
results = []


# In[23]:


best_loss = float("inf")
best_a, best_b, best_c, best_d = -1.0, -1.0, -1.0, -1.0
best_E, best_V = -1.0, -1.0


# In[24]:


for iteration in range(NITER):
  
    ##########################################################################################################
    # Optimization
    iteration_start_time = time.time()
        
    with tf.GradientTape() as tape:        
        p1,p2 = forward_transformation(p0u), forward_transformation(p1u) 
        p3,p4 = forward_transformation(p2u), forward_transformation(p3u) 
        
        a,b,c,d =  forward_parametrization(p1,p2,p3,p4)           
        expectation, variance = empirical_moments(a, b, c, d, NSAMPLES_LATENT, NSAMPLES_OUTPUT)        
        loss = discrepancy(expectation, variance) -(LAMBDA*variance if LAMBDA>0 else 0.0) # just to be sure
    
    grads = tape.gradient(loss, trainable_variables)              
    optimizer.apply_gradients(zip(grads, trainable_variables))#, global_step=tf.train.get_or_create_global_step())
    expectation_exact, variance_exact = theoretical_moments(a, b, c, d)
    #print(iteration,">>",a,b,c,d,expectation_exact, variance_exact)
    computation_time += (time.time()-iteration_start_time)
    
    if loss<best_loss: #TODO Can do better by evaluating here the loss with more samples
        best_loss = loss.numpy()
        best_a, best_b, best_c, best_d = a.numpy(), b.numpy(), c.numpy(), d.numpy()
        best_E, best_V = theoretical_moments(best_a, best_b, best_c, best_d)
        
    ##########################################################################################################
    # Reporting:
    failed = not is_valid(np.array(expectation))
    elapsed = time.time() - start
    if (failed) or (iteration<100) or (iteration%50==0): 
        r = (ID, D, SEED,
             a0, b0, c0, d0, NITER, LR,
             K, NSAMPLES_LATENT, NSAMPLES_OUTPUT, 
             DESIRED_EXPECTATION, DESIRED_VARIANCE, LAMBDA,
             ((NITER-1) if failed else iteration), computation_time, 
             a.numpy(), b.numpy(), c.numpy(), d.numpy(), 
             loss.numpy(), expectation_exact.numpy(), variance_exact.numpy(), expectation.numpy(), variance.numpy(),
             best_loss, best_a, best_b, best_c, best_d, best_E, best_V,
             failed)
        results.append(r)

    if (failed) or (iteration<10) or (iteration<3000 and iteration%100==0)  or (iteration%1000==0): 
        print("[%.2f][%.2f] best:\tloss:%.1f\thyperparams: a=%.5f b=%.5f c=%.5f d=%.5f => E=%.3f V=%.3f" % 
              (computation_time, elapsed, best_loss, best_a, best_b, best_c, best_d, best_E, best_V))        
        print("[%.2f][%.2f] %i/%i,\tloss:%.1f\thyperparams: a=%.5f b=%.5f c=%.5f d=%.5f" % 
              (computation_time, elapsed, iteration, NITER, loss.numpy(), 
               a.numpy(), b.numpy(), c.numpy(), d.numpy()))                
        print(" empirical: E: %.2f V: %.2f" % (expectation, variance) )    
        print(" theoretic: E: %.2f V: %.2f" % (expectation_exact, variance_exact) )
        print(" desired: E: %.2f V: %.2f" % (DESIRED_EXPECTATION, DESIRED_VARIANCE) )                
        sys.stdout.flush()
        
    if failed:
        print("Error: ran into invalid values!")
        break


    if (iteration%1000==0 or iteration>=NITER-1):
        path = ID+".csv"
        print("Saving results to: %s" % path)
        df = pd.DataFrame(results)
        CN = ["ID", "D", "SEED", "a0", "b0", "c0", "d0", "NITER", "LR", "K", "S", "SY", "E", "V", "LAMBDA",
              "iteration", "elapsed", "ct", "rt", "cb", "rb", "loss", "Et", "Vt", "Ee", "Ve", 
              "best_loss", "best_a", "best_b", "best_c", "best_d", "best_E", "best_V", "failed"]
        df.rename(columns=dict(enumerate(CN)), inplace=True)
        df.to_csv(path, header=True, index=False);

    if (DESIRED_EXPECTATION<0 or abs(best_E-DESIRED_EXPECTATION)/DESIRED_EXPECTATION<0.05) and \
       (DESIRED_VARIANCE<0 or abs(best_V-DESIRED_VARIANCE)/DESIRED_VARIANCE<0.05):
        NITER = min(NITER, iteration+1000)
    if iteration>=NITER: break




