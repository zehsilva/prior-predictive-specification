"""Methods calculating E[Y] and E[Y^2] (and therefore also Var[Y]) over prior predictive distribution for Hierarchical Poisson Matrix Factorization."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import warnings


def _ttf(v):
    return tf.Variable(v, dtype=tf.float64) if type(v) is float else v


#a=0.3, ap=0.3, bp=1.0, c=0.3, cp=0.3, dp=1.0, 
def empirical_Ey_and_Ey2_np_fast(a=3, ap=3, bp=1.0, c=3, cp=3, dp=1.0, 
                            nsamples_latent=100, nsamples_latent1=1, 
                            nsamples_output=10, K=25, N=1, M=1):        
    """
        Returns E_prior[Y] and E_prior[Y^2] for given set of hyperparameters.
        Parametrization like in: http://jakehofman.com/inprint/poisson_recs.pdf
    """        
    if N!=1: warnings.warn("N!=1 will be ignored!")
    if N!=1: warnings.warn("M!=1 will be ignored!")
          
    N = nsamples_latent*nsamples_latent1*nsamples_output
    ksi = np.random.gamma(ap, bp/ap, size=N)
    theta = np.random.gamma(a, 1.0/ksi, size=(K, N)) 
    eta = np.random.gamma(cp, dp/cp, size=N)
    beta =  np.random.gamma(c, 1.0/eta, size=(K, N))
    latent = (theta*beta).sum(0)
    ys = np.random.poisson(latent)
    expectation = np.mean(ys)
    expectation_squared = np.mean(np.array(ys)**2)
    return expectation, expectation_squared


#a=0.3, ap=0.3, bp=1.0, c=0.3, cp=0.3, dp=1.0, 
def empirical_Ey_and_Ey2_np(a=3, ap=3, bp=1.0, c=3, cp=3, dp=1.0, 
                            nsamples_latent=100, nsamples_latent1=1, 
                            nsamples_output=10, K=25, N=1, M=1):        
    """
        Returns E_prior[Y] and E_prior[Y^2] for given set of hyperparameters.
        Parametrization like in: http://jakehofman.com/inprint/poisson_recs.pdf
        Naive numpy implementation for testing purposes.
    """             
    if N!=1: warnings.warn("N!=1 will be ignored!")
    if N!=1: warnings.warn("M!=1 will be ignored!")
     
    ys = []
    for _ in range(nsamples_latent*nsamples_latent1*nsamples_output): 
      ksi = np.random.gamma(ap, bp/ap)
      theta = np.random.gamma(a, 1.0/ksi, size=K) 

      eta = np.random.gamma(cp, dp/cp)
      beta =  np.random.gamma(c, 1.0/eta, size=K)

      latent = (theta*beta).sum()
      if latent>1e16: 
          warnings.warn("Skipping too large latent value: %s" % latent)
          continue
      y = np.random.poisson(latent)
      ys.append(y)
    expectation = np.mean(ys)
    expectation_squared = np.mean(np.array(ys)**2)
    return expectation, expectation_squared


@tf.function
def empirical_Ey_and_Ey2_tf_logscore(a=3, ap=3, bp=1.0, c=3, cp=3, dp=1.0,  
                            nsamples_latent=100, nsamples_latent1=1, 
                            nsamples_output=10, K=25, N=1, M=1):        
    """
        Returns E_prior[Y] and E_prior[Y^2] for given set of hyperparameters.
        Parametrization like in: http://jakehofman.com/inprint/poisson_recs.pdf
        Gradients obtained with log-score derivative trick.
    """     
    if N!=1: warnings.warn("N!=1 will be ignored!")
    if N!=1: warnings.warn("M!=1 will be ignored!")
    
    ksi = tfd.Gamma(ap, ap/bp).sample(nsamples_latent) # NL0
    theta = tfd.Gamma(a, ksi).sample((K,nsamples_latent1)) # K x NL1 x NL0
    
    eta = tfd.Gamma(cp, cp/dp).sample(nsamples_latent)
    beta =  tfd.Gamma(c, eta).sample((K,nsamples_latent1))
    
    latent = tf.reduce_sum(theta*beta, 0) # NL1 x NL0
    latent = tf.reshape(latent, [-1]) # NL1*NL0
      
    poisson = tfd.Poisson(rate=latent)
    y_samples = poisson.sample([nsamples_output])

    conditional_expectation = tfp.monte_carlo.expectation(
        f=lambda x: x,
        samples=y_samples,
        log_prob=poisson.log_prob, use_reparameterization = False)

    conditional_expectation_squared = tfp.monte_carlo.expectation(
        f=lambda x: x*x,
        samples=y_samples,
        log_prob=poisson.log_prob, use_reparameterization = False)
    
    expectation = tf.reduce_mean(conditional_expectation)
    expectation_squared = tf.reduce_mean(conditional_expectation_squared)
    
    return expectation, expectation_squared


@tf.function
def empirical_Ey_and_Ey2_tf(a=3, ap=3, bp=1.0, c=3, cp=3, dp=1.0,  
                            nsamples_latent=100, nsamples_latent1=1, 
                            nsamples_output=10, K=25, N=1, M=1):        
    """
        Returns E_prior[Y] and E_prior[Y^2] for given set of hyperparameters.
        Parametrization like in: http://jakehofman.com/inprint/poisson_recs.pdf
    """     
    if N!=1: warnings.warn("N!=1 will be ignored!")
    if N!=1: warnings.warn("M!=1 will be ignored!")
             
    #a, ap, bp, c, cp, dp = _ttf(a), _ttf(ap), _ttf(bp), _ttf(c), _ttf(cp), _ttf(dp) # cast to tf
    
    ksi = tfd.Gamma(ap, ap/bp).sample(nsamples_latent) # NL0
    theta = tfd.Gamma(a, ksi).sample((K,nsamples_latent1)) # K x NL1 x NL0
    
    eta = tfd.Gamma(cp, cp/dp).sample(nsamples_latent)
    beta =  tfd.Gamma(c, eta).sample((K,nsamples_latent1))
    
    latent = tf.reduce_sum(theta*beta, 0) # NL1 x NL0
    latent = tf.reshape(latent, [-1]) # NL1*NL0
      
    poisson = tfd.Poisson(rate=latent)
    #y_samples = np.random.poisson(latent, size=[nsamples_output, nsamples_latent*nsamples_latent1]) # NO x NL1*NL0
    y_samples = tf.stop_gradient( poisson.sample([nsamples_output]) )
    
    y_probs = tf.exp( poisson.log_prob(y_samples) )
    #y_probs1 = np.array([[tf.exp(tfd.Poisson(rate=latent[i]).log_prob(y_samples[j,i])).numpy() 
    #                                                        for j in range(nsamples_output)] 
    #                                                        for i in range(nsamples_latent * nsamples_latent1)]).T
    #assert (y_probs - y_probs1).numpy().max()<1e-12
    total_prob = tf.reduce_sum(y_probs, 0)
    conditional_expectation = tf.reduce_sum(y_probs * y_samples, 0) / total_prob
    conditional_expectation_squared = tf.reduce_sum(y_probs * (y_samples**2), 0) / total_prob
    
    expectation = tf.reduce_mean(conditional_expectation)
    expectation_squared = tf.reduce_mean(conditional_expectation_squared)
    
    return expectation, expectation_squared


def create_moments_estimator(K=25, ESTIMATOR_NO=-1, N=1, M=1, 
                             empirical_Ey_and_Ey2 = empirical_Ey_and_Ey2_tf_logscore):
    

    def theoretical_moments(a=3, ap=3, bp=1.0, c=3, cp=3, dp=1.0, nsamples=1000000):
        """ No theoretical derivation for HPF available; 
            using numpy approximation instead.
        """
        ey,ey2 = empirical_Ey_and_Ey2_np_fast(a, ap, bp, c, cp, dp, 
                nsamples_latent=nsamples, nsamples_latent1=1, nsamples_output=1, K=K)
        e, v = ey, ey2-(ey**2)
        return e, v
      

    def empirical_moments_EV(a, ap, bp, c, cp, dp, 
                             nsamples_latent, nsamples_output, 
                             N=N, M=M, K=K): 
        """ Returns empirical estimates of expectation & variance.""" 
        expectation, expectation_squared = empirical_Ey_and_Ey2(a, ap, bp, c, cp, dp, 
                                                    nsamples_latent=nsamples_latent, 
                                                    nsamples_output=nsamples_output, 
                                                    N=N, M=M, K=K)
        variance = expectation_squared - expectation**2        
        expectation, variance = tf.reduce_mean(expectation), tf.reduce_mean(variance) 
        return expectation, variance


    def empirical_moments_EV_decoupled(a, ap, bp, c, cp, dp, 
                                       nsamples_latent, nsamples_output, 
                                       N=N, M=M, K=K):  
        """ Returns empirical estimates of expectation & variance.
            Decoupled E & V but E[y] and E[y^2] for V computed together.
        """
        expectation, _ = empirical_moments_EV(a, ap, bp, c, cp, dp, 
                                              nsamples_latent=nsamples_latent, 
                                              nsamples_output=nsamples_output, 
                                              N=N, M=M, K=K)
        _, variance = empirical_moments_EV(a, ap, bp, c, cp, dp, 
                                           nsamples_latent=nsamples_latent, 
                                           nsamples_output=nsamples_output, 
                                           N=N, M=M, K=K)
        return expectation, variance
      
        
    def empirical_moments_V_decoupled(a, ap, bp, c, cp, dp, 
                                      nsamples_latent, nsamples_output, 
                                      N=N, M=M, K=K):  
        """ Returns empirical estimates of expectation & variance.
            Uses the same E[y] for both E and V; E[y^2] independent of E[y]. 
        """
        expectation, _ = empirical_Ey_and_Ey2(a, ap, bp, c, cp, dp, 
                                              nsamples_latent=nsamples_latent, 
                                              nsamples_output=nsamples_output, 
                                              N=N, M=M, K=K)
        _, expectation_squared = empirical_Ey_and_Ey2(a, ap, bp, c, cp, dp, 
                                                      nsamples_latent=nsamples_latent, 
                                                      nsamples_output=nsamples_output, 
                                                      N=N, M=M, K=K)    
        variance = expectation_squared - expectation**2        
        expectation, variance = tf.reduce_mean(expectation), tf.reduce_mean(variance)    
        
        return expectation, variance    


    def empirical_moments_fully_decoupled(a, ap, bp, c, cp, dp, 
                                          nsamples_latent, nsamples_output, 
                                          N=N, M=M, K=K):  
        """ Returns empirical estimates of expectation & variance.
            Computes E[y] for E independent of E[y] for V. E[y^2] independent of E[y]. 
        """        
        expectation, _ = empirical_Ey_and_Ey2(a, ap, bp, c, cp, dp, 
                                              nsamples_latent=nsamples_latent, 
                                              nsamples_output=nsamples_output, 
                                              N=N, M=M, K=K)
        _, expectation_squared = empirical_Ey_and_Ey2(a, ap, bp, c, cp, dp, 
                                                      nsamples_latent=nsamples_latent, 
                                                      nsamples_output=nsamples_output, 
                                                      N=N, M=M, K=K)    
        variance = expectation_squared - expectation**2        
        variance = tf.reduce_mean(variance) # empirical
        
        expectation, _ = empirical_Ey_and_Ey2(a, ap, bp, c, cp, dp, 
                                              nsamples_latent=nsamples_latent, 
                                              nsamples_output=nsamples_output, 
                                              N=N, M=M, K=K)
        expectation = tf.reduce_mean(expectation)
        
        return expectation, variance   

    NO2ESTIMATOR = {-1: theoretical_moments,
                     0: empirical_moments_fully_decoupled, 
                     1: empirical_moments_EV_decoupled, 
                     2: empirical_moments_EV, 
                     3: empirical_moments_V_decoupled}
    if ESTIMATOR_NO not in NO2ESTIMATOR: raise Exception("Wrong estimator number! Try: %s" % NO2ESTIMATOR)
    estimator = NO2ESTIMATOR[ESTIMATOR_NO]
    return estimator





