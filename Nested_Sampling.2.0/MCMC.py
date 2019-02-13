###############################################################################
# Metropolis Hasting algorithm (Markov chain Monte Carlo)                     #
#  by:  Hope Moloko                                                           #
#  Msc 2018                                                                   #
#  April 2018                                                                 #
#                                                                             #
#                                                                             #
###############################################################################

#Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stat
import corner


def metropolis_hasting(Likelihood,thetai,N,sigma_theta,Flat_prior,Flat_interval=None,Prior=None):
    
    '''
    Returns Posterior distribution of theta (parameters)
    
    Parameters
    -------------
    Likelihood : function
        The Likelihood function
      
    thetai : array_like
        initial paramater values (theta1,theta2,etc)
   
    N : int 
        sample size
    
    sigma_theta : array_like
        step size for each parameter (for proposal jump)
    
    Flat_prior : Bool
        True for a flat prior and False for non-flat prior.
    
    Flat_interval: [tuple,tuple,...], eg.([(4,6),(2,5)])
        each tuple represents the range of each parameter
    
    Prior: Function
      The prior function if prior is not flat.
    
    Returns
    --------
    out: array_like ,array_like, int
        Sample and likelihood values for each theta
        and number of accepted sample.
    '''
    
    #Initialize 
    ###########################################
    theta_candidates = np.zeros((N,len(thetai)))
    n_acc_sample = 0
    likelihood_val = np.zeros(N)
    samples = np.zeros((N,len(thetai)))
    ##########################################
    
    for i in range(N):
        theta_candidates[i,:] = thetai + np.random.normal(0,sigma_theta)
    
        if Flat_prior == True :
            add = 0
            for j in range(len(thetai)):
                if Flat_interval[j][0] <= theta_candidates[i,:][j] <= Flat_interval[j][1]:
                    add += 1
                else:
                    pass
            if add == len(thetai):
                Prior = 1
            else:
                Prior = 0

        elif Flat_prior == False:
            'I will update this section for different types of priors'
            
            pass
        
        Likelihood_proposed = Likelihood(theta_candidates[i,:])*Prior
        Likelihood_current = Likelihood(thetai)
        
        likelihood_val[i] = Likelihood_proposed
        alpha = Likelihood_proposed/Likelihood_current
    
        if alpha >= 1:
            thetai = theta_candidates[i,:]
            n_acc_sample += 1
        else:
            u = np.random.uniform()
            if u <= alpha :
                thetai = theta_candidates[i,:]
                n_acc_sample += 1
            else:
                thetai = thetai
        samples[i,:] = theta_candidates[i,:]
        
    return samples ,likelihood_val, n_acc_sample 