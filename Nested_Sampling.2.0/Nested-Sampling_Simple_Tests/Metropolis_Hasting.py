########################Import Modules##########################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import corner
import copy as duplicate
import time
from matplotlib import colors
import scipy as sp

from ipywidgets import IntProgress
from IPython.display import display

from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

#####################################


#Metropolis Hasting algorithm for Nested Sampling


def Metropolis_Hasting_NS(Likelihood_func,args_likelihood,Prior_func,args_prior,theta,stepsize,mcmc_steps,\
                       likelihood_theta,prior_theta,logspace=True):
    
    """Metropolis  Hasting Algorithm for Nested Sampling
    
    Parameters
    
    --------------
    Likelihood_func : Function
                    The Likelihood function
    
    args_likelihood : arguments
                      Any extra arguments for the likelihood function
                    
    Prior_func      :Funcion
                    The prior function
                    
    args_prior : arguments
                 Any extra arguments for the prior function
    
    theta : array or list
            initial parameters to do mcmc from.
    
    stepsize : array or scalar
                step length of each mcmc step
    
    mcmc_steps: scalar
                Number of mcmc runs
                
    logspace: Bool,
             Are the functions in logpsace or not. If in logspace then True ,else False.
             This is essential for computing the Acceptance Ratio
             
    Likelihood_theta : scalar
                    Likelihood of theta provided within nested sampling.
                    
    prior_theta : scalar
                    prior of theta provided within nested sampling.
    -------------
    Ouput : Array or list,scalar,scalar,scalar
           new_particle,new_particle_prior,new_particle_oolikelihood,Acceptance_ratio """
    naccept = 0      #count number of accepted samples
    nreject = 0     #count number of rejected samples
    # Do mcmc on the random survivor
    for mcmci in range(mcmc_steps):  
        
        #Generate new sample
        theta_new = theta + stepsize * np.random.randn()
       
        
        #Likelihood of new sample
        likelihood_new = Likelihood_func(theta_new,*args_likelihood)                                
        np.seterr(invalid='ignore')
        #prior prob of new sample
        prior_new = Prior_func(theta_new,*args_prior) 
        
        
        # mcmc Acceptance ratio
        # R = Likelihood_new*Prior_new/Likelihood_old*Prior_old
        if logspace == True: # Yes work in logspace
            # rename them in terms of log for less confusion
            #new sample
            logl_new = likelihood_new
            logp_new = prior_new
            
            #old sample
            logl_old = likelihood_theta
            logp_old = prior_theta
            
            #Compate Acceptance Ratio in logspace
            Ratio = (logl_new+logp_new)-(logl_old+logp_old)
            
        elif logspace == False: # Not in logspace
           
            #Standard mcmc non-logspace ratio
            Ratio = (likelihood_new*prior_new)/(likelihood_theta*prior_theta)
        
        
        #Accept new sample
        if (Ratio >= 1):
            
            #To avoid unboundlocalerror
            #Trying to execute the next lines without global, 
            #python tries to read the the values of the variables before their assigned
            #resulting in an unboundlocal error
            
            global new_sample
            global new_sample_likelihood
            global new_sample_prior
            #Accept new sample 
            new_sample = theta_new

            
            # Accept prior of new sample
            new_sample_prior = prior_new
            
            
            #Accept likelihood of new sample
            new_sample_likelihood = likelihood_new                                      
            
             #Count number of accepted sample 
            naccept += 1                                             
            
        else: #Accept or reject when ratio is greater than a certain random uniform probability U(0,1)
              # This allows certain worst samples to be accepted in order to explore multi-modes
            
            #Generate uniform number between 0,1
            u = np.random.uniform()
            
            if Ratio > u:
                
                #Accept new sample 
                new_sample = theta_new
                
                # Accept prior of new sample
                new_sample_prior = prior_new
            
                #Accept likelihood of new sample
                new_sample_likelihood = likelihood_new                                     
            
                #Count number of accepted samples
                naccept += 1
                
            
            else:
                #stay at current position and look for new sample again
                theta = theta
    
    #Acceptance Ratio
    Acceptance_ratio = naccept/mcmc_steps
    
    #Just Renaming the outputs
    new_particle = new_sample
    new_like = new_sample_likelihood
    new_prior = new_sample_prior
    
    
    return new_particle,new_prior,new_like,Acceptance_ratio