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

def Accept_Sample(sample,logLikelihood,prior,Storage):
    """This funtion takes in a new sample, its loglikelihood and prior and stores them

    Parameters

    --------------------
    Sample : scalar or array
            The sample to be stored
    loglikelihood: scalar
                    The loglikelihood of the new sample
    prior : scalar
            The prior of the new sample
    Storage : [chain_sample,chain_loglike,chain_prior]

    -------------------
    Output : Array,Array,Array
            chain_sample,chain_loglike,chain_prior"""

    chain_of_sample,chain_of_loglike,chain_of_prior = Storage

    #Accept new sample
    #store new sample
    chain_of_sample = np.append(chain_of_sample,sample)


    # Accept prior of new sample
    #store prior of new sample
    chain_of_prior = np.append(chain_of_prior,prior)


    #Accept likelihood of new sample
    #store logLikelihood of new sample
    chain_of_loglike = np.append(chain_of_loglike,logLikelihood)

    return chain_of_sample,chain_of_loglike,chain_of_prior


def test_Accept_Sample():  #Test functiion for the acceptance function
    #It takes in empty Storage arrays and stores the accepted sample
    list1 = np.array([])
    list2 = np.array([])
    list3 = np.array([])

    #Asuume accepted samples
    input1 = 1.0
    input2 = 2.0
    input3 = 3.0

    lists = [list1,list2,list3]
    assert Accept_Sample(input1,input2,input3,lists) == (np.array([1.0]),np.array([2.0]),np.array([3.0]))



#Metropolis Hasting algorithm for Nested Sampling


def Metropolis_Hasting(logLikelihood_func,args_loglikelihood,Prior_func,args_prior,theta,stepsize,mcmc_steps,\
                       loglikelihood_theta,prior_theta,Nested_Sampling=False,logspace=True):

    """Metropolis  Hasting Algorithm for Nested Sampling

    Parameters

    --------------
    logLikelihood_func : Function
                    The logLikelihood function

    args_loglikelihood : arguments
                      Any extra arguments for the loglikelihood function

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

    Nested_Sampling : Bool
                    If using mcmc for Nested Sampling
    logspace: Bool,
             Are the functions in logpsace or not. If in logspace then True ,else False.
             This is essential for computing the Acceptance Ratio

    loglikelihood_theta : scalar
                    logLikelihood of theta provided within nested sampling.

    prior_theta : scalar
                    prior of theta provided within nested sampling.
    -------------
    Ouput : Array,Array,Array,scalar
           chain_particles,chain_loglikelihoods,chain_prior,Acceptance_ratio """
    naccept = 0      #count number of accepted samples
    nreject = 0     #count number of rejected samples

    #Store samples
    chain_sample = np.array([])
    chain_loglike = np.array([])
    chain_prior = np.array([])


    #Storage of everything
    Storage = [chain_sample,chain_loglike,chain_prior]

    # Do mcmc on the random survivor
    for mcmci in range(mcmc_steps):

        #Generate new sample
        theta_new = theta + np.random.normal(0,stepsize)


        #Likelihood of new sample
        loglikelihood_new = logLikelihood_func(theta_new,args_loglikelihood)
        np.seterr(invalid='ignore')
        #prior prob of new sample
        prior_new = Prior_func(theta_new,args_prior)

        if Nested_Sampling == False:
            loglikelihood_theta = logLikelihood_func(theta,args_loglikelihood)
            prior_theta = Prior_func(theta,args_prior)

        elif Nested_Sampling == True:
            pass


        # mcmc Acceptance ratio
        # R = Likelihood_new*Prior_new/Likelihood_old*Prior_old
        if logspace == True: # Yes work in logspace
            # rename them in terms of log for less confusion
            #new sample
            logl_new = loglikelihood_new
            logp_new = prior_new

            #old sample
            logl_old = loglikelihood_theta
            logp_old = prior_theta

            #Compate Acceptance Ratio in logspace
            Ratio = (logl_new+logp_new)-(logl_old+logp_old)

        elif logspace == False: # Not in logspace
            #Rename for convenience
            likelihood_new = loglikelihood_new
            likelihood_old = loglikelihood_theta
            #Standard mcmc non-logspace ratio
            Ratio = (likelihood_new*prior_new)/(likelihood_old*prior_theta)


        #Accept new sample
        if (Ratio >= 1):
            chain_sample,chain_loglike,chain_prior = Accept_Sample(theta_new,loglikelihood_new,prior_new,Storage)
            theta = theta_new
             #Count number of accepted sample
            naccept += 1

        else: #Accept or reject when ratio is greater than a certain random uniform probability U(0,1)
              # This allows certain worst samples to be accepted in order to explore multi-modes

            #Generate uniform number between 0,1
            u = np.random.uniform()

            if Ratio > u:
                chain_sample,chain_loglike,chain_prior = Accept_Sample(theta_new,loglikelihood_new,prior_new,Storage)
                theta = theta_new
                #Count number of accepted samples
                naccept += 1
            else:
                #stay at current position and look for new sample again
                theta = theta

    #Acceptance Ratio
    Acceptance_ratio = naccept/mcmc_steps


    return chain_sample,chain_loglike,chain_prior,Acceptance_ratio
