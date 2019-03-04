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
def MH_acceptance(loglikelihood_new,Prior_new,loglikelihood_old,Prior_old):

    """ Metropolis Hasting Acceptance Criteria

    Parameters
    -----------------

    loglikelihood_new : scalar
                        The loglikelihood value of the new sample

    Prior_new          :scalar
                        The Prior value of the new sample

    loglikelihood_old  :scalar
                        The logLikelihood value of the old(current) sample

    Prior_old          :scalar
                        The Prior value of the old(current) sampled

    -------------------
    Output :  0.0 or 1.0
            0.0 for rejected sample or 1.0 for accepted sample"""


    #Acceptance ratio

    logR = (loglikelihood_new - loglikelihood_old)

    R = np.exp(logR)*(Prior_new/Prior_old)

    #Generate a uniform random Number
    u = np.random.uniform()

    if (R>= 1): #Acceptance criteria

        #Accept new sample

        return 1.0

    elif (R > u):

        #Accept new sample
        return 1.0

    return 0.0


# Unit tests for acceptance function
#####################################################################################################
def test_accept_good_sample_mh_accetance():
    #inputs
    loglikelihood_new = -6.5
    Prior_new         = 0.5

    loglikelihood_old = -8.5
    Prior_old         = 0.5

    # these values will yeild R = 7.38 which is greater than 1, and therefore the sample must be accepted

    assert (MH_acceptance(loglikelihood_new,Prior_new,loglikelihood_old,Prior_old) == 1.0)

def test1_reject_worst_sample_mh_accetance():
    #inputs
    loglikelihood_new = -9.5
    Prior_new         = np.nan  #out of the uniform bounds , the sample must be rejected

    loglikelihood_old = -6.5
    Prior_old         = 0.5

    # these values will yield R = nan, therefore the sample must be rejected

    assert (MH_acceptance(loglikelihood_new,Prior_new,loglikelihood_old,Prior_old) == 0.0)

def test2_reject_worst_sample_mh_accetance():
    #inputs
    np.random.seed(1) #will yeild u = 0.41
    loglikelihood_new = -9.5
    Prior_new         = 0.5  #in the bounds but its the worst sample

    loglikelihood_old = -6.5
    Prior_old         = 0.5

    #these values will yield R = 0.049 , so they are smaller than 1 and u, therefore the sample must be rejected

    assert (MH_acceptance(loglikelihood_new,Prior_new,loglikelihood_old,Prior_old) == 0.0)

def test_accept_worse_sample_mh_acceptance(): #sometimes
    #inputs
    #we set random.seed to known seed so that we know the value of u in the MH_acceptance functions
    np.random.seed(1)  #will have u = 0.41

    loglikelihood_new = -7.5
    Prior_new         = 0.5

    loglikelihood_old = -7.07
    Prior_old         = 0.5

    #these values should yield R = 0.65 which is greater than u but smaller than 1 and thus we can accept the worse sample at random

    assert (MH_acceptance(loglikelihood_new,Prior_new,loglikelihood_old,Prior_old) == 1.0)

def test_literal_types_int_mh_acceptance():
    #inputs (logLikelihood ints not floats)
    np.random.seed(1) #will yeild u = 0.41
    loglikelihood_new = -9
    Prior_new         = 0.5  #in the bounds but its the worst sample

    loglikelihood_old = -6
    Prior_old         = 0.5

    #these values will yield R = 0.049 , so they are smaller than 1 and u, therefore the sample must be rejected

    assert (MH_acceptance(loglikelihood_new,Prior_new,loglikelihood_old,Prior_old) == 0.0)
#########################################################################################################


#####################          Main mcmc     ##########################################################

def MH_mcmc(loglikelihood_func,Prior_func,theta,args_loglike,args_prior,mcmc_steps,stepsize,Nested_Sampling=False,Nested_Sampling_args=None):
    """ Metropolis Hasting MCMC Algorithm

    Parameters

    --------------------

    loglikelihood_func : function
                        The logLikelihood function

    Prior_func         : function
                        The Prior function

    theta              : scalar or array
                        Initial sample to do mcmc from

    args_loglike       :array or list
                        extra arguments for the loglikelihood function

    args_prior       :array or list
                        extra arguments for the Prior function

    mcmc_steps         : scalar,int
                        The number of mcmc runs

    stepsize           : scalar
                        The jump length of a new sample

    Nested_Sampling   : bool,default=False
                       If running mcmc for nested sampling.

    Nested_Sampling_args : array,default=None
                          if Nested_Sampling = True, then set arguments
    -------------------

    Output :  Array, Array , Array , scalar
            chain_sample , chain_loglikelihood ,chain_prior and the Acceptance_ratio
    """
    naccept = 0      #count number of accepted samples
    nreject = 0     #count number of rejected samples

    #Store samples
    chain_sample = np.array([])
    chain_loglikelihood = np.array([])
    chain_prior = np.array([])

    #Initialize
    loglikelihood_old = 0.0
    Prior_old         =0.0

    # Do mcmc on the random survivor
    for mcmci in range(mcmc_steps):

        #Generate new sample by adding a random stepsize from a normal distribution centred at 0

        New_theta = theta + np.random.normal(0,stepsize)

        #Compute Likelihood and prior of New_theta

        loglikelihood_new = loglikelihood_func(New_theta,args_loglike)
        Prior_new         = Prior_func(New_theta,args_prior)

        if Nested_Sampling == True:
            #If running mcmc fro ns, then take the logLikelihood and prior thresholds from ns
            loglstar,logpstar = Nested_Sampling_args
            loglikelihood_old = loglstar
            Prior_old         = logpstar

        #Compute Likelihood and prior of old sample (theta)
        elif Nested_Sampling == False:
            loglikelihood_old = loglikelihood_func(theta,args_loglike)
            Prior_old         = Prior_func(theta,args_prior)


        #Take a decision to accept of reject ne wsample (1.0 for accept and 0.0 for reject)
        Decision = MH_acceptance(loglikelihood_new,Prior_new,loglikelihood_old,Prior_old)

        if Decision == 1.0 :  #if it was accepted
            #move to new sample
            theta = New_theta

            #Record new sample details
            chain_sample        = np.append(chain_sample,New_theta)
            chain_loglikelihood = np.append(chain_loglikelihood,loglikelihood_new)
            chain_prior         = np.append(chain_prior,Prior_new)

            naccept += 1

        else:
            #Reject the new sample and stay at the old sample (theta)
            theta = theta

            #Record the details of the current position (old sample) sample

            chain_sample        = np.append(chain_sample,theta)
            chain_loglikelihood = np.append(chain_loglikelihood,loglikelihood_old)
            chain_prior         = np.append(chain_prior,Prior_old)

    #Calcu;ate the Acceptance ratio

    Acceptance_ratio = naccept/mcmc_steps

    return chain_sample , chain_loglikelihood ,chain_prior ,Acceptance_ratio
