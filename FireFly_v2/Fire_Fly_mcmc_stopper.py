########################Import Modules##########################################
import numpy as np
#from numba import jit, prange
import nest_functions_dev as nsf
from tqdm import tqdm 

import copy as duplicate
from progressbar import ProgressBar


import gelman_rubin

#####################################



def MH_acceptance(loglikelihood_new, Prior_new, loglikelihood_old, Prior_old):

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

    try:
        loglikelihood_new = float(loglikelihood_new)
 #       Prior_new = float(Prior_new)
        loglikelihood_old = float(loglikelihood_old)
#        Prior_old = float(Prior_old)
    except TypeError:
        raise TypeError("Please check input values, either integers or floats")

    # Acceptance ratio

    logR_likelihood = loglikelihood_new - loglikelihood_old

    if Prior_new == -np.inf :
        return 0.0

    if Prior_new == 1.0 and Prior_old == -np.inf:
        Prior_old = 1.0
       # pass

    if Prior_new == 1.0 and Prior_old == 1.0:
        pass

    try:

        np.seterr(all="ignore")
        R = np.exp(logR_likelihood) #* (Prior_new / Prior_old)

    except ZeroDivisionError:
        raise ZeroDivisionError("Please check division by zero")

    # Generate a uniform random Number
    u = np.random.uniform()

    # Acceptance criteria
    if R > u:

        # Accept new sample
        return 1.0

    else:
        return 0.0  # reject sample


#####################          Main mcmc     ##########################################################


#@jit
def MH_mcmc(
    loglikelihood_func, Prior_func, theta, mcmc_steps,thresh,stepsize,switch
): 
    """ Metropolis Hasting MCMC Algorithm

    Parameters

    --------------------

    loglikelihood_func : function
                        The logLikelihood function

    Prior_func         : function
                        The Prior function

    theta              : scalar or array
                        Initial sample to do mcmc from


    mcmc_steps         : scalar,int
                        The number of mcmc runs

    stepsize           : scalar
                        The jump length of a new sample

    -------------------

    Return
    ------------------

    chain_sample        : array
                         Samples from the mcmc chain

    chain_loglikelihood : array
                         Loglikelihood values of each sample

    chain_prior         : array
                        prior values of each sample

    Acceptance_ratio    : scalar
                        Acceptance ratio of the mcmc chain
    """

    naccept = 0  # count number of accepted samples
    nreject = 0  # count number of rejected samples

    # Store samples
    chain_sample = []  # np.zeros((mcmc_steps,len(theta))) EDITED
    chain_loglikelihood = np.array([])
    chain_prior = np.array([])
   

    # Do mcmc on the random survivor
    bar = ProgressBar().start()

    i = 0

    while i< mcmc_steps:

        # Generate new sample by adding a random stepsize from a normal distribution centred 0

        # Compute Likelihood and prior of old sample (theta)

        loglikelihood_old = loglikelihood_func(theta)
        Prior_old = Prior_func(theta)

        Update = switch(theta,loglikelihood_old,Prior_old,thresh,stepsize)  # added u

        
        New_theta = Update['sample_new']
        # Compute Likelihood and prior of New_theta

        loglikelihood_new = Update['loglikelihood_new'] #loglikelihood_func(New_theta)
        Prior_new =  Update['logp_new']          #Prior_func(New_theta)

        #print(New_theta)

        # Take a decision to accept of reject ne wsample (1.0 for accept and 0.0 for reject)
        Decision = MH_acceptance(
            loglikelihood_new, Prior_new, loglikelihood_old, Prior_old
        )

        if Decision == 1.0:  # if it was accepted
            # move to new sample
            theta = New_theta
           

            # Record new sample details
            chain_sample.append(New_theta)  
            chain_loglikelihood = np.append(chain_loglikelihood, loglikelihood_new)
            chain_prior = np.append(chain_prior, Prior_new)
            

            naccept += 1
            

        else:
            # Reject the new sample and stay at the old sample (theta)
            theta = theta
        
            # Record the details of the current position (old sample) sample

            chain_sample.append(theta)
            chain_loglikelihood = np.append(chain_loglikelihood, loglikelihood_old)
            chain_prior = np.append(chain_prior, Prior_old)
            nreject += 1
            

        bar.update(i)

        # if naccept > nreject:
        #     stepsize *= np.exp(1.0 / naccept)
        # if naccept < nreject:
        #     stepsize /= np.exp(1.0 / nreject)



        if (i+1) %30000 == 0:   # FOR EVERY 30000 ITERATIONS test for convergence (exclude iteration 1)
           # burn (where o cut burn-in)
            
            burn= np.where(chain_loglikelihood>=np.max(chain_loglikelihood)/2)[0]

            if burn.size == 0:
                burn = np.array([0])
            
            chain_samples = np.array(chain_sample)  #  3-D := (N,no.of sources , no. params per source) 

            
            
        
            chain0 = chain_samples.reshape(len(chain_samples),chain_samples.shape[1]*chain_samples.shape[2])[burn[0]:] #Convert to 2-D (N,no of params)

            print(chain0.shape)

            splits = np.array_split(chain0,3,axis=1)  #split the whole chain into 3
            chain1= splits[0]                            
            chain2= splits[1]                                      
            chain3 = splits[2]
            ### Test convergence with gelman rubin ###
            
            step= gelman_rubin.converge_from_list([chain1,chain2,chain3],jump=500)
            
            if step!=1 and step!=-1:
           

                break                 #chain Converged
                
            else:
                mcmc_steps += 30000    #Chain did not converge
        
            

        i += 1
    bar.finish()

        
    mcmc_steps = i
    # Calculate the Acceptance ratio

    MCMC_final_steps = mcmc_steps    #naccept / mcmc_steps
    Acceptance_ratio = naccept/(naccept+nreject)
    

    return (
        np.array(chain_sample),
        chain_loglikelihood,
        chain_prior,
        MCMC_final_steps,
        Acceptance_ratio
    ) 
