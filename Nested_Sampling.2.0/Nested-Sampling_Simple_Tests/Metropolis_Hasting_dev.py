########################Import Modules##########################################
import numpy as np
from numba import jit , prange 

import copy as duplicate
#####################################

def switcher(theta,prior_transform,proposal_sample,stepsize,stepper,kwargs):
    """
    switch a source off or on or just do a random walk
    
    
    return : new sample
    """
    random_source = np.random.randint(len(theta))   #generate random number to choose a random source
    selected_source = theta[random_source]
    
    
    flux_old = selected_source[0]                   #take the flux of the selected source
    position = selected_source[1:]
    
    random_state = np.random.uniform()              #draw a random number between (0,1)

    if random_state >= 0.5:                         #50% chance of switching on or off
        state_new = 1.0     #On state

    else:
        state_new = 0.0     #Off state

    # Conditions 

    if flux_old == 0.0 and state_new == 0.0: #from off to off
        #random position and Flux_new = 0
        flux_new = 0.0
        position = prior_transform(**kwargs)[0][1:]  #directly sample a random position from the prior.
        

    if flux_old != 0.0 and state_new == 0.0:  #from on to off
        #random position and flux_new = .0
        flux_new = 0.0
        position =  prior_transform(**kwargs)[0][1:]  #directly sample a random position from the prior.
        
        
    if flux_old == 0.0 and state_new == 1.0:
        #random position and sample flux_new directly from prior.
        flux_new = prior_transform(**kwargs)[0][0] 
        position = prior_transform(**kwargs)[0][1:] 
        

    if flux_old !=0.0 and state_new == 1.0:
        #Continue with random walk in position and flux.
        theta = proposal_sample(theta,stepsize,stepper)
        
        return theta
    
    selected_source[0] = flux_new
    selected_source[1:]    = position
    
    theta[random_source] = selected_source
    
    return theta


@jit
def proposal_sample(theta,stepsize,stepper):   
    """
    Randomly select source to perform change by sampling a proposal step 
    from a uniform distribution with mean = 0
    """
    new = duplicate.deepcopy(theta)


    assert new.shape == theta.shape
    
    ##################################
    NUM = list(range(0,len(theta)))       #generate numbers between 0 and number of sources

    changers = []

    for i in range(stepper):             #choose how many sources to change
        indx = np.random.choice(NUM)     # randomly choose source to chnage
        changers.append(indx)
        NUM.remove(indx)                 #remove the previously chosen source

    delta = np.random.normal(0,stepsize,size=(len(changers),3))
    
    new[changers] = new[changers] + delta
    ##################################
        
    return new 

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
    
    try:
        loglikelihood_new = float(loglikelihood_new)
        Prior_new = float(Prior_new)
        loglikelihood_old = float(loglikelihood_old)
        Prior_old        = float(Prior_old)
    except TypeError:
        raise TypeError('Please check input values, either integers or floats')


    #Acceptance ratio
    
    logR_likelihood = (loglikelihood_new - loglikelihood_old )

    try:
        
        np.seterr(all='ignore')
        R = np.exp(logR_likelihood)*(Prior_new/Prior_old)
    
        

    except ZeroDivisionError:
        raise ZeroDivisionError('Please check division by zero')

    #Generate a uniform random Number
    u = np.random.uniform()

    #Acceptance criteria
    if (R > u):

        #Accept new sample
        return 1.0

    else: return 0.0 #reject sample




#####################          Main mcmc     ##########################################################


def MH_mcmc(loglikelihood_func,Prior_func,theta,mcmc_steps,stepper,stepsize,prior_transform,kwargs):  
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

    naccept = 0      #count number of accepted samples
    nreject = 0     #count number of rejected samples

    #Store samples
    chain_sample = []     #np.zeros((mcmc_steps,len(theta))) EDITED
    chain_loglikelihood = np.array([])
    chain_prior = np.array([])
    

    # Do mcmc on the random survivor
    
    for m in range(mcmc_steps):
        #Generate new sample by adding a random stepsize from a normal distribution centred 0
        

        New_theta  = switcher(theta,prior_transform,proposal_sample,stepsize,stepper,kwargs)

        #Compute Likelihood and prior of New_theta
        loglikelihood_new = loglikelihood_func(New_theta)
        Prior_new         = Prior_func(New_theta)

        #Compute Likelihood and prior of old sample (theta)

        loglikelihood_old = loglikelihood_func(theta)
        Prior_old         = Prior_func(theta)



        #First try no switching off
        #Take a decision to accept of reject new sample (1.0 for accept and 0.0 for reject)
        Decision = MH_acceptance(loglikelihood_new,Prior_new,loglikelihood_old,Prior_old)

            

        if Decision == 1.0 :  #if it was accepted
            #move to new sample
            theta = New_theta
            
            

            #Record new sample details
            chain_sample.append(New_theta)    # edited
            chain_loglikelihood = np.append(chain_loglikelihood,loglikelihood_new)
            chain_prior         = np.append(chain_prior,Prior_new)
            

            naccept += 1

        else:
            
            #Reject the new sample and stay at the old sample (theta)
            theta = theta


            #Record the details of the current position (old sample) sample

            chain_sample.append(theta)
            chain_loglikelihood = np.append(chain_loglikelihood,loglikelihood_old)
            chain_prior         = np.append(chain_prior,Prior_old)
            
            nreject += 1

       
    #Calculate the Acceptance ratio

    Acceptance_ratio = naccept/mcmc_steps

    

    return np.array(chain_sample) , chain_loglikelihood ,chain_prior ,Acceptance_ratio ,naccept,  nreject 
