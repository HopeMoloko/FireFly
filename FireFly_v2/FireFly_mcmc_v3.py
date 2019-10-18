# uncompyle6 version 3.5.0
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.2 (default, Dec 29 2018, 06:19:36) 
# [GCC 7.3.0]
# Embedded file name: /home/hope-moloko/FireFly/FireFly/Fire_Fly_mcmc_stopper.py
# Size of source mod 2**32: 8998 bytes
import numpy as np
import nest_functions_dev as nsf
from tqdm import tqdm
import copy as duplicate
from progressbar import ProgressBar
import gelman_rubin
import os
import psutil
import pandas as pd

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
        loglikelihood_old = float(loglikelihood_old)
    except TypeError:
        raise TypeError('Please check input values, either integers or floats')

    logR_likelihood = loglikelihood_new - loglikelihood_old
   
    try:
        np.seterr(all='ignore')
        R = np.exp(logR_likelihood)
    except ZeroDivisionError:
        raise ZeroDivisionError('Please check division by zero')

    u = np.random.uniform()
    
    
    # Acceptance criteria
    if R > u:

        # Accept new sample
        return 1.0

    else:
        return 0.0  # reject sample


def MH_mcmc(loglikelihood_func, Prior_func, theta, mcmc_steps, thresh, stepsize, switch, sample_file_name=None, checkpoint=True):
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
    
    bar = ProgressBar().start()
    i = 0

    if checkpoint == True:
        if os.path.exists('restore/' + sample_file_name + '-samples-restore.txt'):  #check if restore file exists
            chain_sample_restored = np.loadtxt('restore/' + sample_file_name + '-samples-restore.txt',
                                     delimiter=',') # load saved samples from unfinished run
            chain_logliklihood_restored = np.loadtxt('restore/' + sample_file_name + '-loglikelihood-restore.txt'
                                    , delimiter=',') # load saved loglikelihood values for each sample from unfinished run

            stats = pd.read_csv('restore/'+sample_file_name+'-stats.csv')
            
            chain_sample_new = chain_sample_restored.reshape(chain_sample_restored.shape[0], 
                                int(chain_sample_restored.shape[1] / 3), 3)  # Reshape to (N,sources,parameters)
            
            chain_sample = chain_sample_new
            chain_loglikelihood = chain_logliklihood_restored
            chain_prior = np.array([])
            theta = chain_sample[(-1)]   #initialize mcmc with the last saved samples
            naccept = stats['naccept']

        else: #if restoring file doesn't exist, continue with normal mcmc
            os.makedirs('restore')
            chain_sample = np.empty((0, theta.shape[0], theta.shape[1]))  #shape to (N,sources,parameters)
            chain_loglikelihood = np.array([])
            chain_prior = np.array([])
    else:
        chain_sample = np.empty((0, theta.shape[0], theta.shape[1])) #shape to (N,sources,parameters)
        chain_loglikelihood = np.array([])
        chain_prior = np.array([])
    
    while i < mcmc_steps:  #Run chain until a stopping condition

        # Generate new sample by adding a random stepsize from a normal distribution centred 0

        # Compute Likelihood and prior of old sample (theta)
        
        loglikelihood_old = loglikelihood_func(theta)
        Prior_old = Prior_func(theta)

        #Generate a new sample from FireFly
        Update = switch(theta, loglikelihood_old, Prior_old, thresh, stepsize)
        New_theta = Update['sample_new']
        loglikelihood_new = Update['loglikelihood_new']
        Prior_new = Update['logp_new']

        # Take a decision to accept of reject ne wsample (1.0 for accept and 0.0 for reject)
        Decision = MH_acceptance(loglikelihood_new, Prior_new, loglikelihood_old, Prior_old)

        if Decision == 1.0: # if it was accepted

            # move to new sample
            theta = New_theta

            # Record new sample details
            chain_sample = np.append(chain_sample, [New_theta], axis=0)
            chain_loglikelihood = np.append(chain_loglikelihood, loglikelihood_new)
            chain_prior = np.append(chain_prior, Prior_new)
            naccept += 1

        else:  # Reject the new sample and stay at the old sample (theta)
            theta = theta

            # Record the details of the current position (old sample) sample
            chain_sample = np.append(chain_sample, [New_theta], axis=0)
            chain_loglikelihood = np.append(chain_loglikelihood, loglikelihood_old)
            chain_prior = np.append(chain_prior, Prior_old)
            nreject += 1
        bar.update(i)

        #in the case of load-shedding (save current chain and stop code)

        battery = psutil.sensors_battery()
        percent = battery.percent
        if percent < 10.0: #if battery is less than 10 %
            np.savetxt('restore/' + sample_file_name + '-samples-restore.txt', chain0, delimiter=',')
            np.savetxt('restore/' + sample_file_name + '-loglikelihood-restore.txt', chain_loglikelihood, delimiter=',')
            np.savetxt('restore/' + sample_file_name + '-stat.txt',naccept)
            stat = pd.DataFrame.from_records([{ 'naccapt': naccept}])
            stat.to_csv('restore/' + sample_file_name + '-stats.txt')
            break

        if (i + 1) % 10000 == 0:  # Check gelman_rubin ( Convergence Condition)
            
            burn = np.where(chain_loglikelihood >= np.max(chain_loglikelihood) / 2)[0]
            
            if burn.size == 0:
                burn = np.array([0])
            
            chain0 = chain_sample.reshape(len(chain_sample), chain_sample.shape[1] * chain_sample.shape[2])[burn[0]:]

            #checkpoint (Save current samples)
            np.savetxt('restore/' + sample_file_name + '-samples-restore.txt', chain0, delimiter=',')
            np.savetxt('restore/' + sample_file_name + '-loglikelihood-restore.txt', chain_loglikelihood, delimiter=',')
            stat = pd.DataFrame.from_records([{ 'naccept': naccept}])
            stat.to_csv('restore/' + sample_file_name + '-stats.csv')
        
            splits = np.array_split(chain0, 3, axis=1)  #split chain samples into 3 parts for gelman_rubin
            chain1 = splits[0]
            chain2 = splits[1]
            chain3 = splits[2]
            step = gelman_rubin.converge_from_list([chain1, chain2, chain3], jump=500)  #Gelman rubin condition
            if step != 1:
                if step != -1:
                    break
                else:
                    mcmc_steps += 30000
                    
            # Refine step-size to let acceptance ratio converge around 50%
            if naccept > nreject:
                stepsize *= np.exp(1.0 / naccept)
            if naccept < nreject:
                stepsize /= np.exp(1.0 / nreject)
        i += 1

    bar.finish()

    MCMC_final_steps = len(chain_sample)
    Acceptance_ratio = naccept / MCMC_final_steps
    
    return (
     np.array(chain_sample),
     chain_loglikelihood,
     chain_prior,
     MCMC_final_steps,
     Acceptance_ratio)

