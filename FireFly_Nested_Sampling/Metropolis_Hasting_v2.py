###############################################################################
#            Markov Chain Monte-Carlo - Metropolis - Hasting                  #
#                   by : Oarabile Hope Moloko                                 #   
#                          2019                                               #
###############################################################################


########################Import Modules##########################################
import numpy as np

import copy as duplicate

#####################################


def proposal_sample(theta, stepsize, change):
    """
    Randomly select source to perform change by sampling a proposal step 
    from a uniform distribution with mean = 0


     Parameters
    -------------------------
    theta : array
            Parameters (X,Y)

    stepsize           : scalar
                        The jump length of a new sample

    change : int
            Which source parameters to change (e.g 2-sources = arr[[X,Y,A],[X,Y,A]])

    ------------------------------------
    Output : New sample (array)
    """
    new = duplicate.deepcopy(theta)

    assert new.shape == theta.shape

    delta = np.random.normal(0, stepsize, size=(1, 3))
   

    new[change] = new[change] + delta
    ##################################

    return new


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
        raise TypeError("Please check input values, either integers or floats")

    # Acceptance ratio

    logR_likelihood = loglikelihood_new - loglikelihood_old

    try:

        np.seterr(all="ignore")
        R = np.exp(logR_likelihood) * (Prior_new / Prior_old)

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


def MH_mcmc(loglikelihood_func, Prior_func, theta, mcmc_steps, change, stepsize):
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

    for m in range(mcmc_steps):

        # Generate new sample by adding a random stepsize from a normal distribution centred 0
       

        New_theta = proposal_sample(theta, stepsize, change)

        # Compute Likelihood and prior of New_theta
        loglikelihood_new = loglikelihood_func(New_theta)
        Prior_new = Prior_func(New_theta)

        # Compute Likelihood and prior of old sample (theta)

        loglikelihood_old = loglikelihood_func(theta)
        Prior_old = Prior_func(theta)

        # First try no switching off
        # Take a decision to accept of reject new sample (1.0 for accept and 0.0 for reject)
        Decision = MH_acceptance(
            loglikelihood_new, Prior_new, loglikelihood_old, Prior_old
        )

        if Decision == 1.0:  # if it was accepted
            # move to new sample
            theta = New_theta

            # Record new sample details
            chain_sample.append(New_theta)  # edited
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
            

    # Calculate the Acceptance ratio

    Acceptance_ratio = naccept / mcmc_steps

    return (
        np.array(chain_sample),
        chain_loglikelihood,
        chain_prior,
        Acceptance_ratio
    )
