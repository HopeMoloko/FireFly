import Metropolis_Hasting_3_0T as MCMC
import numpy as np
import pytest
import Functions as F

# Unit tests for acceptance function
#####################################################################################################
def test_accept_good_sample_mh_accetance():
    #inputs
    loglikelihood_new = -6.5
    Prior_new         = 0.5

    loglikelihood_old = -8.5
    Prior_old         = 0.5

    # these values will yeild R = 7.38 which is greater than 1, and therefore the sample must be accepted

    assert (MCMC.MH_acceptance(loglikelihood_new,Prior_new,loglikelihood_old,Prior_old) == 1.0)

def test1_reject_worst_sample_mh_accetance():
    #inputs
    loglikelihood_new = -9.5
    Prior_new         = -np.inf  #out of the uniform bounds , the sample must be rejected

    loglikelihood_old = -6.5
    Prior_old         = 0.5

    # these values will yield R = nan, therefore the sample must be rejected

    assert (MCMC.MH_acceptance(loglikelihood_new,Prior_new,loglikelihood_old,Prior_old) == 0.0)

def test2_reject_worst_sample_mh_accetance():
    #inputs
    np.random.seed(1) #will yeild u = 0.41
    loglikelihood_new = -9.5
    Prior_new         = 0.5  #in the bounds but its the worst sample

    loglikelihood_old = -6.5
    Prior_old         = 0.5

    #these values will yield R = 0.049 , so they are smaller than 1 and u, therefore the sample must be rejected

    assert (MCMC.MH_acceptance(loglikelihood_new,Prior_new,loglikelihood_old,Prior_old) == 0.0)


def test_accept_worse_sample_mh_acceptance(): #sometimes
    #inputs
    #we set random.seed to known seed so that we know the value of u in the MH_acceptance functions
    np.random.seed(1)  #will have u = 0.41

    loglikelihood_new = -7.5
    Prior_new         = 0.5

    loglikelihood_old = -7.07
    Prior_old         = 0.5

    #these values should yield R = 0.65 which is greater than u but smaller than 1 and thus we can accept the worse sample at random

    assert (MCMC.MH_acceptance(loglikelihood_new,Prior_new,loglikelihood_old,Prior_old) == 1.0)

def test_input_types_mh_acceptance():
    #inputs (logLikelihood ints not floats)
    np.random.seed(1) #will yeild u = 0.41
    loglikelihood_new = -9
    Prior_new         = 0.5  #in the bounds but its the worst sample

    loglikelihood_old = -6.5
    Prior_old         = 0.5


    assert (MCMC.MH_acceptance(loglikelihood_new,Prior_new,loglikelihood_old,Prior_old) == 0.0)

def test_input_complex():
    # complex values
    loglikelihood_new = 8j
    Prior_new        = 2j

    loglikelihood_old = 2j
    Prior_old      = 0.5j

    with pytest.raises(TypeError, match=r'Please check input values, either integers or floats'):
        MCMC.MH_acceptance(loglikelihood_new,Prior_new,loglikelihood_old,Prior_old)

def test_zero_division_error():
    # complex values
    loglikelihood_new = 8
    Prior_new        = 2

    loglikelihood_old = 2
    Prior_old      = 0.0

    with pytest.raises(ZeroDivisionError, match=r'Please check division by zero'):
        MCMC.MH_acceptance(loglikelihood_new,Prior_new,loglikelihood_old,Prior_old)
#########################################################################################################


def test_prior():
    #theta within the bounds
    limits = [[5,20],[0,20]]
    theta  = [8,6]

    assert (F.Prior(theta,limits) == 1.0)


def test_prior_1():
    #theta not in the bounds
    limits = [[5,20],[0,20]]
    theta  = [-8,-6]

    assert (F.Prior(theta,limits) == -np.inf)

def test_likelihood():
    np.random.seed(12)

    x    = np.linspace(-5,5)
    sigma = 1
    data = F.model_equation([3.,2.,1.],x)
    args =  [data, sigma, x]

    theta = [3.,2.,1.]

    #should yeild logl = np.sum(-0.5*np.log(2*np.pi))

    assert (F.logLikelihood(theta,args) == np.sum(-0.5*np.log(2*np.pi)))


def test_likelihood_cubic():
    np.random.seed(12)

    x    = np.linspace(-5,5)
    sigma = 2
    data = F.model_equation_cubic([3.,2.,1.,.5],x)
    args =  [data, sigma, x]

    theta = [3.,2.,1.,.5]

    #should yeild logl = np.sum(-0.5*np.log(2*np.pi*4))

    assert (F.logLikelihood_cubic(theta,args) == np.sum(-0.5*np.log(2*np.pi*4)))
