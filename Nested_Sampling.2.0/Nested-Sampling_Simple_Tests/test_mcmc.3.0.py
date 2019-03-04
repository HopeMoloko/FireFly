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

import Metropolis_Hasting_3_0 as MH
import pytest

#####################################


# Needed functions
def logLikelihood(mu,args2):

    """Simple gaussian Likelihood

    Parameters
    --------------
    mu_true : scalar
            True mean value
    mu     : scalar
            sampled mu

    *args : list
            Likelihood extra arguments
    """

    mu_val = args2[0]


    L = (1/np.sqrt(2*np.pi)*np.exp(-(mu_val-mu)**2))

    return np.log(L)


def Prior(mu,args1):


    """Evaluate the prior probability of mu

    Parameters

    ----------------

    mu : scalar
            mean

    mu_limits: list or array
            The limits of mu

    *args : list
            prior extra arguments

    ----------------
     output: scalar
            The log Prior of mu.

    """

    mulimits = args1[0]
    #print(mulimits)
    #If position is out of limits discard
    if (mu < mulimits[0]) | (mu > mulimits[1]):
        mu_prior = np.nan
    else:
        mu_prior = 1.0/(mulimits[1] - mulimits[0])

    return mu_prior


#Example problem
x_vals = np.linspace(0,10)
y_vals = (1/np.sqrt(2*np.pi))*np.exp(-(x_vals-5)**2)

plt.plot(x_vals,y_vals)
plt.xlabel('$\mu$')
plt.show()


# MCMC INITIAL VALUES
# MCMC steps per NS iteration
mcmc_steps = 10000

# mcmc stepsize
mustep = 0.5
stepsize = np.array([mustep])

#Intial value
theta = 3.2
mu_value = 5.0
mulimits = [0,10]

#Likelihood function arguments
args_loglike = [mu_value]

#Prior function arguments
args_prior = [mulimits]


chain_particles , chain_loglikelihood_particles , chain_prior_particles, Acceptance_ratio_particles = MH.MH_mcmc(logLikelihood,Prior,theta,args_loglike,args_prior,mcmc_steps,stepsize)

print(np.mean(chain_particles))
plt.hist(chain_particles)
plt.xlabel('$\mu$')
plt.show()

pytest.approx(np.mean(chain_particles),mu_value,0.1)

print(Acceptance_ratio_particles)
