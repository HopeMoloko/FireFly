import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import corner
import copy as duplicate
import time
from matplotlib import colors
import scipy as sp
import corner

import time
from ipywidgets import IntProgress
from IPython.display import display

from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

###############################################################

def model_equation(thetas,x):
    """sinusoidal Model equation for the simulated data
        a*x*np.sin(b*x + c)
    Parameters

    ------------------
    x  : array
        x input values
    a : scalar
        amplitude
    b : scalar
        frequency
    c: scalar
        period
    ------------------
    ouput: Array
            y values
    """
    a,b,c = thetas
    y = a*x*np.sin(b*x + c)

    return y


def logLikelihood(thetas,args2):

    """Simple gaussian Likelihood

    Parameters
    --------------
    thetas : array
            parameter values [a,b,c,d]

    *args : list
            Likelihood extra arguments

    --------------
    output: scalar
            loglikelihood value
    """

    #args_loglike = [y_values, erorr , x_values]
    mu_data , sigma,  x = args2

    mu_model = model_equation(thetas,x)

    loglikelihood_value = np.sum(-0.5*np.log(2*np.pi*(sigma**2))) - np.sum(((mu_data-mu_model)**2)/(2*sigma**2))

    return loglikelihood_value


def Prior(thetas,limits):
    """Evaluate the prior probability of mu

    Parameters

    ----------------

    thetas : array
            parameter values [a,b,c,d]

    limits: list or array
            The limits of [a,b,c,d]

    *args : list
            prior extra arguments

    ----------------
     output: scalar
            The Prior of parameters [a,b,c,d].
    """

    # ordered (alimits, blimits , climits , dlimits = limits)

    # this is how they are ordered (a, b, c ,d = theta)

    #If position is out of limits discard

    for i in range(len(thetas)):

        if (thetas[i] < limits[i][0]) or (thetas[i] > limits[i][1]):
            return -np.inf
        #else:
        #    return prior = 1.0  #/(limits[i][1] - limits[i][0])


    return 1.0

def model_equation_cubic(thetas,x):
    """ Cubic model equation for the simulated data
                  a*x**3 + b*x**2 + c*x + d
    Parameters

    ------------------
    x  : array
        x input values
    a : scalar
        cubic function parameter  a
    b : scalar
        cubic function parameter  b
    c: scalar
        cubic function parameter  c
    d: scalar
        cubic function parameter   d
    ------------------
    ouput: Array
            y values
    """
    a,b,c,d = thetas
    y = a*x**3 + b*x**2 + c*x + d

    return y

def logLikelihood_cubic(thetas,args2):

    """Simple gaussian Likelihood

    Parameters
    --------------
    thetas : array
            parameter values [a,b,c,d]

    *args : list
            Likelihood extra arguments

    --------------
    output: scalar
            loglikelihood value
    """

    #args_loglike = [y_values, erorr , x_values]
    mu_data , sigma,  x = args2

    mu_model = model_equation_cubic(thetas,x)

    loglikelihood_value = np.sum(-0.5*np.log(2*np.pi*(sigma**2))) - np.sum(((mu_data-mu_model)**2)/(2*sigma**2))

    return loglikelihood_value
