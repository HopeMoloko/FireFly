#############################################################
import numpy as np
import matplotlib.pyplot as plt
import corner
import time
import copy as duplicate

import progressbar

#test numba
from numba import jit , prange


################################################################


# logarithmic addition log(exp(x)+exp(y))
def plus(x,y):
    if x>y:
        return x+log(1+exp(y-x))
    else:
        return y+log(1+exp(x-y))

class NestedSampler(object):
    """docstring fo NestedSampler."""
    def __init__(self, ):

def nestedsampler(nlive_points,nest_steps,sample_from_prior,Exploration_technique):
    """
    This is an implementation of John Skilling's Nested Sampling algorithm
    for computing the normalizing constant of a probability distribution
    (usually the posterior in Bayesian inference).

    parameters
    -----------------------------
    nlive_points : int

    nest_steps   : int

    sample_from_prior : function

    Exploration_technique : function
    -----------------------------
    return
    """
    Obj = []              # Collection of n objects
    Samples = []          # Objects stored for posterior results
    logwidth = None       # ln(width in prior mass)
    logLstar = None       # ln(Likelihood constraint)
    H    = 0.0            # Information, initially 0
    logZ =-1e300          # ln(Evidence Z, initially 0)
    logZnew = None        # Updated logZ
    copy = None           # Duplicated object
    worst = None          # Worst object
    nest = None           # Nested sampling iteration count
