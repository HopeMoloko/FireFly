#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals, print_function
import numpy
from numpy import pi, cos
from pymultinest.solve import solve
import os
if not os.path.exists("chains"): os.mkdir("chains")
    
    
########################    MY IMPORTS  ######################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stat
import corner
import copy as Makecopy

#%matplotlib nbagg
#%matplotlib inline
np.random.seed(1)
plt.style.use('ggplot')
######################################################

######################## Read in the Data ###########################
Data = pd.read_csv('example_data.txt',delimiter=' ',names=['x','y','err'])
x_values = Data.x
y_values = Data.y
error_values = Data.err
#####################################################################

    
    
 
# probability function, taken from the eggbox problem.

def Model_function(params,x):
    '''Return the model value given a,b,c, and x paramters'''
    
    a = params[0]
    b = params[1]
    c = params[2]
    model_value = a*x*np.sin(b*x+c)
    return model_value

def Log_Likelihood(data=None,sigma=None,model=None):
    
    '''Return the Log-Likelihood.
    lnLikelihood = sum(f(data|model,sigma^2))
                =-(n/2)*ln(2*pi) - (n/2)*ln(sigma^2) - (1/2*sigma^2)*sum((data-model)^2)    
    Keyword arguments:
    sigma -- noise level
    model    -- Model
    data     -- The data'''
    #Unit test for shape of data amd model
    assert data.shape == model.shape
    
    
    Term1 = -0.5*np.log(2*np.pi*(sigma**2))
    Term2 = -0.5*(1/sigma**2)*(data-model)**2
    
    Log_Like = sum(Term1 + Term2)
    
    return Log_Like

def prior_transform(u = None):
    '''Return the transformed prior space,array.'''
    
    a = 4.0*u[0]
    b = 3.0*u[1]
    c = 3.1*u[2]
    
    return np.array([a,b,c])

# number of dimensions our problem has
parameters = ["a", "b","c"]
n_params = len(parameters)
# name of the output files
prefix = "chains/3-"

# run MultiNest
result = solve(LogLikelihood=Log_Likelihood, Prior=prior_transform, 
	n_dims=n_params, outputfiles_basename=prefix, verbose=True)

print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for name, col in zip(parameters, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

# make marginal plots by running:
# $ python multinest_marginals.py chains/3-
# For that, we need to store the parameter names:
import json
with open('%sparams.json' % prefix, 'w') as f:
json.dump(parameters, f, indent=2)