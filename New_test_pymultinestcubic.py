import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stat
import corner
import copy as Makecopy

np.random.seed(1)
plt.style.use('ggplot')


Data = pd.read_csv('example_cubic_data-Copy1.txt',delimiter=' ',names=['x','y','err'])
x_values = Data.x
y_values = Data.y
error_values = Data.err


import json
import sys
import numpy
from numpy import log, exp, pi
import scipy.stats, scipy
import pymultinest


# a more elaborate prior
def Model_function(cube,x):
    '''Return the model value given a,b,c, and x paramters'''
    
    a = cube[0]
    b = cube[1]
    c = cube[2]
    d = cube[3]
    model_value = a*x**3 + b*x**2 + c*x + d
    return model_value

def Log_Likelihood(cube,ndim, nparam):
    
    '''Return the Log-Likelihood.
    lnLikelihood = sum(f(data|model,sigma^2))
                =-(n/2)*ln(2*pi) - (n/2)*ln(sigma^2) - (1/2*sigma^2)*sum((data-model)^2)    
    Keyword arguments:
    sigma -- noise level
    model    -- Model
    data     -- The data'''
    #Unit test for shape of data amd model
    x = x_values
    data = y_values
    sigma = error_values
    model = Model_function(cube,x)
    assert data.shape == model.shape
    
    
    Term1 = -0.5*np.log(2*np.pi*(sigma**2))
    Term2 = -0.5*(1/sigma**2)*(data-model)**2
    
    Log_Like = sum(Term1 + Term2)
    
    return Log_Like

def prior_transform(cube,ndim, nparam):
    '''Return the transformed prior space,array.'''
    
    a = 15.0*cube[0]
    b = 15.0*cube[1]
    c = 15.0*cube[2]
    d = 15.0*cube[3]
    
    return np.array([a,b,c,d])

# analyse with 1 gaussian

# number of dimensions our problem has
parameters = ["a", "b", "c","d"]
n_params = len(parameters)
datafile = "chains-cubic-unconstrained-"
# run MultiNest
pymultinest.run(Log_Likelihood, prior_transform, n_params, outputfiles_basename=datafile , resume = False, verbose = True,n_live_points = 400)
json.dump(parameters, open(datafile + 'params.json', 'w')) # save parameter names

# plot the distribution of a posteriori possible models

plt.figure()
plt.plot(x_values, y_values, '+ ', color='red', label='data')
am = pymultinest.Analyzer(outputfiles_basename=datafile , n_params = n_params)
for (a, b, c,d) in am.get_equal_weighted_posterior()[::1000,:-1]:
	plt.plot(x_values, Model_function([a,b,c,d], x_values), '-', color='blue', alpha=0.3, label='data')

plt.savefig(datafile + 'posterior.pdf')
plt.close()

#a = pymultinest.Analyzer(outputfiles_basename=datafile + '_1_', n_params = n_params)
a_lnZ = am.get_stats()['global evidence']
print()
print ('************************')
print ('MAIN RESULT: Evidence Z ')
print ('************************')
print ('  log Z for model with 1 line = %.1f' % (a_lnZ / log(10)))
