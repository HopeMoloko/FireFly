import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stat
import corner
import copy as Makecopy

np.random.seed(1)
plt.style.use('ggplot')

###############################
# truth
a_true = 10.0
mu = 1.0
sigma = 1.0

# generate mock data

x = np.linspace(-5,5,1000)

yerr = np.random.normal(0,1.0,len(x))

y_true = a_true*np.exp(-0.5*(x-mu)**2/sigma**2)

y_dat = y_true + yerr
##############################


import json
import sys
import numpy
from numpy import log, exp, pi
import scipy.stats, scipy
import pymultinest


# a more elaborate prior
# log-likelihood
def Model_function(a,x):

    model_value = a*np.exp(-0.5*(x-mu)**2/sigma**2)

    return model_value

def loglike(cube,ndim,nparam):
        a  = cube[0]
        model = Model_function(a,x)
        sigma2 = yerr
                        
        Term1 = -0.5*np.log(2*np.pi*(sigma2**2))
        Term2 = -0.5*(1/sigma2**2)*(y_dat-model)**2
                                    
        Log_Like = sum(Term1 + Term2)
                                            
        return Log_Like

                                            # prior transform
def prior_transform(cube,ndim,nparam):
    um  = cube[0]
    a = 15.0*um
    return np.array([a])

# analyse with 1 gaussian

# number of dimensions our problem has
parameters = ["a"]
n_params = len(parameters)
datafile = "chains-analytic1"
# run MultiNest
pymultinest.run(loglike, prior_transform, n_params, outputfiles_basename=datafile , resume = False, verbose = True,n_live_points = 600,max_iter=10000)
json.dump(parameters, open(datafile + 'params.json', 'w')) # save parameter names

# plot the distribution of a posteriori possible models

plt.figure()
plt.plot(x, y_dat, '+ ', color='red', label='data')
am = pymultinest.Analyzer(outputfiles_basename=datafile , n_params = n_params)
for a in am.get_equal_weighted_posterior()[::1000,:-1]:
	plt.plot(x, Model_function(a, x), '-', color='blue', alpha=0.3, label='data')

plt.savefig(datafile + 'posterior.pdf')
plt.close()


am = pymultinest.Analyzer(outputfiles_basename=datafile , n_params = n_params)
a_lnZ = am.get_stats()['global evidence']
print()
print ('************************')
print ('MAIN RESULT: Evidence Z ')
print ('************************')
print ('  log Z for model with 1 line = %.1f' % (a_lnZ / log(10)))
