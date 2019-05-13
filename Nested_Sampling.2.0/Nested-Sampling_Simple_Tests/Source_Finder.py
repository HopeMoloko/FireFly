####################################################################################################
import numpy as np
import copy as duplicate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import Nested_Sampling_3 as NS3
import Metropolis_Hasting_3_0T as MH3

import Nest_iterator_functions as nsf

import pandas as pd
from tabulate import tabulate
#########################################################################################################
from configparser import  ConfigParser
from optparse import OptionParser

import sys
import os
#%matplotlib inline
config = ConfigParser()

usage = "usage: %prog options"
parser = OptionParser(usage=usage)

parser.add_option("--config", "-c", type='string', dest='config',
        default=None, help='Name of config file')
parser.add_option("--output", "-o", type='string', dest='output', 
default='sos.log', help='File for output messages. [name.log]') 
    
(options, args) = parser.parse_args() 
    
config.read(options.config)   # Read from config file optionsS


##  parameters

source_list_path = 	config.get('execute','source_list_name')
pixel_size = 			config.getint('execute','pixel_size')                 #npix by npix image
n_walkers = 			config.getint('execute','n_walkers')
noise_level = 		config.getfloat('execute','noise_level')            #sigma noise level
num_of_model_sources = 	config.getint('execute','num_of_model_sources')  #number of model sources
ndim = 			config.getint('execute','ndim')                   #number of dimensions

nlive_points = config.getint('nest','nlive_points')
nest_steps =  config.getint('nest','nest_steps')

mcmc_steps = config.getint('nest','mcmc_steps')
#param_limits = config.get('nest','param_limits')
#param_step = config.get('nest','param_step')

########################################################################################################

def logLikelihood(thetas):

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

    mu_data , sigma = args2

    Model_simu   = nsf.WALKERS(n_walkers,pixel_size,thetas)
    Model_simu.walkers()
    
    mu_model = Model_simu.source_model

    loglikelihood_value = np.sum(-0.5*np.log(2*np.pi*(sigma**2))) - np.sum(((mu_data-mu_model)**2)/(2*sigma**2))

    return loglikelihood_value


def Prior(thetas):
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
    limits = param_limit
    
    for m in range(len(thetas)):
        theta = thetas[m]
        for i in range(len(theta)):

            if (theta[i] < limits[i][0]) or (theta[i] > limits[i][1]):
                return -np.inf
        #else:
        #    return prior = 1.0  #/(limits[i][1] - limits[i][0])


    return 1.0

#############################################################################
# Set fixed pameters. A ,X ,Y ,R 
def make_source():
        np.random.seed(18)
        source_list = []

        A_list = []
        X_list = []
        Y_list = []
        R_list = []

        for i in range(num_of_model_sources):

                u = np.random.random()
                X_true = pixel_size*np.random.random()
                Y_true = pixel_size*np.random.random()
                R_true = 2
                A_true =  5*np.random.random_sample() + 5
        
                source_list.append([A_true,X_true,Y_true,R_true])
                A_list.append(A_true)
                X_list.append(X_true)
                Y_list.append(Y_true)
                R_list.append(R_true)

        params = np.array(source_list)

        sources = pd.DataFrame(columns=['A_true','X_true','Y_true','R_true'])
        sources['A_true'] = A_list
        sources['X_true'] = X_list
        sources['Y_true'] = Y_list
        sources['R_true'] = R_list

        sources.to_csv('True_sources.csv')

        return params , sources

params , sources = make_source()



#params = np.array([[15,25,25,2],[15,35,5,2],[15,10,10,2]])
#pixel_size = 50
#noise_levelS = 0.7

#n_walkers = 3

#Generate some noise with sigma.

Noise = np.random.normal(0,noise_level,(pixel_size,pixel_size))

# Generate simulation with one source

Simulation = nsf.WALKERS(num_of_model_sources,pixel_size,params)
Simulation.walkers()

# Add noise to the Simulated data.
Model = Simulation.source_model + Noise


fig, (ax) = plt.subplots(figsize=(10,8),nrows=1)

ax.imshow(Model,origin='lower',cmap='afmhot')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
plt.savefig('Simulated_Image.png')
plt.show()

np.savetxt('Model_dataset.csv',Model)
###################################################################
# Generate N samples
# nlive_points = 900
# nest_steps = 45000  #50 000

# mcmc_steps = 80

#Parameter limits
param_limits = [[5,10],[0,50],[0,50],[0,5]]
param_limit = np.array(param_limits)

# mcmc stepsize
param_step =  [.5,1,1,.1]   #[.5,.5,.3,.28]
stepsize = np.array(param_step) 

#Number of parameters
#ndim = 4

args2  = Model , noise_level


labels = ['A','X','Y','R']

#Instiantiate the Explorer
explore = nsf.EXPLORER(logLikelihood, Prior ,stepsize,mcmc_steps)

Image_sources = NS3.Nested_Sampling(logLikelihood, Prior, ndim, n_walkers,nsf.prior_transform_1, nest_steps,\
                                   nlive_points,labels,explore.mcmc_explorer)


Image_sources.nestedsampler()
Image_sources.analyze(fontsize=11)