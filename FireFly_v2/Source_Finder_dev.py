####################################################################################################

import numpy as np
import copy as duplicate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import Nested_sampling_dev as NS3
import Metropolis_Hasting_dev as MH3

import Fire_fly_mcmc_stopper as FireFly
#import FireFly_mcmc as FireFly
import corner
from numba import jit
import time



import nest_functions_dev as nsf

import pandas as pd


#########################################################################################################
from configparser import ConfigParser
from optparse import OptionParser

import sys
import os

#%matplotlib inline
config = ConfigParser()

usage = "usage: %prog options"
parser = OptionParser(usage=usage)

parser.add_option(
    "--config",
    "-c",
    type="string",
    dest="config",
    default=None,
    help="Name of config file",
)
parser.add_option(
    "--output",
    "-o",
    type="string",
    dest="output",
    default="sos.log",
    help="File for output messages. [name.log]",
)

(options, args) = parser.parse_args()

config.read(options.config)  # Read from config file optionsS


##  parameters

thresh = config.getfloat("execute", "threshold")
simulated_image = config.get("execute", "simulated_image_name")
samples_name = config.get("execute", "samples_name")
stepper = config.getint("execute", "stepper")
true_sources_name = config.get("execute", "true_sources_name")
pixel_size = config.getint("execute", "pixel_size")  # npix by npix image
n_walkers = config.getint("execute", "n_walkers")
noise_level = config.getfloat("execute", "noise_level")  # sigma noise level
num_of_real_sources = config.getint(
    "execute", "num_of_real_sources"
)  # number of model sources
ndim = config.getint("execute", "ndim")  # number of dimensions

nlive_points = config.getint("nest", "nlive_points")
nest_steps = config.getint("nest", "nest_steps")

mcmc_steps = config.getint("nest", "mcmc_steps")
mcmc_firefly = config.getint("nest", "mcmc_firefly")



########################################################################################################


@jit
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

    mu_data, sigma = args2

    Model_simu = nsf.WALKERS(n_walkers, pixel_size, thetas)
    Model_simu.walkers()

    mu_model = Model_simu.source_model

    loglikelihood_value = np.sum(-0.5 * np.log(2 * np.pi * (sigma ** 2))) - np.sum(
        ((mu_data - mu_model) ** 2) / (2 * sigma ** 2)
    )

    return loglikelihood_value


@jit
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
            0.0 or 1.0
    """

    # ordered (alimits, blimits , climits , dlimits = limits)

    # this is how they are ordered (a, b, c ,d = theta)

    # If position is out of limits discard

    for source in thetas:
        # source = [A,X,Y]
        # param_limit` = np.array([[5,10],[0,50],[0,50]])

        a_lim = param_limit[0]
        x_lim = param_limit[1]
        y_lim = param_limit[2]

        # check if the X paramater is within the unfiorm specified range
        # if not return -inf
        if (source[1] <= x_lim[0]) or (source[1] >= x_lim[1]):

            return -np.inf

        # check if the Y paramater is within the unfiorm specified range
        # if not return -inf
        if (source[2] <= y_lim[0]) or (source[2] >= y_lim[1]):

            return -np.inf

        # check if the A paramater is within the unfiorm specified range
        # or if it set to 0.0

        if (source[0] <= a_lim[0]) or (source[0] >= a_lim[1]):

            # if set to 0.0 then return 1.0
            # else return -np.inf since it's not within the uniform range
            if source[0] == 0.0:
                pass
            else:
                return -np.inf

    return 1.0


#############################################################################

def save_results(label,n_walkers, ndim, keeps):
    """
    reshape file and 
    Save results to a csv dataframe using pandas
    """
    New_keep = np.array(keeps)
    df = pd.DataFrame()
    labels = label
    

    m = 0
    num_label = 0
    #print(New_keep[0:])
    for s in range(n_walkers):
        for i in range(ndim):

            df[labels[m] + '_' + str(num_label)] = pd.Series(New_keep[:,i])
            #print(New_keep[:,i])

            m += 1
            if m > (ndim-1):
                m = 0
                num_label += 1
    

    return df

def live_point_switcher(sample,thresh):
    """
    switch off some sources in the live points
    """
    
    choose = np.random.randint(len(sample))
    choose_state = np.random.uniform()

    if choose_state >= thresh:
        sample[choose][0] = 0.0
    
    else:
        pass
    
    return sample

if __name__ == "__main__":

    params  = np.loadtxt(true_sources_name,delimiter=',',usecols=(1,2,3),skiprows=1)
    sources = pd.read_csv(true_sources_name)

    #labels = np.loadtxt(true_sources_name)
    #print(labels)

    # n_walkers = 3

    # Generate some noise with sigma.

    Noise = np.random.normal(0, noise_level, (pixel_size, pixel_size))

    # Generate simulation with one source

    Simulation = nsf.WALKERS(num_of_real_sources, pixel_size, params)
    Simulation.walkers()

    # Add noise to the Simulated data.
    Model = Simulation.source_model + Noise


    fig, (ax) = plt.subplots(figsize=(10, 8), nrows=1)

    im = ax.imshow(Model, origin="lower", cmap="afmhot", interpolation="gaussian")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.2)

    plt.colorbar(im, cax=cax)
    plt.savefig(simulated_image)
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    #np.savetxt("Model_dataset.csv", Model)
    ###################################################################


    # Parameter limits
    param_limits = [[5, 10], [0, pixel_size], [0, pixel_size]]  # just fix R
    param_limit = np.array(param_limits)

    # mcmc stepsize
    param_step = [1.5, 2.5, 2.5]
    stepsize = np.array(param_step)

    # Number of parameters
    # ndim = 4

    args2 = Model, noise_level


    print('number:',np.random.random_sample(5))

    label = ['A','X','Y']
    # Instiantiate the Explorer
    explore = nsf.EXPLORER(
    logLikelihood, Prior, stepsize, mcmc_steps, stepper, ndim, n_walkers, thresh
    )


    
    print(
#        "#------------------------------------------------------------------------------#"
   )
    print("n_walkers : ", n_walkers)
    print("no. of real sources :", num_of_real_sources)
    print("stepper : ", stepper)
    print("mcmc runs : ", mcmc_firefly)

    switch = explore.switch_explorer

    sample_init                   = nsf.prior_transform_1(ndim,n_walkers)
    sample_init                   = live_point_switcher(sample_init,thresh)

    initial_theta = sample_init
    start = time.time()
    samples , Loglike , Prior , Ratio = FireFly.MH_mcmc(logLikelihood,
    Prior, initial_theta, mcmc_firefly,thresh,switch)
    end = time.time()

    seconds = end - start

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    mcmc_firefly = Ratio
 

    Samples_fixed = samples.reshape(len(samples),ndim*n_walkers)

    df = save_results(label,n_walkers,ndim,Samples_fixed)
    
    
    df.to_csv(samples_name+'.csv',index=False)
    
    np.savetxt(samples_name+'loglike.csv',Loglike,delimiter=",")
    #Posterior results
    

    print("{} h {} m {} s".format(h, m, s))
    print(
       "Output filenames : "
       + samples_name
       + ","
       + true_sources_name
       + ","
       + simulated_image
    )
    print('Acceptance Ratio :',Ratio)
    
    figure = corner.corner(Samples_fixed,show_titles=True,title_kwargs={"fontsize": 12})   #edited
    plt.savefig(samples_name+'.png')
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    
   