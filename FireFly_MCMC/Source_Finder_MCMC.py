###############################################################################
#                       FireFly - MCMC                                        #
#                   by : Oarabile Hope Moloko                                 #   
#                          2019                                               #
###############################################################################

####################################################################################################

import numpy as np
import copy as duplicate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import Metropolis_Hasting_v1 as MH3

import FireFly_mcmc_v4 as FireFly
import time



import Main_Firefly as nsf

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



def logLikelihood(thetas):

    """Simple gaussian Likelihood

    Parameters
    --------------
    thetas : array
            parameter values (eg.[X,Y,A,R])

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
                return 1.0
            else:
                return -np.inf

    return 1.0


#############################################################################

def column_labels(label,n_walkers, ndim):

    labels = label
    m = 0
    num_label = 0
    columns = []
    
    for s in range(n_walkers):
        for i in range(ndim):
            lab = labels[m] + '_' + str(num_label)
            columns.append(lab)

            m += 1
            if m > (ndim-1):
                m = 0
                num_label += 1
    return columns

def live_point_switcher(sample,thresh):
    """
    switch off some sources in the live points to on/off
    """
    
    choose = np.random.randint(len(sample))
    choose_state = np.random.uniform()

    if choose_state >= thresh:
        sample[choose][0] = 0.0
    
    else:
        pass
    
    return sample

if __name__ == "__main__":

    params  = np.loadtxt( true_sources_name,delimiter=',',usecols=(1,2,3),skiprows=1)  # load the labels from the genereted csv file
    sources = pd.read_csv( true_sources_name)  # Load the sources from the generated csv


    # Generate some noise with sigma.

    Noise = np.random.normal(0, noise_level, (pixel_size, pixel_size))

    # Generate image-simulation with the given sources

    Simulation = nsf.WALKERS(num_of_real_sources, pixel_size, params)
    Simulation.walkers()

    # Add noise to the Simulated data.
    Model = Simulation.source_model + Noise


    # Plot the generated data: Simulated image
    fig, (ax) = plt.subplots(figsize=(10, 8), nrows=1)

    im = ax.imshow(Model, origin="lower", cmap="afmhot", interpolation="gaussian")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.2)

    plt.colorbar(im, cax=cax)
    plt.savefig(simulated_image)
    ###################################################################


    # Parameter limits
    param_limits = [[5, 10], [0, pixel_size], [0, pixel_size]]  # fixed R
    param_limit = np.array(param_limits)

    # mcmc stepsize
    param_step = [.2,.1,.3]  #Randomly initialize mcmc stepsize

    stepsize = np.array(param_step)

    args2 = Model, noise_level




    label = ['A','X','Y']   #Parameter labels
    # Instiantiate the Explorer
    explore = nsf.EXPLORER(
    logLikelihood, Prior, mcmc_steps, stepper, ndim, n_walkers, thresh, pixel_size
    )


    
    print(
#        "#------------------------------------------------------------------------------#"
   )
    print("n_walkers : ", n_walkers)
    print("no. of real sources :", num_of_real_sources)
    print("stepper : ", stepper)
    print("mcmc runs : ", mcmc_firefly)

    switch = explore.switch_explorer    #Use FireFly explorer

    sample_init                   = nsf.prior_transform_1(pixel_size,ndim,n_walkers)  #Generate live points
    sample_init                   = live_point_switcher(sample_init,thresh) #Switch on/off live points w.r.t threshold.

    initial_theta = sample_init                #LIVE POINTS
    start = time.time()                        
    samples , Loglike , Prior , mcmc_finals , Ratio = FireFly.MH_mcmc(logLikelihood,
    Prior, initial_theta, mcmc_firefly,thresh,stepsize,switch,samples_name)             #Run the MCMC FireFly
    end = time.time()

    seconds = end - start

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    mcmc_firefly = Ratio
 

    Samples_fixed = samples.reshape(len(samples),ndim*n_walkers)  #Reshape samples to store them in pandas dataframe

    COLUMNS = column_labels(label,n_walkers, ndim)
    df = pd.DataFrame(Samples_fixed,columns=COLUMNS)
 
    df.to_csv(samples_name+'.csv',index=False)        #Save samples
    
    np.savetxt(samples_name+'loglike.csv',Loglike,delimiter=",")  #Save loglikelihoods of samples
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
    print('Final steps :',mcmc_finals)
    
  