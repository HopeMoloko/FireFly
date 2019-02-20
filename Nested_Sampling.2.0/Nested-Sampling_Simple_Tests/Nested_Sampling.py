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

####Import test mcmc module

import Metropolis_Hasting as MCMC

from ipywidgets import IntProgress
from IPython.display import display

from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

#####################################


def Nested_Sampling(nest_runs,particles,Likelihood_of_particles,prior_of_particles,\
                    Exploration_function,Exploration_args,Exploration='mcmc',plots=True):
    """Nested Sampling
    
    Parameters
    
    -------------
    
    nest_runs: scalar
               Number of nested sampling runs
    
    particles: array or list
                Generated walkers or particles
                
    Likelihood_of_particles: array or list
                            Likelihood values of Generated walkers
                            
    prior_of_particles: array or list
                        Prior values of generated walkers
                        
    Exploration : string
                the explorartion technique to be used.
    
    Exploration_function: Function
                            Type of exploration technique for generating a new sample
                            
    Exploration_args: array or list or dict
                                Arguments to be passed to the exploration function
       
    plots  : bool
            render plots
    
    -----------------
    Output : If plots is True
            Plots: 1. Acceptance Ratio plot
                   2.Likelihood vs Prior Mass plot (usually LogL vs LogX)
                   3.Posterior weights vs Prior Mass
                   4.Posterior samples plots (usually histograms)
                   5. Z distribtion plot (usually logZ histogram)
            #Information
            scalar,scalar,scalar
            logZ (Z Evidence) , Information H , Effective Sample size
            
            #.txt files
            1. Posterior_Samples.txt (Only the effective posterior samples)
            2. Keep.txt   (All the information from Nested Sampling)
    """
    
    #Initials
    # Storage for results
    keep = np.empty((nest_runs, 1 + 1))

    #store acceptance ratio and rejectance
    nacceptance = np.zeros(nest_runs)
    nrejectance = np.zeros(nest_runs)
    
    if Exploration=='mcmc':
        
            Likelihood,args_like,Prior,args_prior,stepsize,mcmc_steps = Exploration_args
    else:
        """Will be updated"""
        pass
    
    for i in range(nest_runs):
    
        # Particle with worst likelihood
        worst = np.argmin(Likelihood_of_particles)   
    
        # Save the details of the worst particle, (likelihood)
        keep[i,:-1] = particles[worst] 
        keep[i,-1] = Likelihood_of_particles[worst]
    
        # Copy random survivor
        #----copy a random point and do mcmc from there-----
        while True:
            copy = np.random.randint(len(particles))
            if (copy != worst):break
            
        # Throw away worst particle and replace with random particle 
        #and do mcmc from the random particle (will change with new sample)
        particles[worst] = duplicate.deepcopy(particles[copy]) 
    
        particle_new_copy =  particles[worst]
    
        # Likelihood of worst point
        logl_threshold = duplicate.deepcopy(Likelihood_of_particles[worst])
        logp_threshold = prior_of_particles[worst]
    
    
        theta = particle_new_copy
        likelihood_theta = logl_threshold
        prior_theta = logp_threshold
        #Do mcmc on the survivor particle
        New_particle , New_prior , New_like , Acceptance_R = MCMC.Metropolis_Hasting_NS(Likelihood,args_like,\
                                                        Prior,args_prior,theta,stepsize,mcmc_steps,\
                                                        likelihood_theta,prior_theta)
    
        #copy new particle from worst
        particles[worst]                    = New_particle
        Likelihood_of_particles[worst]      = New_like
        prior_of_particles[worst]           = New_prior
    
        #Save acceptance ratio for each mcmc
        nacceptance[i] = Acceptance_R

    #Prior Mass    
    # evalute prior mass
    logX = -(np.arange(0, (nest_runs))+ 1.)/len(particles)  
    
    # plots
    if plots == True:
        
        #Acceptance Ratio Plot
        #Visualize Likelihood vs Prior Mass
        fig , (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,8))


        ax1.plot(nacceptance,'o')
        ax1.set_ylabel('Acceptance Ratio')

        ax2.plot(logX, keep[0:(nest_runs), -1], 'bo')
        ax2.set_ylabel('$\\log(L)$')
        ax2.set_xlabel('$log(X)$')


        #Posterior weights

        logwt = logX.copy() + keep[0:(nest_runs), -1]

        # Normalised posterior weights
        wt = np.exp(logwt - logwt.max())              
        ax3.plot(logX, wt, 'bo-')
        ax3.set_ylabel('Posterior weights (relative)')
        ax3.set_xlabel('$\\log(X)$')
        fig.savefig('Plots')
    

    return None