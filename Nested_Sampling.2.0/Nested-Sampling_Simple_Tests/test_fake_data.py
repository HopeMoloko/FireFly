########################Import Modules##########################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import corner

from tabulate import tabulate

import time


import Metropolis_Hasting_3_0T as MH
import Functions as F


#####################################################

def test_mcmc():
    try:
        assert a-3*a_std < a_true < a+3*a_std
        assert b-3*b_std < b_true < b+3*b_std
        assert c-3*c_std < c_true < c+3*c_std

        print(tabulate([['A', a ,a_std, a_true], ['B', b, b_std, b_true], ['C', c, c_std, c_true]], \
                        headers=['Parameter', 'Estimated', 'Standard dev', 'True value']))
        print(' Test Passed!')

    except AssertionError:
        print('WARNING!!  : True value not within 3*sigma')

if __name__ == "__main__":
    #Read simulated data
    Data  = pd.read_csv('example_data.txt', delimiter = ' ', names = ['x', 'y', 'err'])

    x_values = Data.x
    y_values = Data.y
    erorr = Data.err



    # MCMC INITIAL VALUES
    # MCMC steps per NS iteration
    mcmc_steps = 9000

    ## mcmc stepsize
    param_step =  [.2,.3,.2] #was [0.5,0.5,0.2]
    stepsize = np.array(param_step)

    #Intial value
    theta =  np.array([1.5,0.3,.5])

    param_limits = [[0,10],[0,10],[0,10]]  #[[0,20],[0,20],[0,20]]

    #Likelihood function arguments
    args_loglike = [y_values, erorr , x_values]

    #Prior function arguments
    args_prior = np.array(param_limits)


    chain_particles , chain_loglikelihood_particles , chain_prior_particles, Acceptance_ratio_particles = MH.MH_mcmc(F.logLikelihood,F.Prior,theta,args_loglike,args_prior,mcmc_steps,stepsize)


    #print(np.mean(chain_particles))
    plt.plot(chain_particles,'--')
    #plt.savefig('traceplot')
    plt.show()



    #MEAN values
    a, a_std = np.mean(chain_particles[:,0][500:]), np.std(chain_particles[:,0][500:])
    b, b_std = np.mean(chain_particles[:,1][500:]), np.std(chain_particles[:,1][500:])
    c, c_std = np.mean(chain_particles[:,2][500:]), np.std(chain_particles[:,2][500:])

    figure = corner.corner(chain_particles,labels=['a','b','c'],show_titles=True)
    #plt.savefig('histo')
    plt.show()
    #

    #Plot simulated data and model with estimated parameter values
    y_mod = F.model_equation([a,b,c],x_values)

    plt.plot(x_values,y_mod,'r')
    plt.errorbar(x_values, y_values, yerr = erorr, fmt='o')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.savefig('fit')
    plt.show()

    #Acceptance
    print(Acceptance_ratio_particles)
    a_true = 2
    b_true = 0.5
    c_true = 1

    test_mcmc()
