########################Import Modules##########################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import corner

from tabulate import tabulate


import Metropolis_Hasting_3_0T as MH
import Functions as F

#####################################

def test_mcmc():
    try:
        assert a-3*a_std < a_true < a+3*a_std
        assert b-3*b_std < b_true < b+3*b_std
        assert c-3*c_std < c_true < c+3*c_std
        assert d-3*d_std < d_true < d+3*d_std

        print(tabulate([['A', a ,a_std, a_true], ['B', b, b_std, b_true], ['C', c, c_std, c_true], ['D', d, d_std, d_true]], \
                        headers=['Parameter', 'Estimated', 'Standard dev', 'True value']))
        print(' Test Passed!')

    except AssertionError:
        print('WARNING!!  : True value not within 3*sigma')



if __name__ == "__main__":
    #Read simulated data
    Data  = pd.read_csv('example_cubic_data.txt', delimiter = ' ', names = ['x', 'y', 'err'])

    x_values = Data.x
    y_values = Data.y
    erorr = Data.err


    # MCMC INITIAL VALUES
    # MCMC steps per NS iteration
    mcmc_steps = 9000

    # mcmc stepsize
    param_step = [0.5,0.5,0.2,0.3]
    stepsize = np.array(param_step)

    #Intial value
    theta =  np.array([3.2,2.,2.6,3.5])

    mulimits = [[0,10],[0,10],[0,10],[0,10]]

    #Likelihood function arguments
    args_loglike = [y_values, erorr , x_values]

    #Prior function arguments
    args_prior = np.array(mulimits)


    chain_particles , chain_loglikelihood_particles , chain_prior_particles, Acceptance_ratio_particles = MH.MH_mcmc(F.logLikelihood_cubic,F.Prior,theta,args_loglike,args_prior,mcmc_steps,stepsize)



    #Trace plot
    plt.plot(chain_particles,'--')
    plt.savefig('traceplot_cubic')
    plt.show()


    #Pramater mean values
    a, a_std = np.mean(chain_particles[:,0][500:]) , np.std(chain_particles[:,0][500:])
    b, b_std = np.mean(chain_particles[:,1][500:]) , np.std(chain_particles[:,1][500:])
    c, c_std = np.mean(chain_particles[:,2][500:]) , np.std(chain_particles[:,2][500:])
    d, d_std = np.mean(chain_particles[:,3][500:]) , np.std(chain_particles[:,3][500:])

    # Histograms of the paramter [a,b,c,d] samples
    figure = corner.corner(chain_particles,labels=['a','b','c','d'],show_titles=True)
    plt.savefig('histo_cubic')
    plt.show()

    #Plot data and model with estimated parameter [a,b,c,d] values
    y_mod = F.model_equation_cubic([a,b,c,d],x_values)
    #
    plt.plot(x_values,y_mod,'r')
    plt.errorbar(x_values, y_values, yerr = erorr, fmt='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('fit_cubic')
    plt.show()

    #Acceptance Ratio
    print('Acceptance ratio: ',Acceptance_ratio_particles)  #0.15

    # tests mcmc
    a_true = 10
    b_true = 10
    c_true = 1
    d_true = 1

    test_mcmc()
