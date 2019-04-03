#############################################################
import numpy as np
import matplotlib.pyplot as plt
import corner
import time
import copy as duplicate

#test numba
from numba import jit , prange


################################################################




# Useful function Compute the log of the sum of exponentials of input elements.
# OR from scipy.misc import logsumexp

def logsumexp(values):
    """ Computes the logsum of exponentials

    Parameters

    ----------------
    values  : array
            log values
    ---------------
    return  : array
            logsum"""
    biggest = np.max(values)
    x = values - biggest
    result = np.log(np.sum(np.exp(x))) + biggest
    return result


######### Main Nested Sampling ##################

class Nested_Sampling:
    """ Main Nested Sampling Algorithm

    Parameters

    ------------------
    particles :  array
                N drawn samples from the prior(live samples)

    loglikelihood_particles : array
                            loglikelihood values of each particle (sample in particles)

    Prior     : array
              Prior values of each particle (sample in particles)
    num_params: integer
                Number of Parameters

    nest_steps: integer
                Number of nested sampling runs

    Exploration_technique : function
                            returns: new particle , new loglikelihood value , new prior value
    """

    def __init__(self, particles, loglikelihood_particles,num_params, Prior, nest_steps, Exploration_technique):

        self.particles               = particles
        self.loglikelihood_particles = loglikelihood_particles
        self.num_params              = num_params
        self.Prior                   = Prior
        self.nest_steps              = nest_steps
        self.Exploration_technique   = Exploration_technique

        self.logX                    = None
        self.logZ                    = None
        self.posterior_samples       = None
        self.wt                      = None
        self.nsize                   = None

        #Storage of results
        self.keep                    = np.empty((self.nest_steps, self.num_params+1))

        #Acceptance ratios
        self.nacceptance             = np.array([])


    def NestedSampler(self):

        """ The main nested sampling algorithm"""

        print('Begin nested sampling...')
        print('No. of runs : ',self.nest_steps)
        print('No. of parameters : ',self.num_params)

        for i in prange(self.nest_steps):

            # Index of particle with worst likelihood values
            worst  = np.argmin(self.loglikelihood_particles)

            #Save the detaila of the worst particle
            self.keep[i,:-1]   = self.particles[worst]
            self.keep[i,-1]    = self.loglikelihood_particles[worst]


            #Copy random survivor
            while True:
                copy = np.random.randint(len(self.particles))
                if (copy != worst): break

            #Discard the worst particle and replace with random survivor
            #and do exploration from the copied particles

            particle_copy = self.particles[copy]

            #Take likelihood of worst particles as the likelihood constraint
            loglikelihood_constraint = self.loglikelihood_particles[worst]  #L*


            Sample_chain, loglikelihood_chain, Prior_chain, Acceptance_ratio = self.Exploration_technique(particle_copy)

            if max(loglikelihood_chain) > loglikelihood_constraint:
                # Accept sample with higher likelihood than the likelihood constraint
                # Replace worst sample with new sample

                self.loglikelihood_particles[worst]   = max(loglikelihood_chain)
                self.particles[worst]          = duplicate.deepcopy(Sample_chain[np.argmax(loglikelihood_chain)])
                self.Prior[worst]              = duplicate.deepcopy(Prior_chain[np.argmax(loglikelihood_chain)])

                #Save acceptance ratio for each mcmc
                self.nacceptance  = np.append(self.nacceptance, Acceptance_ratio)

        #Evaluate the Prior Mass

        self.logX    = -(np.arange(0, (self.nest_steps))+1.)/len(self.loglikelihood_particles)

        # Evaluate Posterior Weights
        logwt  = self.logX.copy() + self.keep[0:(self.nest_steps), -1]
        self.wt = np.exp(logwt - logwt.max())

        #Normalised posterior weights
        normalised_wt = self.wt/self.wt.sum()

        #Prior Weights
        logw  = self.logX.copy()
        #Normalised Prior Weights
        logw -= logsumexp(logw)

        # Effective Sample stepsize
        effective_sample_size = int(np.exp(-np.sum(normalised_wt*np.log(normalised_wt + 1E-300))))

        #Evaluate marginal Likelihood
        self.logZ = logsumexp(logw + self.keep[:,-1])
        #Evaluate the information
        H   = np.sum(normalised_wt*(self.keep[:,-1] - self.logZ))


        print('\n'+'logZ = {logZ} '.format(logZ=self.logZ))
        print('Information = {H}'.format(H=H))
        print('Effective Sample Size = {ess}'.format(ess=effective_sample_size))


        ####################### Posterior Samples ##################################
        self.posterior_samples = np.empty((effective_sample_size, self.keep.shape[1]))
        k = 0
        while True:
          # Choose one of the samples
            which = np.random.randint(self.keep.shape[0])

          # Acceptance probability
            prob = normalised_wt[which]/normalised_wt.max()

            if np.random.rand() <= prob:
                self.posterior_samples[k, :] = self.keep[which, :]

                k += 1

            if k >= effective_sample_size:
                break

        np.savetxt('keep_Many.txt', self.posterior_samples)

        print('\n'+'End nested sampling.')
        
    def analyze(self,labels,truths=None):
        """ Visualize the results from nested sampling

        Parameters

        --------------
        labels : list ['param1','param2',etc]
                Parameter labels

        truths : list
                True parameter values, default = None
                """
        #Plot the acceptance Ratio plot and save it
        plt.figure(figsize=(10,8))
        plt.plot(self.nacceptance,'o')
        plt.ylabel('Acceptance Ratio')
        plt.savefig('Acceptance_Ratio_plot.png')
        plt.show()


        # Plot logl vs Prior mass and Posterior vs Prior ass
        fig, (ax1, ax2) = plt.subplots(figsize=(12,10), nrows=2)
        ax1.plot(self.logX, self.keep[0:(self.nest_steps), -1], 'bo')
        ax1.set_ylabel('$\\loglikelihood$')

        ax2.plot(self.logX, self.wt, 'bo-')
        ax2.set_ylabel('Posterior weights (relative)')
        ax2.set_xlabel('$log(X)$')

        plt.show()


        #Posterior results
        figure = corner.corner(self.posterior_samples[:,:-1],labels=labels ,truths = truths, show_titles=True)
        plt.savefig('Posterior_Results.png')
        plt.show()

    def logZ_distribution(self,nsize):
        """ Evaluate the evidence distribution

        Parameters
        -------------
        nszie  : integer
                Sample size
        ------------
        returns: logZ distribution histogram"""

        logZ_dist  = np.array([])

        for i in prange(nsize):
            uniform_numbers  = np.log(np.random.uniform(0,1,len(self.keep[:,-1])))
            number           = 0
            logX_mass        = np.array([])

            for num in uniform_numbers:
                number += num
                logX_mass = np.append(logX_mass,number/len(self.particles))

            logw_dist       = logX_mass
            logw_dist      -= logsumexp(logw_dist)

            logZ_d         = logsumexp(logw_dist + self.keep[:,-1])

            logZ_dist     = np.append(logZ_dist,logZ_d)

        fig, ax = plt.subplots(figsize=(10,8), nrows = 1)
        ax.hist(logZ_dist);
        ax.set_xlabel('logZ')
        plt.show()
        plt.savefig('logZ_distribution.png')
