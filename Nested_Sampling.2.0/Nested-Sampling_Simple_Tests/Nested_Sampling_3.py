#############################################################
import numpy as np
import matplotlib.pyplot as plt
import corner
import time
import copy as duplicate

import progressbar

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
            large values
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
    logLikelihood : function
                    The loglikelihoodfunction

    Prior        : function
                   The Prior function
    ndim       : integer
                Number of Parameters

    nest_steps: integer
                Number of nested sampling runs

    Exploration_technique : function
                            returns: new particle , new loglikelihood value , new prior value
    """

    def __init__(self,loglikelihood,Prior, ndim, prior_transform, nest_steps,nlive_points, Exploration_technique):

        self.loglikelihood           = loglikelihood
        self.Prior                   = Prior
        self.ndim                    = ndim
        self.nest_steps              = nest_steps
        self.Exploration_technique   = Exploration_technique
        self.nlive_points            = nlive_points

        self.posterior_samples       = None
        self.nsize                   = None

        #Storage of results
        self.keep                    = np.empty((nest_steps, ndim+1))

        #Acceptance ratios
        self.nacceptance             = np.array([])

        if ndim != None:

            self.draw_uniform = lambda: np.random.uniform(0, 1, size=ndim)  #function

        else:
            pass

        loglikelihood_particles = np.zeros((nlive_points))
        Prior_particles = np.zeros((nlive_points))

        #particles
        particles = np.zeros((nlive_points,ndim))


        #Draw N samples
        print('Begin Drawing Live points from prior...')
        print('No. live points : ',nlive_points)
        for i in prange(nlive_points):
            u                          = self.draw_uniform()
            sample                     = prior_transform(u)
            particles[i]               = sample
            loglikelihood_particles[i] = loglikelihood(sample)
            Prior_particles[i]         = Prior(sample)

        self.particles               = particles
        self.loglikelihood_particles = loglikelihood_particles
        self.Prior                   = Prior_particles

        print('End Draw!')

    def nestedsampler(self):

        """ The main nested sampling algorithm"""

        print('Begin nested sampling...')
        print('No. of runs : ',self.nest_steps)
        print('No. of parameters : ',self.ndim)


        bar = progressbar.ProgressBar(max_value=self.nest_steps)
        for i in prange(self.nest_steps):

            # Index of particle with worst likelihood values
            worst  = np.argmin(self.loglikelihood_particles)

            #Save the detaila of the worst particle
            self.keep[i,:-1]   = self.particles[worst]
            self.keep[i,-1]    = self.loglikelihood_particles[worst]


            #Copy random survivor
            copy = np.random.choice(np.concatenate(\
               (np.arange(1,worst),np.arange(worst+1,len(self.particles)))))

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

            bar.update(i)

        #Evaluate the Prior Mass

        logX       = -(np.arange(0, (self.nest_steps))+1.)/len(self.loglikelihood_particles)

        # Prior width  wi = 0.5*(Xi-1 Xi+1)
        logPrior_width = np.array([0.5*(logX[i]-logX[i+1]) for i in range(len(logX)-1)])

        logPrior_width = np.append(logPrior_width,0.5*logX[-1])

        # Calculate the Evidence ( marginal likelihood equaion 8) Z = SUM(Li*wi)

        logZ = logsumexp(logPrior_width + self.keep[:,-1])

        # Importance weights (equation 10)

        logImportance_weights = np.array([ (li+wi)-logZ for li,wi in zip(self.keep[:,-1],logPrior_width)])

        Importance_weights = np.exp(logImportance_weights)


        #Evaluate the information
        H = int(np.exp(-np.sum(Importance_weights*np.log(Importance_weights + 1E-300))))

        self.logX  = logX
        self.logZ = logZ
        self.Importance_weights = Importance_weights

        print('\n'+'logZ = {logZ} '.format(logZ=self.logZ))
        print('Information = {H} nats'.format(H=H))


        ####################### Posterior Samples ##################################
        posterior_samples = np.empty((H, self.keep.shape[1]))
        k = 0
        while True:
          # Choose one of the samples
            which = np.random.randint(self.keep.shape[0])

          # Acceptance probability
            prob = self.Importance_weights[which]/self.Importance_weights.max()

            if np.random.rand() <= prob:
                posterior_samples[k, :] = self.keep[which, :]

                k += 1

            if k >= H:
                break

        np.savetxt('keep_Many.txt', posterior_samples)

        self.posterior_samples = posterior_samples

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

        ax2.plot(self.logX, self.Importance_weights/self.Importance_weights.max(), 'bo-')
        ax2.set_ylabel('Importance weights (relative)')
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

        logZ_distribution  = np.array([])

        for i in prange(nsize):
            uniform_numbers  = np.log(np.random.uniform(0,1,len(self.keep[:,-1])))
            number           = 0
            logX_mass        = np.array([])

            for num in uniform_numbers:
                number += num
                logX_mass = np.append(logX_mass,number/len(self.particles))

            # Prior width
            logPrior_width_un = np.array([0.5*(logX_mass[i]-logX_mass[i+1])\
                                          for i in range(len(logX_mass)-1)])

            logPrior_width_un = np.append(logPrior_width_un,0.5*logX_mass[-1])


            logZ_un = logsumexp(logPrior_width_un + self.keep[:,-1])                        #Evaluate logZ

            logZ_distribution.append(logZ_un)

        fig, ax = plt.subplots(figsize=(10,8), nrows = 1)
        ax.hist(logZ_distribution);
        ax.set_xlabel('logZ')
        plt.show()
        plt.savefig('logZ_distribution.png')
