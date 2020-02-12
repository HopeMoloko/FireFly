#############################################################
import numpy as np
import matplotlib.pyplot as plt
import time
import copy as duplicate
import pandas as pd
import os



################################################################

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

    for s in range(n_walkers):
        for i in range(ndim):

            df[labels[m] + '_' + str(num_label)] = pd.Series(New_keep[:][:,s][:,i])

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
    
    def __init__(self,loglikelihood,Prior, ndim, n_walkers,thresh, prior_transform, nest_steps,nlive_points ,labels,samples_name, Exploration_technique,checkpoint=True):

        self.loglikelihood           = loglikelihood
        self.Prior                   = Prior
        self.ndim                    = ndim
        self.n_walkers               = n_walkers
        self.nest_steps              = nest_steps
        self.Exploration_technique   = Exploration_technique
        self.nlive_points            = nlive_points
        self.labels                  = labels
        self.posterior_samples       = None
        self.nsize                   = None
        self.prior_transform         = prior_transform
        self.posterior_sample_name   = samples_name 
        self.thresh                  = thresh


        #Storage of results
        # nest_stesp , n_walkers , num_params
        self.keep                    = np.empty((0,n_walkers,ndim))
        self.keep_logl               = []

        #Acceptance ratios
        self.nacceptance             = np.array([])

    

        loglikelihood_particles = np.array([])
        Prior_particles = []

        #particles
        particles = np.empty((0,n_walkers,ndim))  #np.zeros((nlive_points,ndim))

    
        #Draw N samples
        print('Begin Drawing Live points from prior...')
        print('No. live points : ',nlive_points)

        if checkpoint == True:
            if os.path.exists('restore/'+ self.posterior_sample_name + '-samples-restore.txt'):
                self.iteration = True
                chain_sample_restored = np.loadtxt('restore/' + self.posterior_sample_name + '-samples-restore.txt',
                                     delimiter=',')
                chain_logliklihood_restored = np.loadtxt('restore/' + self.posterior_sample_name + '-loglikelihood-samples-restore.txt'
                                    , delimiter=',')
                chain_loglike_live_restored = np.loadtxt('restore/' + self.posterior_sample_name + '-loglikelihood-live-samples-restore.txt'
                                    , delimiter=',')

                chain_live_restored = np.loadtxt('restore/' + self.posterior_sample_name + '-live-samples-restore.txt'
                                    , delimiter=',')

                chain_prior = np.loadtxt('restore/' + self.posterior_sample_name + '-prior-restore.txt'
                                    , delimiter=',')

                #reshape to chain dimensions
                
                keep_reshape = chain_sample_restored.reshape(chain_sample_restored.shape[0],int(chain_sample_restored.shape[1]/3),
                3)
                
                self.keep = keep_reshape
                self.keep_logl = chain_logliklihood_restored
                #initialize chain to continue nested sampling
                
                particles_reshape = chain_live_restored.reshape(chain_live_restored.shape[0],int(chain_live_restored.shape[1]/3),
                3)

                particles = particles_reshape
                loglikelihood_particles = chain_loglike_live_restored

                Prior_particles = chain_prior
                




            else:
                self.iteration =False
                if os.path.exists('restore'):pass
                else:
                    os.makedirs('restore')
                for s in range(nlive_points):

                    sample                    = prior_transform(self.ndim,self.n_walkers)
                    sample                    = live_point_switcher(sample,self.thresh)
                    particles = np.append(particles,[sample],axis=0)
                    loglikelihood_particles = np.append(loglikelihood_particles,[loglikelihood(sample)],axis=0)
                    Prior_particles.append(Prior(sample))

    
             
        self.particles               = particles
        self.loglikelihood_particles = loglikelihood_particles
        self.Prior                   = Prior_particles

        print('End Draw!')
    
    def nestedsampler(self):

        """ The main nested sampling algorithm"""

        print('Begin nested sampling...')
        print('No. of runs : ',self.nest_steps)
        print('No. of parameters : ',self.n_walkers*self.ndim)

        if self.iteration == True:
            i = self.keep.shape[0]-1
        else:
            i = 0
        
        
        while i< self.nest_steps:
            
            # Index of particle with worst likelihood values
            worst  = np.argmin(self.loglikelihood_particles)

            #Save the details of the worst particle
            self.keep = np.append(self.keep,[self.particles[worst]],axis=0)
            self.keep_logl = np.append(self.keep_logl,self.loglikelihood_particles[worst])


            #Copy random survivor
            copy = np.random.choice(np.concatenate(\
               (np.arange(1,worst),np.arange(worst+1,len(self.particles)))))

            assert copy != worst
            #Discard the worst particle and replace with random survivor
            #and do exploration from the copied particles
            

            particle_copy = self.particles[copy]
            logl_copy     = self.loglikelihood_particles[copy]
   

            assert logl_copy >= self.loglikelihood_particles[worst]
            #Take likelihood of worst particles as the likelihood constraint
            loglikelihood_constraint = self.loglikelihood_particles[worst]  #L*
            prior_copy = self.Prior[copy]
            
            #print(prior_copy)
            assert prior_copy == 1.0

            Update = self.Exploration_technique(particle_copy,logl_copy,loglikelihood_constraint,prior_copy,self.thresh) 

            self.loglikelihood_particles[worst] = Update['loglikelihood_new']
            self.particles[worst]               = Update['sample_new']
            self.Prior[worst]                   = Update['logp_new']
            
            #counts.append(Update['count'])
            assert self.loglikelihood_particles[worst] >= loglikelihood_constraint
            #Save acceptance ratio for each mcmc
            self.nacceptance  = np.append(self.nacceptance, Update['Acceptance_Ratio'])

            
  #          bar.update(i)
            
            
            if (i+1) %1000 == 0:

                chain0 = self.keep.reshape(len(self.keep),self.n_walkers * self.ndim)
                chain1 = self.particles.reshape(len(self.particles),self.n_walkers * self.ndim)
                #checkpoint (save current samples)
                np.savetxt('restore/' + self.posterior_sample_name + '-samples-restore.txt', 
                            chain0, delimiter=',')
                np.savetxt('restore/' + self.posterior_sample_name + '-live-samples-restore.txt', 
                            chain1, delimiter=',')
                np.savetxt('restore/' + self.posterior_sample_name + '-loglikelihood-samples-restore.txt',
                            self.keep_logl, delimiter=',')
                np.savetxt('restore/' + self.posterior_sample_name + '-loglikelihood-live-samples-restore.txt'
                                ,self.loglikelihood_particles, delimiter=',')
                np.savetxt('restore/' + self.posterior_sample_name + '-prior-restore.txt'
                                ,self.Prior, delimiter=',')
            
                #Evaluate the Prior Mass

                logX = -(np.arange(0, (i))+1.)/self.nlive_points

                # Prior width  wi = 0.5*(Xi-1 Xi+1)
                logPrior_width = np.log(0.5) - np.arange(0,i+1)/self.nlive_points

                # Calculate the Evidence ( marginal likelihood equaion 8) Z = SUM(Li*wi)

                

                logZ = logsumexp(logPrior_width + self.keep_logl) #[:][:,1]  #the current evidence from all saved samples

                logZ_remain = logsumexp(logPrior_width[-1]+self.keep_logl[-1]) #estimated contribution from the remaining volume
            

                # Importance weights (equation 10)

                logImportance_weights = np.array([ (li+wi)-logZ for li,wi in zip(self.keep_logl,logPrior_width)])  #[:][:,1] put back

                Importance_weights = np.exp(logImportance_weights)
                Importance_weights /= np.sum(Importance_weights)

                
                if np.logaddexp(logZ,logZ_remain)  - logZ < 1e-5:
                    break 

                else:
                    self.nest_steps += 3000

            i += 1
   #     bar.finish()

        self.nest_steps = i
    
        #Evaluate the Prior Mass

        # Prior width  wi = 0.5*(Xi-1 Xi+1)
        logPrior_width = np.log(0.5) - np.arange(0,self.nest_steps+1)/self.nlive_points

        # Calculate the Evidence ( marginal likelihood equaion 8) Z = SUM(Li*wi)

        logZ = logsumexp(logPrior_width + self.keep_logl) #[:][:,1]
    

        # Importance weights (equation 10)

        logImportance_weights = np.array([ (li+wi)-logZ for li,wi in zip(self.keep_logl,logPrior_width)])  #[:][:,1] put back

        Importance_weights = np.exp(logImportance_weights)
        Importance_weights /= np.sum(Importance_weights)
        #Normalize them and see what happens    


        

        #Evaluate the information
        H = int(np.exp(-np.sum(Importance_weights*np.log(Importance_weights + 1E-300))))

        self.logX  = logPrior_width
        self.logZ = logZ
        self.Importance_weights = Importance_weights
    

        print('\n'+'logZ = {logZ} '.format(logZ=self.logZ))
        print('Information = {H} nats'.format(H=H))
        print('Final Nest runs : ',self.nest_steps)

        keeper = save_results(self.labels,self.n_walkers,self.ndim, self.keep)   #edited
        keeper.to_csv('Keep-'+self.posterior_sample_name + '.csv')
        ####################### Posterior Samples ##################################
        posterior_samples = []        #np.empty((H, self.n_walkers, self.ndim))  #np.empty((H, self.keep.shape[1])) #
        k = 0
        while True:
          # Choose one of the samples
            which = np.random.randint(self.nest_steps)

          # Acceptance probability
            prob = self.Importance_weights[which]/self.Importance_weights.max()
            
            if np.random.rand() <= prob:
                posterior_samples.append(self.keep[which]) 

                k += 1

            if k >= H:
                break

        self.posterior_samples = save_results(self.labels,self.n_walkers,self.ndim, posterior_samples)   #edited
        self.posterior_samples.to_csv(self.posterior_sample_name+'.csv')
	
        print('\n'+'End nested sampling.')
	
        np.savetxt(self.posterior_sample_name+'loglike.csv',self.keep_logl,delimiter=",")
        np.savetxt(self.posterior_sample_name+'importance.csv',self.Importance_weights/self.Importance_weights.max(),delimiter=",")
    def analyze(self,fontsize,truths=None):
        """ Visualize the results from nested sampling

        Parameters

        --------------
        labels : list ['param1','param2',etc]
                Parameter labels

        truths : list
                True parameter values, default = None
                """
        #Plot the acceptance Ratio plot and save it

        
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=(10,18), nrows=3)
        
        ax1.plot(self.nacceptance,'o')
        ax1.set_ylabel('Acceptance Ratio',fontsize=fontsize)
        ax1.set_xlabel('Iterations',fontsize=fontsize)
        #plt.savefig('Acceptance_Ratio_plot.png')
        #plt.show()


        # Plot logl vs Prior mass and Posterior vs Prior ass
        
        ax2.plot(self.logX, self.keep_logl, 'bo')   #[:][:,1]
        ax2.set_ylabel('$\\loglikelihood$',fontsize=fontsize)

        ax3.plot(self.logX, self.Importance_weights/self.Importance_weights.max(), 'bo-')
        ax3.set_ylabel('Importance weights (relative)',fontsize=fontsize)
        ax3.set_xlabel('$log(X)$',fontsize=fontsize)
        
        fig.savefig(self.posterior_sample_name+'.png')



    
