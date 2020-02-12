import numpy as np
import Metropolis_Hasting_v2 as MH3
import copy
import pandas as pd


def prior_transform_1(pixel_size, ndim, n_walkers):
    """Draw a sample from a uniform prior distribution for each parameter
    
    Returns : array
            A sample of each paramater dran from the prior.
    """

    sample = []

    for i in range(n_walkers):
        u = np.random.uniform(0, 1, size=ndim)
        A = u[0] * 5 + 5
        X = u[1] * pixel_size
        Y = u[2] * pixel_size

        sample.append([A, X, Y])

    sample = np.array(sample)
    return sample


class WALKERS:
    def __init__(self, n_walkers, pixel_size, params):

        self.n_walkers = n_walkers
        self.params = params
        self.pixel_size = pixel_size

        self.source_model = np.zeros((self.pixel_size, self.pixel_size))

    def model_equation(self, x, y, params):
        """
        Circular gassian shaped function

        Parameters
        ----------------
        params : array
                A, X, Y

        x     : array 
                (row vector)

        y     : array
                (column vector)
            
        ----------------
        Output: (pixel X pixel array)
        """

        A, X, Y = params
        R = 2.0
        tau = A * np.exp(-((x - X) ** 2 + (y - Y) ** 2) / (2 * R ** 2))

        return tau

    # still editing this function below
    def walkers(self):
        """
        Generate n number of walkers
        """

        ###----Pixel dimensions

        x = np.arange(0, self.pixel_size, 1, float) #row vector
        y = x[:, np.newaxis]                        #column vector

        for i in range(self.n_walkers):
            params = self.params[i]
            self.source_model += self.model_equation(x, y, params)


class EXPLORER:
    def __init__(
        self,
        logLikelihood,
        Prior,
        stepsize,
        mcmc_steps,
        stepper,
        ndim,
        n_walkers,
        thresh,
        pixel_size
    ):

        self.stepsize = stepsize
        self.mcmc_steps = mcmc_steps
        self.logLikelihood = logLikelihood
        self.Prior = Prior
        self.stepper = stepper
        self.ndim = ndim
        self.n_walkers = n_walkers
        self.thresh = thresh
        self.pixel_size = pixel_size

    # -------------------------------------------------------------------------------------------

    def switch_explorer(
        self, theta_out, logl_theta, loglikelihood_constraint, prior, thresh
    ):
        """ Exploration technique used for generating a new sample in nested sampling
        the stepsize is only changed after a full mcmc run and not during the mcmc.
        
        Parameters
        ---------------
        theta : array
                Sample to start mcmc from
                
        -------------
        Returns: 
            
        chain_particle : array
                        mcmc sample chain
        
        chain_loglike :array
                    loglikelihood value of each sample
                    
        chain_prior   : array
                        prior value of each sample
        
        Acceptance_R  : scalar
                        Acceptance ratio of the mcmc chain"""

        kwargs = {"pixel_size": self.pixel_size,"ndim": self.ndim, "n_walkers": self.n_walkers}

        loglikelihood_new = logl_theta
        theta_new = copy.deepcopy(theta_out)
        logp_new = prior
        Acceptance_Ratio = 0.0

        # for iid in range(len(theta_out)):
        
        count = 0
        for index, source in enumerate(theta_out):

            source_num = index  # iid
            selected_source = copy.deepcopy(
                source
            ) 
            flux_old = selected_source[0]
            # position = selected_source[1:]

            random_state = np.random.uniform()

            if random_state >= self.thresh:
                state_new = 0.0  # Off state

            else:
                state_new = 1.0  # On state

            # Conditions
            # ---------------1--------------------------------------------
            if flux_old == 0.0 and state_new == 0.0:  # off to off
                # random position and Flux_new = 0
                flux_new = 0.0
                position = prior_transform_1(**kwargs)[0][
                    1:
                ]  # directly sample a random position from the prior.

                selected_source[0] = flux_new
                selected_source[1:] = position

                # ---------------------------------
                theta_trail = copy.deepcopy(theta_out)
                theta_trail[source_num] = selected_source
                # -----------------------------------
                assert np.array_equal(theta_trail, theta_out) == False

                # evaluate the loglikelihood of the new sample
                loglikelihood_trail = self.logLikelihood(theta_trail)
                logp_trail = self.Prior(theta_trail)

                if loglikelihood_trail > loglikelihood_constraint and logp_trail == 1.0:
                    theta_new = copy.deepcopy(theta_trail)  # deepcopy
                    loglikelihood_new = loglikelihood_trail
                    logp_new = logp_trail
                    theta_out = theta_new
                    loglikelihood_constraint = loglikelihood_new
                    Acceptance_Ratio = 1.0
                    count += 1


            # ------------------------2-------------------------
            if flux_old != 0.0 and state_new == 0.0:  # on to off
                # random position and flux_new = .0
                flux_new = 0.0
                position = prior_transform_1(**kwargs)[0][
                    1:
                ]  # directly sample a random position from the prior.

                selected_source[0] = flux_new
                selected_source[1:] = position

                theta_trail = copy.deepcopy(theta_out)
                theta_trail[source_num] = copy.deepcopy(selected_source)

                assert np.array_equal(theta_trail, theta_out) == False
                # evaluate the loglikelihood of the new sample
                loglikelihood_trail = self.logLikelihood(theta_trail)
                logp_trail = self.Prior(theta_trail)

                if loglikelihood_trail > loglikelihood_constraint and logp_trail == 1.0:
                    theta_new = copy.deepcopy(theta_trail)
                    theta_out = theta_new
                    loglikelihood_new = loglikelihood_trail
                    loglikelihood_constraint = loglikelihood_new
                    logp_new = logp_trail
                    Acceptance_Ratio = 1.0
                    count += 1


            # -----------------------3-----------------------
            if flux_old == 0.0 and state_new == 1.0:  # off to on
                # random position and sample flux_new directly from prior.
                flux_new = prior_transform_1(**kwargs)[0][0]
                position = prior_transform_1(**kwargs)[0][1:]

                selected_source[0] = flux_new
                selected_source[1:] = position

                theta_trail = copy.deepcopy(theta_out)
                theta_trail[source_num] = selected_source

                assert np.array_equal(theta_trail, theta_out) == False
                # evaluate the loglikelihood of the new sample
                loglikelihood_trail = self.logLikelihood(theta_trail)
                logp_trail = self.Prior(theta_trail)

                if loglikelihood_trail > loglikelihood_constraint and logp_trail == 1.0:
                    theta_new = copy.deepcopy(theta_trail)
                    loglikelihood_new = loglikelihood_trail
                    logp_new = logp_trail
                    theta_out = theta_new
                    loglikelihood_constraint = loglikelihood_new
                    Acceptance_Ratio = 1.0

            # -----------------------------4---------------------------------
            if flux_old != 0.0 and state_new == 1.0:  # on to on
                # Continue with random walk in position and flux.
                # Continue with Random walk

                chain_particle, chain_loglike, chain_prior, Acceptance_R = MH3.MH_mcmc(
                    self.logLikelihood,
                    self.Prior,
                    theta_out,
                    self.mcmc_steps,
                    source_num,
                    self.stepsize
                )

                # trail theta from mcmc
                theta_trail = copy.deepcopy(chain_particle[np.argmax(chain_loglike)])

                # evaluate the loglikelihood of the new sample
                loglikelihood_trail = self.logLikelihood(theta_trail)
                logp_trail = self.Prior(theta_trail)


                if loglikelihood_trail > loglikelihood_constraint and logp_trail == 1.0:
                    theta_new = copy.deepcopy(theta_trail)  # update new theta
                    logp_new = logp_trail  # update new logp
                    theta_out = (
                        theta_new
                    )  # update old theta used for checking switch on-off in the loop
                    loglikelihood_new = (
                        loglikelihood_trail
                    )  # update new logliklelihood value
                    loglikelihood_constraint = (
                        loglikelihood_new
                    )  # update likelihood constraint to get the best sample
                    Acceptance_Ratio = Acceptance_R  # update acceptance ratio

        return {
            "sample_new": theta_new,
            "loglikelihood_new": loglikelihood_new,
            "logp_new": logp_new,
            "Acceptance_Ratio": Acceptance_Ratio,
            "count" : count
        }
