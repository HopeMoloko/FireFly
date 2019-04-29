#############################################################
import numpy as np
import matplotlib.pyplot as plt
import corner
import time
import copy as duplicate
import Functions as F
import pandas as pd
import progressbar

from mininest import nested_sampling

#test numba
from numba import jit , prange
###############################################################
Data  = pd.read_csv('example_cubic_data.txt', delimiter = ' ', names = ['x', 'y', 'err'])

x_values = Data.x
y_values = Data.y
erorr = Data.err

args_loglike = [y_values, erorr, x_values]
################################################################

n = 200
max_iter = 5000

class Object:
    def __init__(self):
        self.u      = None      # Uniform-prior controlling parameter for x
        self.params = None      # [a,b,c,d]
        self.logL   = None     # logLikelihood = ln Prob(data | position)
        self.logWT  = None


def logLike(theta):
        return F.logLikelihood_cubic(theta,args_loglike)


def sample_from_prior():
    """
    """
    obj  = Object()
    obj.u  = np.random.random(size=4)
    a = obj.u[0]*20
    b = obj.u[1]*20
    c = obj.u[2]*10
    d = obj.u[3]*10
    params = np.array([a,b,c,d])
    obj.params = params
    obj.logL   = logLike(obj.params)

    return obj


def explore(   # Evolve object within likelihood constraint
    Obj,       # Object being evolved
    logLstar): # Likelihood constraint L > Lstar

    ret = Object()
    ret.__dict__ = Obj.__dict__.copy()
    step = 0.1;   # Initial guess suitable step-size in (0,1)
    accept = 0;   # # MCMC acceptances
    reject = 0;   # # MCMC rejections
    Try = Object();          # Trial object

    for m in range(30):  # pre-judged number of steps

        # Trial object
        Try.u = ret.u + step * (2.*np.random.uniform() - 1.);  # |move| < step
        Try.u -= np.floor(Try.u);      # wraparound to stay within (0,1)
        Try.params = np.array([Try.u[0]*20,Try.u[1]*20,Try.u[2]*10,Try.u[3]*10]);  # mapping

        Try.logL = logLike(Try.params);  # trial likelihood value

        # Accept if and only if within hard likelihood constraint
        if Try.logL > logLstar:
            ret.__dict__ = Try.__dict__.copy()
            accept+=1
        else:
            reject+=1

        # Refine step-size to let acceptance ratio converge around 50%
        if( accept > reject ):   step *= np.exp(1.0 / accept);
        if( accept < reject ):   step /= np.exp(1.0 / reject);
    return ret

def process_res(results):

    ni = results['num_iterations']
    samples = results['samples']
    logZ = results['logZ']
    logZ_sdev = results['logZ_sdev']
    H = results['info_nats']
    H_sdev = results['info_sdev']


    print("# iterates: %i"%ni)
    print("Evidence: ln(Z) = %g +- %g"%(logZ,logZ_sdev))
    print("Information: H  = %g nats = %g bits"%(H,H/np.log(2.0)))

    Samples = np.empty((ni,4))
    Importance_w = np.empty((ni))
    for i in range(ni):
        Importance_w[i] = np.exp(samples[i].logWt - logZ);
        Samples[i] = samples[i].params

    H_new = int(np.exp(-np.sum(Importance_w*np.log(Importance_w + 1E-300))))
    posterior_samples = np.empty((H_new,4))
    k = 0
    print("H_new:",H_new)
    while True:
        # Choose one of the samples
        which = np.random.randint(ni)

        # Acceptance probability
        prob = Importance_w[which]/Importance_w.max()

        if np.random.rand() <= prob:
            posterior_samples[k,:] = Samples[which,:]

            k += 1

        if k >= H_new:
            break
        

    fig = corner.corner(posterior_samples,labels=['a','b','c','d'],show_titles=True)
    fig.savefig('SKillposter.png')
    plt.show()
if __name__ == "__main__":

    results = nested_sampling(n, max_iter, sample_from_prior,explore)
    process_res(results)


    