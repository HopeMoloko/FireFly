from __future__ import absolute_import, unicode_literals, print_function
import numpy
from numpy import pi, cos
from pymultinest import run,solve
import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
import os
if not os.path.exists("chains"): os.mkdir("chains")

num_of_model_sources = 2
npix = 200
R_constant_value = 4
noise_level = 1


Sources = pd.read_csv('source_list2_new.csv')
def tau(x,y,X,Y,R,A):  #Circularly  Gaussian Shaped function
    term1 = ((x-X)**2 + (y-Y)**2)/(2*R**2)
    return A*np.exp(-term1)

########################### Source Image ###############################################
x = np.arange(0, npix, 1, float)
y = x [:,np.newaxis]

source1_template = np.zeros((npix,npix))
for i in range(len(Sources)):

    X_true = Sources['X'][i]
    Y_true = Sources['Y'][i]
    R_true = Sources['R'][i]
    A_true = Sources['A'][i]

    source1_template += tau(x,y,X_true,Y_true,R_true,A_true)



# ADD noise level

noise = 	np.random.normal(0,noise_level,source1_template.shape)
Source = 	source1_template  + noise

plt.figure(figsize=(5,7))
plt.imshow(Source,origin='lower',cmap='afmhot',)
plt.xlabel('X Position')
plt.ylabel('Y Position')


def Model(x,y,Xm,Ym,Rm,Am):  #Model that describes each source
    x = np.arange(0, npix, 1, float)
    y = x[:,np.newaxis]

    source_template = np.zeros((npix,npix))

    for i in range(len(Xm)):

        source_template += tau(x,y,Xm[i],Ym[i],Rm[i],Am[i])
    return source_template

def logLike(cube): #Likelihood function
    cubes = cube.tolist()
    Xm = []
    Ym = []
    Rm = []
    Am = []
    for i in range(num_of_model_sources):
        Xm.append(cubes[0])
        Ym.append(cubes[1])
        Rm.append(cubes[2])
        Am.append(cubes[3])

        cubes.pop(0)
        cubes.pop(0)
        cubes.pop(0)
        cubes.pop(0)


    data = 	Source
    mu = 	Model(x,y,Xm,Ym,Rm,Am)
    sigma = 	noise_level
    term1 = 	-len(data)*np.log(2*np.pi)/2
    term2 = 	-(len(data)/2)*np.log(sigma**2)
    term3 = 	-np.sum((data-mu)**2)/2*(sigma**2)

    LogL = 	term1 + term2 + term3

    return LogL

def mypriors(cube):  # Prior
    prior_source_list = []

    for i in range(num_of_model_sources):
        X_prior = npix*np.random.random()
        Y_prior = npix*np.random.random()
        R_prior = R_constant_value
        A_prior = 10*np.random.random()+5

        prior_source_list.extend([X_prior,Y_prior,R_prior,A_prior])

    return np.array(prior_source_list)

# number of dimensions our problem has
parameters = ["x", "y" , "R" , "A","x1","y1","r1","a1"]
n_params = len(parameters)
# name of the output files
prefix = "chains/double-"

# run MultiNest
result = solve(LogLikelihood=logLike, Prior=mypriors, 
            n_dims=n_params, outputfiles_basename=prefix, verbose=True)

#print('---')
#print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
#print('---')
#print('parameter values:')
#for name, col in zip(parameters, result['samples'].transpose()):
#	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

# make marginal plots by running:
# $ python multinest_marginals.py chains/3-
# For that, we need to store the parameter names:
#import json
#with open('%sparams.json' % prefix, 'w') as f:
 #   json.dump(parameters, f, indent=2)
