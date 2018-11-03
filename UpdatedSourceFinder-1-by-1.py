###############################################################################################################
#                                                                                                             #
#  Source Finding Algorithm using Nested sampling (MCMC-SAMPLER)                                              #
#  Created by : Hope Moloko                                                                                   # 
#  Date : 2018                                                                                                #
#                                                                                                             #
#                                                                                                             #
###############################################################################################################
from __future__ import absolute_import, unicode_literals, print_function
import numpy as np
from numpy import pi, cos
#import pymultinest as pys
#import corner
import copy as COPY
import numpy as np
import pylab as plt
from scipy.linalg import logm
#import corner
import scipy.stats as stat
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
plt.style.use('classic')
import pandas as pd


from configparser import  ConfigParser
from optparse import OptionParser

import sys
import os
#%matplotlib inline
config = ConfigParser()

usage = "usage: %prog options"
parser = OptionParser(usage=usage)

parser.add_option("--config", "-c", type='string', dest='config',
        default=None, help='Name of config file')
parser.add_option("--output", "-o", type='string', dest='output', 
default='sos.log', help='File for output messages. [name.log]') 
    
(options, args) = parser.parse_args() 
    
config.read(options.config)   # Read from config file options


##  parameters

source_list_path = 	config.get('execute','source_list_name')
npix = 			config.getint('execute','npix')                 #npix by npix image
noise_level = 		config.getfloat('execute','noise_level')            #sigma noise level
num_of_model_sources = 	config.getint('execute','num_of_model_sources')  #number of model sources
ndim = 			config.getint('execute','ndim')                   #number of dimensions
R_constant_value = 	config.getint('execute','R_value')

####### New folder ##########
wdir = config.get('output','workingdir')
folder_name = config.get('output','output_folder_name')

##Nest parameters
num_live_points = 	config.getint('nest','num_live_points')     #Main nest loop number of live points
nest_runs = 		config.getint('nest','nest_runs')        #Number of Main nest runs
mcmc_step_size = 	config.getfloat('nest','mcmc_step_size')      #mcmc step size in the Main nest 
probability_off = 	config.getfloat('nest','probability_off')     #Probability of switching a source off

Sources = 		pd.read_csv(source_list_path)


################################# Make a new folder for a run ####################################
if os.path.exists(wdir+'/'+folder_name)==True: # If folder exists delete it and make a new one.
	os.system('rm -r '+ folder_name)
    
os.system('mkdir '+ folder_name)    #Create new folder
os.chdir(folder_name)               #Change new folder 

#################### Circularly Gaussian shaped  function ##################
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


if config.getboolean('images','save_img') == True:
    plt.savefig(config.get('images','img_name'),bbox_inches='tight')
plt.show()


####################### Model, Likelihood and prior Equations ##################################

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

def mypriors():  # Prior
    prior_source_list = []
    
    for i in range(num_of_model_sources):
        X_prior = npix*np.random.random()
        Y_prior = npix*np.random.random()
        R_prior = R_constant_value
        A_prior = 10*np.random.random()+5
        
        prior_source_list.extend([X_prior,Y_prior,R_prior,A_prior]) 
    
    return prior_source_list

def plus(x,y):
    if x>y:
        return x+np.log(1+np.exp(y-x))
    
    else:
        return y+np.log(1+np.exp(x-y))
    
###################Sample N pionts from prior##########################

objects = np.zeros((num_live_points,num_of_model_sources*ndim))
l_objects = np.zeros((num_live_points))
for i in range(num_live_points):
    
    objects[i,:] = mypriors()
    l_objects[i] = logLike(cube = objects[i])

    
################## Initialize Nest ####################################
keep = np.zeros((nest_runs,num_of_model_sources*ndim))
logl_sample = []     #Store Log-likelihood of samples
logX_sample = []        #Store prior mass
   

logWT = []  #Store  weight =width*likelihood


logZ = -np.exp(300)     # SUM(weights)= Z Evidence
H = 0                  # Intitialize H(Information)


################################# MAIN NS LOOP ################################

#Outer interval 
logw = np.log(1.0 - np.exp(-1.0 / num_live_points))


sigma = mcmc_step_size*np.random.random()+1.0  #proposal stepsize for mcmc

##### Make a log file######
F = open(options.output, 'a+')
###########################

outtext = "======Begin Main Nest======= \n"
print(outtext)
print('live_points:'+str(num_live_points)+'\n'+'nest_runs:'+str(nest_runs)+'\n'+'probability_off:'+str(probability_off))
if F:
	F.write(outtext)
	F.write('live_points:'+str(num_live_points)+'\n'+'nest_runs:'+str(nest_runs)+'\n'+'probability_off:'+str(probability_off)+'\n \n')
    
    
#-----------------------------Begin Nest ------------------
for i in range(nest_runs):
    # Draw worst object with L* from n points
    worst = 	np.argmin(l_objects)
    
    #Save worst opbject
    keep[i,:] = objects[worst,:]
    logl_sample.append(l_objects[worst])
    
    #Save prior mass
    logX_sample.append(logw)
    
    #Weight
    logwt = 	logw + l_objects[worst]
    
    
    #Save weight
    logWT.append(logwt)
    
    #Update Evidence Z
    logZnew = 	plus(logZ,logwt)  
    
    #Update H information
    H = 	np.exp(logwt-logZnew)*l_objects[worst] \
    		+np.exp(logZ-logZnew)*(H+logZ)-logZnew
    
    #Update logZ
    logZ = 	logZnew
    #Shrink interval
    logw -= 1.0/num_live_points
   

    while True:#----copy a random point and replace worst, then do mcmc from there-----
        copy = np.random.randint(len(objects))
        if (copy != worst):break
            
    objects[worst,:] = COPY.deepcopy(objects[copy,:])
    theta = objects[copy,:]
    Likelihood_thresh = l_objects[copy]
    ####################################################################################################################
    increment = 0
    Probability_off = probability_off    #the probability of switching off a model source
    check_num = 0
    
    for num in range(num_of_model_sources):   # iterative over each model source with ndim
        
        random_value = np.random.uniform()       #uniform random value
        
    
        if (Probability_off > random_value):  #Switch off the model source
            new_x = np.random.random()      #assign the x position a rnd value
            new_y = np.random.random()      #assign the y position a rnd value
            new_r = R_constant_value
            A = 0                              #switch off the amplitude
            
            theta[increment:increment+ndim] = [new_x,new_y,new_r,A]
            check_num += 1
            
    
        else:  #Do an mcmc
            
            while True:
                proposal = [np.random.normal(0,sigma),np.random.normal(0,sigma),0,np.random.normal(0,sigma)]
                theta[increment:increment+ndim] += proposal
         
            
                new_point = theta
        
        
                Likelihood_new = logLike(new_point)
        
        
                alpha =  np.exp(-(np.exp(Likelihood_new)-np.exp(Likelihood_thresh))/2)            
        
                if alpha >= 1:
                    objects[worst,:] = new_point  #Replace worst point with new point
                    l_objects[worst] = Likelihood_new   #Replace the worst likelihood with new one  
                    break
                else:
                    u = np.random.uniform()
                    if u <= alpha :
                        objects[worst,:] = new_point 
                        l_objects[worst] = Likelihood_new
                        break
                    else:
                        theta[increment:increment+ndim] = theta[increment:increment+ndim]
        increment += ndim
                
    if i >nest_runs*np.exp(H)/np.log(2.):
        break       

Z = logZ
Z_err = np.sqrt((H)/num_live_points)
H = H        #np.exp(H)/np.log(2.)

outtext = "=====End Main Nest====== \n"
print('Evidence Z = {0} +-{1} \n Information H = {2} \n '.format(Z,Z_err,H))
print(outtext)

if F:
	F.write('Evidence Z = '+ str(Z)+' +- '+str(Z_err) + '\n Information H = '+str(H)+'\n')
	F.write(outtext)
########################Equally weighted posterior results #############################################

outtext = "=== Begin Equally weighting Posterior Results==== \n"
print(outtext)
if F:
	F.write(outtext)

prob_weighted = [(logX_sample[i]+logl_sample[i])/Z for i in range(nest_runs)]

prob = prob_weighted/sum(prob_weighted)

#Effective sample size
effective_sample_size = int(np.exp(-np.sum(prob*np.log(prob))))

S = 0
sample = np.zeros((effective_sample_size,num_of_model_sources*ndim))

#Selecting the Effective sample

while True:
    rnd_point = np.random.randint(len(keep))
    proba = prob_weighted[rnd_point]/max(prob_weighted)
    
    if np.random.rand() < proba:
        sample[S,:] = keep[rnd_point,:]
        S += 1
    if S >= effective_sample_size:
        break

print('Effective Sample Size : {}'.format(effective_sample_size))

outtext = "====End weighting Posterior Results \n"
print(outtext)
if F:
	F.write('Effective Sample Size : '+str(effective_sample_size))

F.close()
X_sample = []
Y_sample = []
A_sample = []


####################################### Visualize params ######################################################
plt.figure(figsize=(15,25))

for i in range(num_of_model_sources*ndim):
    plt.subplot(num_of_model_sources,ndim,i+1)
    
    plt.hist(sample[:,i][sample[:,i]>0],bins=20)
    
    if i%4==0:
        plt.xlabel('X')
        plt.axvline(107,c='r')
        #plt.axvline(81,c='g')
        #plt.axvline(140.2,c='c')
        X_sample.append(sample[:,i])
        
    elif i%4==1:
        plt.xlabel('Y')
        plt.axvline(90,c='r')
        #plt.axvline(111,c='g')
        #plt.axvline(137.2,c='c')
        Y_sample.append(sample[:,i])
        
    elif i%4==2:
        plt.xlabel('R')
    else:
        plt.xlabel('A')
        plt.axvline(8,c='r')
        #plt.axvline(13,c='g')
        #plt.axvline(10.3,c='c')
        A_sample.append(sample[:,i])
        

    #plt.legend(loc='best')
plt.tight_layout()

if config.getboolean('images','save_hist') == True:
    plt.savefig(config.get('images','hist_name'),bbox_inches='tight')
plt.show()

if config.getboolean('data','save_weights') == True:
    df =  pd.DataFrame({'X':X_sample,'Y':Y_sample,'A':A_sample})
    df.to_csv(config.get('data','weights_name'),sep='\t')


X_sample_r = np.concatenate(X_sample)
Y_sample_r = np.concatenate(Y_sample)
A_sample_r = np.concatenate(A_sample)

plt.figure(figsize=(15,10))
plt.scatter(X_sample_r[np.where(A_sample_r>0)],Y_sample_r[np.where(A_sample_r>0)],alpha=0.08)
plt.xlabel('X')
plt.ylabel('Y')
if config.getboolean('images','scatter_plot') == True:
	plt.savefig(config.get('images','scatter_name'),bbox_inches='tight') 
plt.show()
