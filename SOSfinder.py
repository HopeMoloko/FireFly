###############################################################################################################
#                                                                                                             #
#  Source Finding Algorithm using Nested sampling (MCMC-SAMPLER)                                              #
#  Created by : Hope Moloko                                                                                   # 
#  Date : 2019                                                                                                #
#                                                                                                             #
#                                                                                                             #
###############################################################################################################
from __future__ import absolute_import, unicode_literals, print_function

from numpy import pi, cos
import copy as Makecopy
import scipy.stats as stat
import time
import numpy as np
import pylab as plt
from scipy.linalg import logm
import corner
import scipy.stats as stat
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
plt.style.use('classic')
import pandas as pd

from ipywidgets import IntProgress
from IPython.display import display
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
img_name          =     config.get('images','img_name')

####### New folder ##########
wdir = config.get('output','workingdir')
folder_name = config.get('output','output_folder_name')

##Nest parameters
num_of_live_points = 	config.getint('nest','num_live_points')     #Main nest loop number of live points
nest = 		config.getint('nest','nest_runs')        #Number of Main nest runs
mcmc_steps = 	config.getint('nest','mcmc_steps')      #mcmc step size in the Main nest 
probability_off = 	config.getfloat('nest','probability_off')     #Probability of switching a source off


################################# Make a new folder for a run ####################################
if os.path.exists(wdir+'/'+folder_name)==True: # If folder exists delete it and make a new one.
	os.system('rm -r '+ folder_name)
    
os.system('mkdir '+ folder_name)    #Create new folder
os.chdir(folder_name)               #Change new folder 


####################Special function################
import sys


""" This function creates a progress bar"""

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    #sys.stdout.write('\r\r [%s] %s%s ...%s\r' % (bar, percents, '%', status))
    #sys.stdout.flush() # As suggested by Rom Ruben 
    
    print('[%s] %s%s ...%s' % (bar, percents, '%', status), sep='', end='\r', file=sys.stdout, flush=True)

################################# Define Functions ########################################
def Model_function(params,noise):
    '''Return the model value of a gaussian shaped function given X,Y,R, and A paramters'''
       
    x = np.arange(0, npix, 1, float)
    y = x[:,np.newaxis]
    n = noise
    
    def tau(x,y,X,Y,A,R):  #Circularly  Gaussian Shaped function
        term1 = ((x-X)**2 + (y-Y)**2)/(2*R**2)
        
        return A*np.exp(-term1)
    
    source_template_model = np.zeros((npix,npix))
    params_new = np.reshape(params,(int(len(params)/4),4))  # change shape of array to index each source. from
                                                          # n-dimension to a k-4 dimennsion array.
    for i in range(len(params_new)):
        """ Add each gaussian shaped source"""
        X,Y,A,R = params_new[i][0],params_new[i][1],params_new[i][2],params_new[i][3]
        source_template_model += tau(x,y,X,Y,A,R)
    
    return source_template_model + n

def Log_Likelihood(data=None,sigma=None,model=None):
    
    '''Return the Log-Likelihood.
    lnLikelihood = sum(f(data|model,sigma^2))
                =-(n/2)*ln(2*pi) - (n/2)*ln(sigma^2) - (1/2*sigma^2)*sum((data-model)^2)    
    Keyword arguments:
    sigma -- noise level
    model    -- Model
    data     -- The data'''
    #Unit test for shape of data amd model
    assert data.shape == model.shape
    
    
    Term2 = -0.5*((1/sigma**2)*(data-model)**2)
    
    Log_Like = np.sum(Term2)
    
    return Log_Like


def prior_transform(u=None): #Prior                                                                                
    prior_source_list = []                                                                            
                                                                                                      
    for i in range(num_of_model_sources):                                                             
        X_prior = npix*u[i]                                                             
        Y_prior = npix*u[i+1]
        A_prior = u[i+2]*25                                                             
        R_prior = R_constant_value                                                                    
                                                                   
                                                                                                      
        prior_source_list.extend([X_prior,Y_prior,A_prior,R_prior])                                     
                                                                                                      
    return prior_source_list

def log_plus(x,y):
    '''Return the addition of x + y in log-space'''
    if x>y:
        return x+np.log(1+np.exp(y-x))
    
    else:
        return y+np.log(1+np.exp(x-y))
##############################################################################################

##############Dataset class################
class Dataset:
    
    def __init__(self,data):
        
        self.type = "csv"
        self.data = data
        
    def Source(self):
        Sources = pd.read_csv(self.data)
        
        return Sources

###########################################

####################  Circular Gaussian Shaped Function ###########################
class CircularGaussian:
    """Circurlarly Gaussian-shaped onjects.
    
    Methods:
        Function(): Computes the cirular Gaussian given parameters X,Y,R and A.
        Image : Produce an image of the sources with added noise
        
        
    Data attributes: 
        Position: X,Y 
        Amplitude: A  
        Spatial extent:R
        npix: Number of pixels
        noise_level: rms units
        sources = source list"""
    
    def __init__(self,x,y,sources,npix,noise_level):
        self.x = x 
        self.y = y 
        self.npix = npix
        self.sources = sources
        self.noise_level = noise_level

        
    def tau(self,x,y,X,Y,A,R):  #Circularly  Gaussian Shaped function
        term1 = ((self.x-X)**2 + (self.y-Y)**2)/(2*R**2)
        
        return A*np.exp(-term1)
        
    
    def GenerateModel(self,):
        tau = self.tau
        source_template = np.zeros((self.npix,self.npix))
        
        for i in range(len(self.sources)):
            X_true = self.sources['X'][i]
            Y_true = self.sources['Y'][i]
            A_true = self.sources['A'][i]
            R_true = self.sources['R'][i]
            

            source_template += tau(self.x,self.y,X_true,Y_true,A_true,R_true)
            
        Noise = np.random.normal(0,self.noise_level,source_template.shape)
        data_source = source_template  + Noise
        return Noise,data_source
        
        
    def Image(self,data_source,img_name):
        ax = plt.subplot(111)

        
        im = ax.imshow(data_source,origin='lower',cmap='afmhot',)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        plt.ylabel('Arbitrary Flux')
        plt.savefig(img_name,bbox_inches='tight')
        
        
#######################

## Read the csv
dataset = Dataset(source_list_path)
Sources = dataset.Source()

'''if config.getboolean('images','save_img') == True:
    plt.savefig(config.get('images','img_name'),bbox_inches='tight')
plt.show()
'''

################
x = np.arange(0, 50, 1, float)
y = x [:,np.newaxis]

Objects = CircularGaussian(x,y,Sources,npix,noise_level)   # Create an instance
Noise,Data_source = Objects.GenerateModel()	           # Compute noise and data_source
Objects.Image(Data_source,img_name)                        #Generate image and save


###################### Generate Live points ############
#Save objects
points = np.zeros((num_of_live_points,ndim))

#log_likelihood of live objects
log_like_of_live_points = np.zeros((num_of_live_points))

for i in range(num_of_live_points):
    points[i,:] = prior_transform(u=np.random.uniform(0,1.0,size=ndim))
    model = Model_function(params=points[i,:],noise=Noise)
    log_like_of_live_points[i] = Log_Likelihood(data=Data_source,sigma=Noise,model=model)

    
################## Initialize Nest ####################################
keep = np.zeros((nest,ndim))  #store bad points
logl_sample = []     #Store Log-likelihood of samples
X_sample = []        #Store prior mass
   


logWT = []  #Store  weight =width*likelihood


logZ = -1e300     # SUM(weights)= Z Evidence
H = 0             # Information

Flat_interval = [(0,50),(0,50),(0,25),(3.7,4.1),(0,50),(0,50),(0,25),(3.7,4.1)]   #,(0,200),(0,200),(0,25)]

Acceptance = np.empty((nest))

##### Make a log file######
F = open(options.output, 'a+')
###########################

outtext = "======Begin Main Nest======= \n"
print(outtext)
print('live points:'+str(num_of_live_points)+'\n'+'nest runs:'+str(nest))
if F:
	F.write(outtext)
	F.write('live points:'+str(num_of_live_points)+'\n'+'nest runs:'+str(nest))
    
    

################################# MAIN NS LOOP ################################


## MAIN NS LOOP##

#Outer interval 
logw = np.log(1.0 - np.exp(-1.0 / num_of_live_points))

start = time.time()
for i in range(nest):
    # Draw worst object with L* from n points
    worst = np.argmin(log_like_of_live_points)
    
    #Save worst object
    keep[i,:] = points[worst,:]
    logl_sample.append(log_like_of_live_points[worst])
    
    #Save prior mass
    X_sample.append(logw)
    
    #Weight
    logwt = logw + log_like_of_live_points[worst]
    
    
    #Save weight
    logWT.append(logwt)
    
    #Update Evidence Z
    logZnew = log_plus(logZ,logwt)   #np.logaddexp(logZ,logwt)     #CHANGED THIS LINE
    
    #Update H information
    H = np.exp(logwt-logZnew)*log_like_of_live_points[worst] \
    +np.exp(logZ-logZnew)*(H+logZ)-logZnew
    
    #Update logZ
    logZ = logZnew
    #Shrink interval
    logw -= 1.0/num_of_live_points
    

    while True:#----copy a random point and do mcmc from there-----
        copy = np.random.randint(len(points))
        if (copy != worst):break
            
    points[worst,:] = Makecopy.deepcopy(points[copy,:])
    log_Likelihood_old =  log_like_of_live_points[copy]
    
    theta = points[copy,:]
    
    #UNIT TEST check for correct shape
    assert theta.shape == (ndim,)
    
   # initialize mcmc params
    
    scale = 1.0
    accept = 0
    reject = 0
    for mcmci in range(mcmc_steps):  #Evolve within current worst likelihood L>L* , draw new point under constraint
    #while True:   
        propose_step = np.random.normal(0,scale=scale,size=ndim)
        new_point = theta + propose_step
        
        #Make all Amp == R_constant
        new_point_reshape = np.reshape(new_point,(int(len(new_point)/4),4))
        new_point_reshape[:,-1] = R_constant_value
        
        new_point = new_point_reshape.ravel()
        ##############################################################################
        add = 0
        for j in range(len(theta)):
                if Flat_interval[j][0] <= new_point[j] <= Flat_interval[j][1]:
                    add += 1
                else:
                    pass

        if add == len(theta):
            Prior = 1
        else:
            Prior = 0
        
        #Calculate Log_likelihood of new point
        model = Model_function(params=new_point,noise=Noise)
        
        #Ignore divde by zero runtimewarining
        np.seterr(divide='ignore')
        log_Likelihood_new = Log_Likelihood(data=Data_source,sigma=Noise,model=model) + np.log(Prior)
       
    
        #Acceptance ratio alpha
        alpha = np.exp(log_Likelihood_new-log_Likelihood_old)
        
        
        if alpha>=1:
            points[worst,:] = new_point  #Replace worst point with new point
            log_like_of_live_points[worst] = log_Likelihood_new   #Replace the worst likelihood with new one  
            accept += 1
            break
            
        else:
            u = np.random.uniform()
            if u <= alpha :
                points[worst,:] = new_point 
                log_like_of_live_points[worst] = log_Likelihood_new
                accept += 1
                break
                
            else:
                theta = theta
                reject +=1
                
        #Changing the scale
        if accept > reject:
            scale *= np.exp(1./accept)
        if accept < reject:
            scale /= np.exp(1./reject)

    
    Acceptance_Ratio = accept/(accept+reject)
    Acceptance[i] =  Acceptance_Ratio
    
    progress(i, nest, status='Nested Sampling running')
    
                
    if i >nest*np.exp(H)/np.log(2.):
        break


    
Z = logZ
Z_err = np.sqrt((H)/num_of_live_points)
H = H        #np.exp(H)/np.log(2.)
#print("Acceptance Ratio :",Acceptance_Ratio)
print('Evidence Z = {0} +-{1} : Information H = {2} '.format(Z,Z_err,H))
#print('time:',end-start)

##############################################################################
outtext = "=====End Main Nest====== \n"
print('Evidence Z = {0} +-{1} \n Information H = {2} \n '.format(Z,Z_err,H))
print(outtext)

if F:
	F.write('Evidence Z = '+ str(Z)+' +- '+str(Z_err) + '\n Information H = '+str(H)+'\n')
	F.write(outtext)

##############################Acceptance plot######################################
plt.figure(figsize=(15,8))
plt.plot(np.arange(0,len(Acceptance)),Acceptance,'+')
plt.ylabel('Acceptance Ratio')
plt.axhline(0.2,c='blue')
########################Equally weighted posterior results #############################################

outtext = "=== Begin Equally weighting Posterior Results==== \n"
print(outtext)
if F:
	F.write(outtext)

#Normalized samples
wt = np.exp((logWT)-max(logWT))
Weights = wt/sum(wt)

#Effective sample size
effective_sample_size = int(np.exp(-np.sum(Weights*np.log(Weights+1e-300))))
S = 0
sample = np.zeros((effective_sample_size,ndim))


print('Effective Sample Size : {}'.format(effective_sample_size))



# Selecting the Effective sample
progress_i = 0
while True:
    rnd_point= np.random.randint(len(keep))
    #proba = prob_weighted[rnd_point]/max(prob_weighted)
    proba = Weights[rnd_point]/max(Weights)
    
    if np.random.rand() < proba:
        sample[S,:] = keep[rnd_point,:]
        
        S += 1
        progress(progress_i,effective_sample_size,status='Effective Sampling Running')
        progress_i += 1

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
plt.figure(figsize=(15,10))


for i in range(ndim):
    plt.subplot(4,4,i+1)
    
    plt.hist(sample[:,i],histtype="step")
    
    if i%4==0:
        plt.xlabel('X')
        
    elif i%4==1:
        plt.xlabel('Y')
        
    elif i%4==2:
        plt.xlabel('A')
    else:
        plt.xlabel('R')


if config.getboolean('images','save_hist') == True:
    plt.savefig(config.get('images','hist_name'),bbox_inches='tight')
plt.show()

if config.getboolean('data','save_weights') == True:
    df =  pd.DataFrame({'X':X_sample,'Y':Y_sample,'A':A_sample})
    df.to_csv(config.get('data','weights_name'),sep='\t')


'''X_sample_r = np.concatenate(X_sample)
Y_sample_r = np.concatenate(Y_sample)
A_sample_r = np.concatenate(A_sample)

plt.figure(figsize=(15,10))
plt.scatter(X_sample_r[np.where(A_sample_r>0)],Y_sample_r[np.where(A_sample_r>0)],alpha=0.059)
plt.xlabel('X')
plt.ylabel('Y')
if config.getboolean('images','scatter_plot') == True:
	plt.savefig(config.get('images','scatter_name'),bbox_inches='tight') 
plt.show()'''
