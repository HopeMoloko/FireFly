                                                                                    #
#  Created by : Hope Moloko                                                                                   # 
#  Date : 2018                                                                                                #
#                                                                                                             #
#                                                                                                             #
###############################################################################################################

# Import needed Modules
from __future__ import absolute_import, unicode_literals, print_function
import numpy as np
from numpy import pi, cos
import pymultinest as pys
import corner
import numpy as np
import pylab as plt
from scipy.linalg import logm
import corner
import scipy.stats as stat
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
plt.style.use('classic')
import pandas as pand

import os
from configparser import  ConfigParser
from optparse import OptionParser
#%matplotlib inline

config = ConfigParser()

usage = "usage: %prog options"
parser = OptionParser(usage=usage)

parser.add_option("--config", "-c", type='string', dest='config',
        default=None, help='Name of config file')

(options, args) = parser.parse_args()
config.read(options.config)


source_list_name = config.get('execute','source_list_name')
npix = config.getint('execute','npix')                 #npix by npix image
noise_level = config.getfloat('execute','noise_level')            #sigma noise level
num_of_real_sources = config.getint('generator','num_of_real_sources')  #number of model sources
ndim = config.get('execute','ndim')
R_constant_value = config.getint('execute','R_value')  

def tau(x,y,X,Y,R,A):  #Circularly  Gaussian Shaped function
    term1 = ((x-X)**2 + (y-Y)**2)/(2*R**2)
    return A*np.exp(-term1)

x = np.arange(0, npix, 1, float)
y = x [:,np.newaxis]

source1_template = np.zeros((npix,npix))
source_list = []
for i in range(num_of_real_sources):

    X_true = npix*np.random.random()
    Y_true = npix*np.random.random()
    R_true = R_constant_value
    A_true = 10*np.random.random()+5
    
    source_list.extend([[X_true,Y_true,R_true,A_true]])

    source1_template += tau(x,y,X_true,Y_true,R_true,A_true) 

df = pand.DataFrame(source_list,columns=['X','Y','R','A'])



##################Sources Image ###################################
noise = np.random.normal(0,noise_level,source1_template.shape)
Source = source1_template  + noise


plt.figure(figsize=(5,7))
plt.imshow(Source,origin='lower',cmap='afmhot',)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()

#------Make csv file---------

if os.path.exists(source_list_name)==True:
	os.system('rm ' + source_list_name)
df.to_csv(source_list_name,sep=',')
