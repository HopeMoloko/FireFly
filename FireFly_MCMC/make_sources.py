from configparser import ConfigParser
from optparse import OptionParser
import numpy as np
import pandas as pd

import sys
import os

from configparser import ConfigParser
from optparse import OptionParser

import sys
import os

#%matplotlib inline
config = ConfigParser()

usage = "usage: %prog options"
parser = OptionParser(usage=usage)

parser.add_option(
    "--config",
    "-c",
    type="string",
    dest="config",
    default=None,
    help="Name of config file",
)
parser.add_option(
    "--output",
    "-o",
    type="string",
    dest="output",
    default="sos.log",
    help="File for output messages. [name.log]",
)

(options, args) = parser.parse_args()

config.read(options.config)  # Read from config file optionsS


##  parameters


simulated_image = config.get("execute", "simulated_image_name")
samples_name = config.get("execute", "samples_name")
stepper = config.getint("execute", "stepper")
true_sources_name = config.get("execute", "true_sources_name")
pixel_size = config.getint("execute", "pixel_size")  # npix by npix image
n_walkers = config.getint("execute", "n_walkers")
noise_level = config.getfloat("execute", "noise_level")  # sigma noise level
num_of_real_sources = config.getint(
    "execute", "num_of_real_sources"
)  # number of model sources
ndim = config.getint("execute", "ndim")  # number of dimensions
rand_seed = config.getint("nest", "rand_seed")


# Set fixed pameters. A ,X ,Y ,R
def make_source(rand_seed,num_of_real_sources,pixel_size,true_sources_name):
    """
    Generate sources and save a csv file

    Parameters
    -----------------------

    rand_seed : int
                Choose a random seed
    
    num_real_sources : int
                    number of sources to generated

    pixel_size : int
                The pixel size of your simulated simulated image

    true_sources_name : str
                        File name of the csv file to save to_csv

    ---------------------

    Output: CSV of the generated sources 
    """
    np.random.seed(rand_seed)
    source_list = []

    A_list = []
    X_list = []
    Y_list = []

    for i in range(num_of_real_sources):

        A_true = 5 * np.random.random_sample() + 5
        X_true = pixel_size * np.random.random()
        Y_true = pixel_size * np.random.random()

        source_list.append([A_true, X_true, Y_true])
        A_list.append(A_true)
        X_list.append(X_true)
        Y_list.append(Y_true)

    

    sources = pd.DataFrame(columns=["A_true", "X_true", "Y_true"])
    sources["A_true"] = A_list
    sources["X_true"] = X_list
    sources["Y_true"] = Y_list

    sources.to_csv(true_sources_name)    #Save .csv of the generated sources

    

# Run the source generator
make_source(rand_seed,num_of_real_sources,pixel_size,true_sources_name)
