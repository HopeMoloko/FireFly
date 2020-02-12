# FireFly - MCMC

This is the FireFly MCMC implementation.

# Installation

Git clone the repo.`git clone [repo]`

## Running
In the same cloned repo. 
 - Create a restore folder `mkdir restore`  which will be used to checkpoint the mcmc runs.
 - Firstly edit `Config_file0.txt` file to your desired parameters.
 - run `python3 make_source.py -c Config_file0.txt` to create a csv file of the selected number of real sources.
 - then run, `python3 Source_Finder_MCMC.py -c Config_file0.txt >> [filename].txt`
