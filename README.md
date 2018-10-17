
# Masters_Project

This is all the work related to my project.
=======
# mcmc-tutorial
This is a python tutorial for Bayesian inferences using MCMC. It includes concepts of reject sampling, markov chain stationary distribution. The tutorial is based on PyMC package.

Edited: 
Some equations don't display properly in some browsers. Please view at https://nbviewer.jupyter.org/github/yoyolin/mcmc-tutorial/blob/master/MCMC_for_Bayesian_Inference.ipynb 


# To Run the updated SourceFinder:

!! Download : File1.txt , SourceGenerator.py and SourceFinder.py

1: Edit File1.txt to change parameters
> vim File1.txt

2: Run the SourceGenerator.py first to create a soure list:
  > python3 SourceGenerator.py -c File1.txt 

3: Run the SourceFinder.py
 > python3 SourceFinder.py -c File1.txt -o name.log
