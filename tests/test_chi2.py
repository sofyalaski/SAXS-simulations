#!/home/slaskina/.conda/envs/ma/bin/python
import os
import sys
import pyopencl as cl
import torch
import time
import numpy as np
import pandas as pd
#sys.path.append('/home/slaskina/SAXS-simulations')

from SAXSsimulations import  Sphere, Cylinder, DensityData
from SAXSsimulations.plotting import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['PYOPENCL_CTX'] = "0"
cl.create_some_context()


## In this code for each shape run a simulation in certain size to obtain its metrics on chi2 metrics based on different uncertainty value.

def run_spheres(size):
    simulation = Sphere(size = 250, nPoints = size, volFrac = 0.01)
    simulation.place_shape(rMean = 5, rWidth = 0.1, nonoverlapping=False)
    simulation.pin_memory()
    simulation.calculate_custom_FTI( three_d = True, device = 'cuda', less_memory_use=True)
    
    simulation.reBin(size if size < 150 else 150, for_sas = True)

    simulation.drop_first_bin()
    simulation.init_sas_model()
    simulation.optimize_scaling()
    chi_error = simulation.Chi_squared_norm('ISigma')
    
    one_run  = pd.DataFrame({
        'shape': 'sphere',
        'size': size, 
        'chi2_Istd': simulation.Chi_squared_norm('IStd'),
        'chi2_ISEM': simulation.Chi_squared_norm('ISEM'),
        'chi2_ISigma': simulation.Chi_squared_norm('ISigma'),
        'chi2_IError': simulation.Chi_squared_norm('IError'),
    }, index = [0])
    one_run.to_csv('/home/slaskina/SAXS-simulations/tests/results_chi.csv', mode = 'a', header =False, index = False)



def run_hardspheres(size):
   
    simulation = Sphere(size = 250, nPoints = size, volFrac = 0.15)
    simulation.place_shape(rMean = 5, rWidth = 0.1, nonoverlapping=False)
    simulation.pin_memory()
    simulation.calculate_custom_FTI( three_d = True, device = 'cuda', less_memory_use=True)
    
    simulation.reBin(size if size < 150 else 150, for_sas = True)

    simulation.drop_first_bin()
    simulation.init_sas_model()
    simulation.optimize_scaling()
    chi_error = simulation.Chi_squared_norm('ISigma')
    
    one_run  = pd.DataFrame({
        'shape': 'hardsphere',
        'size': size, 
        'chi2_Istd': simulation.Chi_squared_norm('IStd'),
        'chi2_ISEM': simulation.Chi_squared_norm('ISEM'),
        'chi2_ISigma': simulation.Chi_squared_norm('ISigma'),
        'chi2_IError': simulation.Chi_squared_norm('IError'),
    }, index = [0])
    one_run.to_csv('/home/slaskina/SAXS-simulations/tests/results_chi.csv', mode = 'a', header =False, index = False)

def run_cylinders(size):
    simulation = Cylinder(size = 250, nPoints = size, volFrac = 0.01)
    simulation.place_shape(rMean = 5, hMean = 25,  rWidth = 0.1)
    simulation.pin_memory()
    simulation.calculate_custom_FTI( three_d = True, device = 'cuda', less_memory_use=True)
    
    simulation.reBin(size if size < 150 else 150, for_sas = True)

    simulation.init_sas_model()
    simulation.optimize_scaling()
    chi_error = simulation.Chi_squared_norm('ISigma')
    
    one_run  = pd.DataFrame({
        'shape': 'cylinder',
        'size': size, 
        'chi2_Istd': simulation.Chi_squared_norm('IStd'),
        'chi2_ISEM': simulation.Chi_squared_norm('ISEM'),
        'chi2_ISigma': simulation.Chi_squared_norm('ISigma'),
        'chi2_IError': simulation.Chi_squared_norm('IError'),
    }, index = [0])
    one_run.to_csv('/home/slaskina/SAXS-simulations/tests/results_chi.csv', mode = 'a', header =False, index = False)



if __name__ == "__main__":
    shape, size = sys.argv[1:3]
    size = int(size)
    if shape=='sphere':
        run_spheres(size)
    elif shape == 'hardsphere':
        run_hardspheres(size)
    else: 
        run_cylinders(size)