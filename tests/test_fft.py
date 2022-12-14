#!/home/slaskina/.conda/envs/ma/bin/python
import os
import sys
import pyopencl as cl
import torch
import time
import numpy as np
import pandas as pd
sys.path.append('/home/slaskina/SAXS-simulations')

from SAXSsimulations import  Sphere, Cylinder, DensityData
from SAXSsimulations.plotting import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['PYOPENCL_CTX'] = "0"
cl.create_some_context()

def timer_wrapper(func, **kwargs):
    st = time.time()
    r = func(**kwargs)
    return time.time()-st, r



def run_spheres(size, dtype):
    simulation = Sphere(size = 250, nPoints = size, volFrac = 0.01)
    simulation.place_shape(rMean = 5, rWidth = 0.1, nonoverlapping=False)
    simulation.pin_memory()
    if size>=729:
        split_time,split_mem, split_fft = 0,0,0
    else:
        split_time, (split_mem, split_fft) = timer_wrapper(simulation.calculate_custom_FTI,three_d = True, device = 'cuda', smallest_memory=True,  dtype = dtype, return_value = True)
    simple_split_time, (simple_split_mem, simple_split_fft) = timer_wrapper( simulation.calculate_custom_FTI, three_d = True, device = 'cuda', less_memory_use=True,  dtype = dtype, return_value = True)
    torch_time, (torch_mem, torch_fft)  = timer_wrapper(simulation.calculate_torch_FTI, device = 'cuda', dtype = dtype, return_value = True) 
    if torch_fft ==0:
        torch_time =0
    cpu_time,_ = timer_wrapper(simulation.calculate_custom_FTI, three_d = True, device = 'cpu')     


    try:
        error = compute_error(simulation.FTI.numpy(),simulation.FTI_torch.numpy())
    except AttributeError:
        error = np.nan
    simulation.reBin(size if size < 150 else 150, for_sas = True)

    simulation.drop_first_bin()
    simulation.init_sas_model()
    simulation.optimize_scaling()
    error_sas = compute_error(simulation.I_sas, simulation.binned_slice.I.values)
    chi_error = simulation.Chi_squared_norm('ISigma')
    
    one_run  = pd.DataFrame({
        'shape': 'sphere',
        'size': size, 
        'data type': dtype,
        'time_fftn': torch_time, 
        'time_2D': simple_split_time, 
        'time_1D': split_time, 
        'time_cpu': cpu_time,
        'memory_density_fftn': torch_mem, 
        'memory_density_2D': simple_split_mem, 
        'memory_density_1D': split_mem, 
        'memory_fft_fftn': torch_fft, 
        'memory_fft_2D': simple_split_fft, 
        'memory_fft_1D': split_fft, 
        'rel_error': error, 
        'rel_error_Sas': error_sas, 
        'chi2': chi_error, 
    }, index = [0])
    one_run.to_csv('/home/slaskina/SAXS-simulations/tests/results_fft_all.csv', mode = 'a', header =False, index = False)



def run_hardspheres(size, dtype):
   
    simulation = Sphere(size = 250, nPoints = size, volFrac = 0.15)
    simulation.place_shape(rMean = 5, rWidth = 0.1, nonoverlapping=True)
    simulation.pin_memory()
    if size>=729:
        split_time,split_mem, split_fft = 0,0,0
    else:
        split_time, (split_mem, split_fft) = timer_wrapper(simulation.calculate_custom_FTI,three_d = True, device = 'cuda', smallest_memory=True,  dtype = dtype, return_value = True)
    simple_split_time, (simple_split_mem, simple_split_fft) = timer_wrapper( simulation.calculate_custom_FTI, three_d = True, device = 'cuda', less_memory_use=True,  dtype = dtype, return_value = True)
    torch_time, (torch_mem, torch_fft)  = timer_wrapper(simulation.calculate_torch_FTI, device = 'cuda', dtype = dtype, return_value = True) 
    if torch_fft ==0:
        torch_time =0
    cpu_time,_ = timer_wrapper(simulation.calculate_custom_FTI, three_d = True, device = 'cpu')     


    try:
        error = compute_error(simulation.FTI.numpy(),simulation.FTI_torch.numpy())
    except AttributeError:
        error = np.nan
    simulation.reBin(size if size < 150 else 150, for_sas = True)

    simulation.drop_first_bin()
    simulation.init_sas_model()
    simulation.optimize_scaling()
    error_sas = compute_error(simulation.I_sas, simulation.binned_slice.I.values)
    chi_error = simulation.Chi_squared_norm('ISigma')
    
    one_run  = pd.DataFrame({
        'shape': 'hardsphere',
        'size': size, 
        'data type': dtype,
        'time_fftn': torch_time, 
        'time_2D': simple_split_time, 
        'time_1D': split_time, 
        'time_cpu': cpu_time,
        'memory_density_fftn': torch_mem, 
        'memory_density_2D': simple_split_mem, 
        'memory_density_1D': split_mem, 
        'memory_fft_fftn': torch_fft, 
        'memory_fft_2D': simple_split_fft, 
        'memory_fft_1D': split_fft, 
        'rel_error': error, 
        'rel_error_Sas': error_sas, 
        'chi2': chi_error, 
    }, index = [0])
    one_run.to_csv('/home/slaskina/SAXS-simulations/tests/results_fft_all.csv', mode = 'a', header =False, index = False)

def run_cylinders(size, dtype):
    simulation = Cylinder(size = 250, nPoints = size, volFrac = 0.01)
    simulation.place_shape(rMean = 5, hMean = 25,  rWidth = 0.1, nonoverlapping=False)
    simulation.pin_memory()
    if size>=729:
        split_time,split_mem, split_fft = 0,0,0
    else:
        split_time, (split_mem, split_fft) = timer_wrapper(simulation.calculate_custom_FTI,three_d = True, device = 'cuda', smallest_memory=True,  dtype = dtype, return_value = True)
    simple_split_time, (simple_split_mem, simple_split_fft) = timer_wrapper( simulation.calculate_custom_FTI, three_d = True, device = 'cuda', less_memory_use=True,  dtype = dtype, return_value = True)
    torch_time, (torch_mem, torch_fft)  = timer_wrapper(simulation.calculate_torch_FTI, device = 'cuda', dtype = dtype, return_value = True) 
    if torch_fft ==0:
        torch_time =0
    cpu_time,_ = timer_wrapper(simulation.calculate_custom_FTI, three_d = True, device = 'cpu')     


    try:
        error = compute_error(simulation.FTI.numpy(),simulation.FTI_torch.numpy())
    except AttributeError:
        error = np.nan
    simulation.reBin(size if size < 150 else 150)


    simulation.init_sas_model()
    simulation.optimize_scaling()
    error_sas = compute_error(simulation.I_sas, simulation.binned_slice.I.values.reshape(simulation.nBins, simulation.nBins))
    chi_error = simulation.Chi_squared_norm('ISigma')
    
    one_run  = pd.DataFrame({
        'shape': 'cylinder',
        'size': size, 
        'data type': dtype,
        'time_fftn': torch_time, 
        'time_2D': simple_split_time, 
        'time_1D': split_time, 
        'time_cpu': cpu_time,
        'memory_density_fftn': torch_mem, 
        'memory_density_2D': simple_split_mem, 
        'memory_density_1D': split_mem, 
        'memory_fft_fftn': torch_fft, 
        'memory_fft_2D': simple_split_fft, 
        'memory_fft_1D': split_fft, 
        'rel_error': error, 
        'rel_error_Sas': error_sas, 
        'chi2': chi_error, 
    }, index = [0])
    one_run.to_csv('/home/slaskina/SAXS-simulations/tests/results_fft_all.csv', mode = 'a', header =False, index = False)



if __name__ == "__main__":
    shape, size, dtype = sys.argv[1:4]
    size = int(size)
    if dtype == '32':
        dtype = torch.complex64
    else: 
        dtype = torch.complex128
    if shape=='sphere':
        run_spheres(size, dtype)
    elif shape == 'hardsphere':
        run_hardspheres(size, dtype)
    else: 
        run_cylinders(size, dtype)