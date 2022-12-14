#!/home/slaskina/.conda/envs/ma/bin/python
import os
import pyopencl as cl
import torch
import time
import numpy as np
import pandas as pd
from SAXSsimulations import  Sphere, Cylinder, DensityData
from SAXSsimulations.utils import compute_error
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
cl.create_some_context()

def timer_wrapper(func, **kwargs):
    st = time.time()
    r = func(**kwargs)
    return time.time()-st, r

def run_spheres(sim_size):
    for i in range(10):
        for t, size in enumerate (sim_size):
            simulation = Sphere(size = 10, nPoints = size, volFrac = 0.01)
            simulation.place_shape(rMean = 0.05, rWidth = 0.1, nonoverlapping=False)
            simulation.pin_memory()
            torch_32_time, (torch_mem_32, torch_fft_32)  = timer_wrapper(simulation.calculate_torch_FTI, device = 'cuda') 
            if torch_fft_32 ==0:
                torch_32_time =0
            simple_split_32_time, (simple_split_mem_32, simple_split_fft_32) = timer_wrapper( simulation.calculate_custom_FTI, three_d = True, device = 'cuda', less_memory_use=True)
            cpu_time_32,(_,_) = timer_wrapper(simulation.calculate_custom_FTI, three_d = True, device = 'cpu')
  
            if size>=729:
                split_32_time,split_mem_32, split_fft_32 = 0,0,0
            else:
                split_32_time, (split_mem_32, split_fft_32) = timer_wrapper(simulation.calculate_custom_FTI,three_d = True, device = 'cuda', smallest_memory=True)


            try:
                error_32 = compute_error(simulation.FTI.numpy(),simulation.FTI_torch.numpy())
            except AttributeError:
                error_32 = np.nan
            simulation.reBin(size//3, for_sas = True)

            simulation.drop_first_bin()
            simulation.init_sas_model()
            print(simulation.Chi_squared_norm('ISigma'))
            simulation.optimize_scaling()
            error_32_sas = compute_error(simulation.I_sas, simulation.binned_slice.I.values)
            chi_error_32 = simulation.Chi_squared_norm('ISigma')
            print(simulation.Chi_squared_norm('ISigma'))
            
            simple_split_64_time, (simple_split_mem_64, simple_split_fft_64) = timer_wrapper(simulation.calculate_custom_FTI, three_d = True, device = 'cuda', less_memory_use=True, dtype = torch.complex128)
            torch_64_time, (torch_mem_64, torch_fft_64)  = timer_wrapper(simulation.calculate_torch_FTI, device = 'cuda', dtype = torch.complex128) 
            if torch_fft_64 ==0:
                torch_64_time =0
            cpu_time_64,(_,_) = timer_wrapper(simulation.calculate_custom_FTI, three_d = True, device = 'cpu', dtype = torch.complex128)
            if size>=729:
                split_64_time,split_mem_64, split_fft_64 = 0,0,0
            else:
                split_64_time, (split_mem_64, split_fft_64) = timer_wrapper(simulation.calculate_custom_FTI,three_d = True, device = 'cuda', smallest_memory=True, dtype = torch.complex128)
            
            try:
                error_64 = compute_error(simulation.FTI.numpy(),simulation.FTI_torch.numpy())
            except AttributeError:
                error_64 = np.nan
            simulation.reBin(size//3, for_sas = True)
            simulation.drop_first_bin()
            simulation.init_sas_model()
            simulation.optimize_scaling()
            error_64_sas = compute_error(simulation.I_sas, simulation.binned_slice.I.values)
            chi_error_64 = simulation.Chi_squared_norm('ISigma')

            one_run  = pd.DataFrame({
                'shape': 'sphere',
                'sample':i, 
                'size': size, 
                'time_fftn_ 32': torch_32_time, 
                'time_fftn_ 64': torch_64_time, 
                'time_2D_ 32': simple_split_32_time, 
                'time_2D_ 64': simple_split_64_time, 
                'time_1D_ 32': split_32_time, 
                'time_1D_ 64': split_64_time, 
                'time_cpu_32': cpu_time_32,
                'time_cpu_64': cpu_time_64,
                'memory_density_fftn_32': torch_mem_32, 
                'memory_density_fftn_64': torch_mem_64, 
                'memory_density_2D_32': simple_split_mem_32, 
                'memory_density_2D_64': simple_split_mem_64, 
                'memory_density_1D_32': split_mem_32, 
                'memory_density_1D_64': split_mem_64, 
                'memory_fft_fftn_32': torch_fft_32, 
                'memory_fft_fftn_64': torch_fft_64, 
                'memory_fft_2D_32': simple_split_fft_32, 
                'memory_fft_2D_64': simple_split_fft_64, 
                'memory_fft_1D_32': split_fft_32, 
                'memory_fft_1D_64': split_fft_64, 
                'rel_error_32': error_32, 
                'rel_error_64' : error_64, 
                'rel_error_Sas_32': error_32_sas, 
                'rel_error_Sas_64': error_64_sas, 
                'chi2_32': chi_error_32, 
                'chi2_64': chi_error_64
            }, index = [i*len(sim_size)+t])
            one_run.to_csv('results_fft_new.csv', mode = 'a', header =False)


def run_hardspheres(sim_size):
    for i in range(10):
        for t, size in enumerate (sim_size):
            simulation = Sphere(size = 10, nPoints = size, volFrac = 0.15)
            simulation.place_shape(rMean = 0.05, rWidth = 0.1, nonoverlapping=True)
            simulation.pin_memory()
            torch_32_time, (torch_mem_32, torch_fft_32)  = timer_wrapper(simulation.calculate_torch_FTI, device = 'cuda') 
            if torch_fft_32 ==0:
                torch_32_time =0
            simple_split_32_time, (simple_split_mem_32, simple_split_fft_32) = timer_wrapper( simulation.calculate_custom_FTI, three_d = True, device = 'cuda', less_memory_use=True)
            cpu_time_32,(_,_) = timer_wrapper(simulation.calculate_custom_FTI, three_d = True, device = 'cpu')
            if size>=729:
                split_32_time,split_mem_32, split_fft_32 = 0,0,0
            else:
                split_32_time, (split_mem_32, split_fft_32) = timer_wrapper(simulation.calculate_custom_FTI,three_d = True, device = 'cuda', smallest_memory=True)

            try:
                error_32 = compute_error(simulation.FTI.numpy(),simulation.FTI_torch.numpy())
            except AttributeError:
                error_32 = np.nan
            simulation.reBin(size//3, for_sas = True)

            simulation.drop_first_bin()
            simulation.init_sas_model()
            print(simulation.Chi_squared_norm('ISigma'))
            simulation.optimize_scaling()
            error_32_sas = compute_error(simulation.I_sas, simulation.binned_slice.I.values)
            chi_error_32 = simulation.Chi_squared_norm('ISigma')
            print(simulation.Chi_squared_norm('ISigma'))
            
            simple_split_64_time, (simple_split_mem_64, simple_split_fft_64) = timer_wrapper(simulation.calculate_custom_FTI, three_d = True, device = 'cuda', less_memory_use=True, dtype = torch.complex128)
            torch_64_time, (torch_mem_64, torch_fft_64)  = timer_wrapper(simulation.calculate_torch_FTI, device = 'cuda', dtype = torch.complex128) 
            if torch_fft_64 ==0:
                torch_64_time =0
            cpu_time_64,(_,_) = timer_wrapper(simulation.calculate_custom_FTI, three_d = True, device = 'cpu', dtype = torch.complex128)
            if size>=729:
                split_64_time,split_mem_64, split_fft_64 = 0,0,0
            else:
                split_64_time, (split_mem_64, split_fft_64) = timer_wrapper(simulation.calculate_custom_FTI,three_d = True, device = 'cuda', smallest_memory=True, dtype = torch.complex128)
            
            try:
                error_64 = compute_error(simulation.FTI.numpy(),simulation.FTI_torch.numpy())
            except AttributeError:
                error_64 = np.nan
            simulation.reBin(size//3, for_sas = True)
            simulation.drop_first_bin()
            simulation.init_sas_model()
            simulation.optimize_scaling()
            error_64_sas = compute_error(simulation.I_sas, simulation.binned_slice.I.values)
            chi_error_64 = simulation.Chi_squared_norm('ISigma')

            one_run  = pd.DataFrame({
                'shape': 'hardsphere',
                'sample':i, 
                'size': size, 
                'time_fftn_ 32': torch_32_time, 
                'time_fftn_ 64': torch_64_time, 
                'time_2D_ 32': simple_split_32_time, 
                'time_2D_ 64': simple_split_64_time, 
                'time_1D_ 32': split_32_time, 
                'time_1D_ 64': split_64_time, 
                'time_cpu_32': cpu_time_32,
                'time_cpu_64': cpu_time_64,
                'memory_density_fftn_32': torch_mem_32, 
                'memory_density_fftn_64': torch_mem_64, 
                'memory_density_2D_32': simple_split_mem_32, 
                'memory_density_2D_64': simple_split_mem_64, 
                'memory_density_1D_32': split_mem_32, 
                'memory_density_1D_64': split_mem_64, 
                'memory_fft_fftn_32': torch_fft_32, 
                'memory_fft_fftn_64': torch_fft_64, 
                'memory_fft_2D_32': simple_split_fft_32, 
                'memory_fft_2D_64': simple_split_fft_64, 
                'memory_fft_1D_32': split_fft_32, 
                'memory_fft_1D_64': split_fft_64, 
                'rel_error_32': error_32, 
                'rel_error_64' : error_64, 
                'rel_error_Sas_32': error_32_sas, 
                'rel_error_Sas_64': error_64_sas, 
                'chi2_32': chi_error_32, 
                'chi2_64': chi_error_64
            }, index = [i*len(sim_size)+t])
            one_run.to_csv('results_fft_new.csv', mode = 'a', header =False)

def run_cylinders(sim_size):
    for i in range(5):
        for t, size in enumerate (sim_size):
            simulation = Cylinder(size = 10, nPoints = size, volFrac = 0.01)
            simulation.place_shape(rMean = 0.05, rWidth = 0.1, hMean = 3, hWidth = 0.17, theta = 10, nonoverlapping=False)
            simulation.pin_memory()

            torch_32_time, (torch_mem_32, torch_fft_32)  = timer_wrapper(simulation.calculate_torch_FTI, device = 'cuda') 
            if torch_fft_32 ==0:
                torch_32_time =0
            simple_split_32_time, (simple_split_mem_32, simple_split_fft_32) = timer_wrapper( simulation.calculate_custom_FTI, three_d = True, device = 'cuda', less_memory_use=True)
            cpu_time_32,(_,_) = timer_wrapper(simulation.calculate_custom_FTI, three_d = True, device = 'cpu',)
            if size>=729:
                split_32_time,split_mem_32, split_fft_32 = 0,0,0
            else:
                split_32_time, (split_mem_32, split_fft_32) = timer_wrapper(simulation.calculate_custom_FTI,three_d = True, device = 'cuda', smallest_memory=True)

            try:
                error_32 = compute_error(simulation.FTI.numpy(),simulation.FTI_torch.numpy())
            except AttributeError:
                error_32 = np.nan
            simulation.reBin(size//3)
            simulation.init_sas_model()
            print(simulation.Chi_squared_norm('ISigma'))
            simulation.optimize_scaling()
            error_32_sas = compute_error(simulation.I_sas, simulation.binned_slice.I.values.reshape(simulation.nBins, simulation.nBins))
            chi_error_32 = simulation.Chi_squared_norm('ISigma')
            print(simulation.Chi_squared_norm('ISigma'))
            
            simple_split_64_time, (simple_split_mem_64, simple_split_fft_64) = timer_wrapper(simulation.calculate_custom_FTI, three_d = True, device = 'cuda', less_memory_use=True, dtype = torch.complex128)
            torch_64_time, (torch_mem_64, torch_fft_64)  = timer_wrapper(simulation.calculate_torch_FTI, device = 'cuda', dtype = torch.complex128) 
            if torch_fft_64 ==0:
                torch_64_time =0
            cpu_time_64,(_,_) = timer_wrapper(simulation.calculate_custom_FTI, three_d = True, device = 'cpu', dtype = torch.complex128)
            if size>=729:
                split_64_time,split_mem_64, split_fft_64 = 0,0,0
            else:
                split_64_time, (split_mem_64, split_fft_64) = timer_wrapper(simulation.calculate_custom_FTI,three_d = True, device = 'cuda', smallest_memory=True, dtype = torch.complex128)
            try:
                error_64 = compute_error(simulation.FTI.numpy(),simulation.FTI_torch.numpy())
            except AttributeError:
                error_64 = np.nan
            simulation.reBin(size//3)
            simulation.init_sas_model()
            simulation.optimize_scaling()
            error_64_sas = compute_error(simulation.I_sas, simulation.binned_slice.I.values.reshape(simulation.nBins, simulation.nBins))
            chi_error_64 = simulation.Chi_squared_norm('ISigma')

            one_run  = pd.DataFrame({
                'shape': 'cylinder',
                'sample':i, 
                'size': size, 
                'time_fftn_ 32': torch_32_time, 
                'time_fftn_ 64': torch_64_time, 
                'time_2D_ 32': simple_split_32_time, 
                'time_2D_ 64': simple_split_64_time, 
                'time_1D_ 32': split_32_time, 
                'time_1D_ 64': split_64_time, 
                'time_cpu_32': cpu_time_32,
                'time_cpu_64': cpu_time_64,
                'memory_density_fftn_32': torch_mem_32, 
                'memory_density_fftn_64': torch_mem_64, 
                'memory_density_2D_32': simple_split_mem_32, 
                'memory_density_2D_64': simple_split_mem_64, 
                'memory_density_1D_32': split_mem_32, 
                'memory_density_1D_64': split_mem_64, 
                'memory_fft_fftn_32': torch_fft_32, 
                'memory_fft_fftn_64': torch_fft_64, 
                'memory_fft_2D_32': simple_split_fft_32, 
                'memory_fft_2D_64': simple_split_fft_64, 
                'memory_fft_1D_32': split_fft_32, 
                'memory_fft_1D_64': split_fft_64, 
                'rel_error_32': error_32, 
                'rel_error_64' : error_64, 
                'rel_error_Sas_32': error_32_sas, 
                'rel_error_Sas_64': error_64_sas, 
                'chi2_32': chi_error_32, 
                'chi2_64': chi_error_64
            }, index = [i*len(sim_size)+t])
            one_run.to_csv('results_fft_new.csv', mode = 'a', header =False)

def main():
    sim_size = [81, 125,243,343,441,625,729,875,945]
    #run_spheres(sim_size)
    #run_hardspheres(sim_size)
    run_cylinders(sim_size)


if __name__ == '__main__':
    main()