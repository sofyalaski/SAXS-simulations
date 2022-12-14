#!/home/slaskina/.conda/envs/ma/bin/python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pyopencl as cl
import torch
cl.create_some_context()
import sys
sys.path.append('/home/slaskina/SAXS-simulations')

from SAXSsimulations import  Sphere, Cylinder, DensityData
from SAXSsimulations.plotting import *

def compare_hardspheres():
    sizes = [81, 125,243,343,441,625,729,875,945]
    plt.figure(figsize = (12,6))
    for alpha, s in enumerate(sizes):
        simulation = Sphere(size = 250, nPoints = s, volFrac = 0.15)
        simulation.place_shape(rMean = 5, nonoverlapping=True)
        simulation.calculate_custom_FTI(three_d = False)
        simulation.reBin(s//3, for_sas=True)
        simulation.drop_first_bin()
        if s ==441:
            simulation.init_sas_model()
            simulation.optimize_scaling()
            plt.plot(simulation.qx_sas, simulation.I_sas, '-', color = 'red', label = 'SasView')   
        plt.plot(simulation.binned_slice.Q, simulation.binned_slice.I, color = 'blue', alpha = alpha/9+0.1 if alpha<441 else alpha/9,  label = '{} points'.format(s))

    
    plt.xlabel(r"q $nm^{-1}$")
    plt.ylabel(r"I $(m sr)^{-1}$")
    plt.xscale('log') 
    plt.yscale('log') 
    plt.title(r'$Scattering curves for a hardsphere particle size of 5 nm in 250^3 nm simulated box with 0.15 colume fraction$')
    plt.legend(title = 'Simulation' )
    plt.savefig('/home/slaskina/SAXS-simulations/figures/hardspheres_5nm_comp.png', format = 'png')


def compare_spheres():
    sizes = [81, 125,243,343,441,625,729,875,945]
    plt.figure(figsize = (12,6))
    for alpha, s in enumerate(sizes):
        simulation = Sphere(size = 250, nPoints = s, volFrac = 0.01)
        simulation.place_shape(rMean = 5, nonoverlapping=False)
        simulation.calculate_custom_FTI(three_d = False)
        simulation.reBin(s//3, for_sas=True)
        simulation.drop_first_bin()
        if s ==441:
            simulation.init_sas_model()
            simulation.optimize_scaling()
            plt.plot(simulation.qx_sas, simulation.I_sas, '-', color = 'red', label = 'SasView')   
        plt.plot(simulation.binned_slice.Q, simulation.binned_slice.I, color = 'blue', alpha = alpha/9+0.1 if alpha<441 else alpha/9,  label = '{} points'.format(s))

    
    plt.xlabel(r"q $nm^{-1}$")
    plt.ylabel(r"I $(m sr)^{-1}$")
    plt.xscale('log') 
    plt.yscale('log') 
    plt.title(r'$Scattering curves for a sphere particle size of 5 nm in 250^3 nm simulated box$')
    plt.legend(title = 'Simulation' )
    plt.savefig('/home/slaskina/SAXS-simulations/figures/spheres_5nm_comp.png', format = 'png')

def compare_cylinders():
    sizes = [81,243,441,729,945]
    fig,axs = plt.subplots(2,3,figsize = (15,9.5))
    for i, s in enumerate(sizes):
        simulation = Cylinder(size = 250, nPoints = s, volFrac = 0.01)
        simulation.place_shape(rMean = 5, hMean = 25, nonoverlapping=True)
        simulation.calculate_custom_FTI(three_d = False)
        simulation.reBin(s//3, for_sas=True)
        
        simulation.init_sas_model()
        simulation.optimize_scaling()
        if s ==441:
            ax = axs[1,2]
            im = ax.imshow(np.log10(simulation.I_sas), extent = [simulation.qx.min(), simulation.qx.max(), simulation.qx.min(), simulation.qx.max()])
            ax.set_xlabel(r"q $nm^{-1}$")
            ax.set_ylabel(r"q $nm^{-1}$")
            ax.set_title('SasModels simulation')
        ax = axs[i//3, i%3]
        binned_FTI = simulation.binned_slice['I'].values.reshape(simulation.nBins, simulation.nBins)
        im = ax.imshow(np.log10(binned_FTI), extent = [simulation.qx.min(), simulation.qx.max(), simulation.qx.min(), simulation.qx.max()])
        ax.set_xlabel(r"q $nm^{-1}$")
        ax.set_ylabel(r"q $nm^{-1}$")
        ax.set_title('{} points'.format(s))
    
   
    #plt.title(r'$Scattering curves for a cylinder particle with radius 5 nm and length 25 nm in 250^3 nm simulated box$')
    plt.savefig('/home/slaskina/SAXS-simulations/figures/cylinders_5nm_25_nm_comp.png', format = 'png')



#compare_spheres()
#compare_hardspheres()
compare_cylinders()