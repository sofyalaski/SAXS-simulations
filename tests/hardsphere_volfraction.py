#!/home/slaskina/.conda/envs/ma/bin/python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pyopencl as cl
import torch
cl.create_some_context()
import sys
#sys.path.append('/home/slaskina/SAXS-simulations')

from SAXSsimulations import  Sphere, Cylinder, DensityData
from SAXSsimulations.plotting import *

### This code creates 3D simulated hardspheres in the same setting(box size, radius, polydispersity. The only parameter to change is the volume fraction)

def compare_volfrcation():
    vf = [0.01, 0.05, 0.075, 0.1, 0.15, 0.3,0.25]
    plt.figure(figsize = (12,6))
    for alpha, s in enumerate(vf):
        simulation = Sphere(size = 100, nPoints = 243, volFrac = s)
        simulation.place_shape(rMean = 3.4, nonoverlapping=True)
        simulation.calculate_custom_FTI(three_d = False)
        simulation.reBin(200, for_sas=True)
        simulation.drop_first_bin()
        plt.plot(simulation.binned_slice.Q, simulation.binned_slice.I, color = 'purple', alpha = alpha/7+0.1 ,  label = s)

    
    plt.xlabel(r"q $nm^{-1}$")
    plt.ylabel(r"I $(m sr)^{-1}$")
    plt.xscale('log') 
    plt.yscale('log') 
    plt.title(r'Scattering curves for a hardsphere particle size of 5 nm in $250^3$ nm simulated box')
    plt.legend(title = 'Volume fraction' )
    plt.savefig('/home/slaskina/SAXS-simulations/figures/hardspheres_vf_comp.png', format = 'png')

compare_volfrcation()