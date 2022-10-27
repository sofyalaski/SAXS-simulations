#!/home/slaskina/.conda/envs/fft/bin/python
from SAXSsimulations import  Sphere, Cylinder, DensityData
from SAXSsimulations.plotting import *
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    simulation = Sphere(size = 10, nPoints = 1201, volFrac = 0.01)
    simulation.place_shape(nonoverlapping=False)

    simulation.pin_memory()
    simulation.calculate_custom_FTI(three_d = False, device = device, less_memory_use=True)
    simulation.reBin(200, slice = 'center')
    simulation.drop_first_bin()
    plot_Q_vs_I(simulation.binned_slice)
    simulation.save_data(uncertainty =  'IError',directory='dat_files')
    print("the sphere file with radius {r:.3f}  is saved".format(r = simulation.rMean))


if __name__ == '__main__':
    main()