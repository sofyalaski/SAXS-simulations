import numpy as np
import torch
from SAXSsimulations.create_form import Simulation
  
  
class Sphere(Simulation):
    def __init__(self,size, nPoints, volFrac = 0.05):
        Simulation.__init__(self, size, nPoints, volFrac = 0.05)
        self.rWidth = None
        self.rMean = None
        self.center = None  
        self.shape = 'sphere'

    def place_shape(self, single = False, **kwargs):
        """
        Updates the Sphere attributes and places one or many cylinders into the box. If the standard deviation for radius is not defined, sets the default value.
        """
        keys = ['rMean', 'rWidth', 'center']
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in keys)
        if self.rWidth is None:
            self.rWidth = 0.1
        self.__sphere_in_box(single)
         

    def __sphere_in_box(self, single):
        """
        Given a box fill it with sphere(s).
        Box might fit many spheres if the volume fractuion is too small, unless the 'single' argument is specified.
        Calls the `generate_sphere` function that creates spheres as slices.
        Radius Mean is sampled from normal distribution or was specified when instantiating a Sphere object.
        Sphere radius is sampled from a normal distribution around the Radius Mean. Center is sampled from the uniform distribution or was specified for a single sphere.
        input:
            single[boolean] : create a single sphere in a box?
        """
        if self.rMean is None:
            self.rMean = -1
            while self.rMean<=0:
                self.rMean = np.random.normal(loc = self.box_size*0.05, scale= self.box_size*0.1 )
        
        if single:
            if self.center is None:
                self.center = np.random.uniform(low = -self.box_size/2 + self.rMean, high = self.box_size/2 - self.rMean, size = 3)
            self.__generate_sphere(self.rMean, self.center)
            print('volume fraction is {vf:.5f}, radius is {r:.2f}, center at ({cx:.1f},{cy:.1f},{cz:.1f}) '
                    .format(vf = self.volume_fraction, r = self.rMean, cx=self.center[0], cy = self.center[1], cz = self.center[2]))
        else:
            while self.volume_fraction<self.volume_fraction_threshold:
                radius = -1
                while radius <= 0:
                    radius = np.random.normal(loc = self.rMean, scale= self.rWidth)
                # generate a position that is definitely inside the box
                center = np.random.uniform(low = -self.box_size/2 + radius, high = self.box_size/2 - radius, size = 3)
                self.__generate_sphere(radius, center)
                print('volume fraction is {vf:.5f}, radius is {r:.2f}, center at ({cx:.1f},{cy:.1f},{cz:.1f}) '
                        .format(vf = self.volume_fraction, r = self.rMean, cx=center[0], cy = center[1], cz = center[2]))

            
    def __generate_sphere(self, radius, center):
        """
        Create a 3D sphere as 2D slices. The slices are placed at the same distance to each other
        as the grid inside the slices. Updates  the class attributes `box` and `density`.
        input:
            radius: the radius of sphere in nm
            center: triple that points to the center of the sphere
        """

        #working on slices: slice through x
        x2x = self.grid[None,:]
        x2y = self.grid[:,None]
        # if central slice on grid:
        if len(self.grid[self.grid == center[0]])==1:
            central_slice = torch.argwhere(self.grid==center[0])[0,0] # start with the central slice
            for i in range(int(radius//self.grid_space)): # last grid point fully covering the radius is considered , sphere is symmetric so work in both directions
                d = self.grid_space*i # calculate the distance grom the center to the slice
                radius_at_d = torch.sqrt(radius**2-d**2) # calculate the radius of circle at slice using Pythagoras Theorem
                circle_at_d = (x2x-center[1])**2 + (x2y-center[2])**2 < radius_at_d**2 # mask the circle location
                self.box[central_slice+i,circle_at_d] = 1 # density inside sphere
                self.box[central_slice-i,circle_at_d] = 1
        else:
            # if the center of the sphere in between of two grid points, find those points and do the same in both dierections
            nearest_bigger_ind = torch.argwhere(self.grid>center[0])[0,0]
            for i in range(int(radius//self.grid_space)): # last grid point fully covering the radius is considered  
                d1 = self.grid_space*i + center[0] - self.grid[nearest_bigger_ind-1]
                d2 = self.grid_space*i + self.grid[nearest_bigger_ind]-center[0]
                radius_at_d1 = torch.sqrt(radius**2-d1**2)
                radius_at_d2 = torch.sqrt(radius**2-d2**2)
                circle_at_d1 = (x2x-center[1])**2 + (x2y-center[2])**2 < radius_at_d1**2
                circle_at_d2 = (x2x-center[1])**2 + (x2y-center[2])**2 < radius_at_d2**2
                self.box[nearest_bigger_ind+i,circle_at_d1] = 1
                self.box[nearest_bigger_ind-1-i,circle_at_d2] = 1

    def save_data(self,  directory='.', for_SasView = True):
        """
        saves .dat file
        input:
            directory to save
            for_SasView: boolean, if True converts Q and I to SASView compartible values: Armstrong^-1 for Q and (m*sr)^-1
        """
        if for_SasView:
            self.binned_data.assign(Q = self.binned_data.Q/10, I = self.binned_data.I/100, ISigma = self.binned_data.ISigma/100).to_csv(directory+'/polydispersed_spheres_{r}.dat'.
            format(r = int(self.rMean*1000)), header=None, index=None, columns=["Q", "I", "ISigma"])
        else:
            self.binned_data.to_csv(directory+'/polydispersed_spheres_{r}.dat'.
            format(r = int(self.rMean*1000)), header=None, index=None, columns=["Q", "I", "ISigma"])