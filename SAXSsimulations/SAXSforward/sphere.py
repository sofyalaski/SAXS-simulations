import numpy as np
import matplotlib.pyplot as plt
import torch
from SAXSsimulations.SAXSforward.create_form import Simulation
  
  
class Sphere(Simulation):
    """
    Puts polydispersed spheres into the box.
    """
    def __init__(self,size, nPoints, volFrac = 0.05):
        self.rWidth = None
        self.rMean = None
        self.center = None  
        self.shape = 'sphere'
        super(Sphere, self).__init__(size, nPoints, volFrac)
        self.shapes = 0
        self.declined_shapes=0
        self.volfraction = volFrac

    def place_shape(self, single = False, nonoverlapping = False, **kwargs):
        """
        Updates the Sphere attributes and places one or many cylinders into the box. If the standard deviation for radius is not defined, 
        sets the default value.
        Arguments:
            single (bool): puts one or many cylinders into the box
            nonoverlapping(bool): option for hard spheres
            **kwargs
        """
        keys = ['rMean', 'rWidth', 'center']
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in keys)
        if self.rWidth is None:
            self.rWidth = 0.1
        if self.volume_fraction_threshold != 0:
            self.__sphere_in_box(single, nonoverlapping)
        self.hardsphere = nonoverlapping

    def __sphere_in_box(self, single, nonoverlapping):
        """
        Given a box fill it with sphere(s).
        Box might fit many spheres if the volume fractuion is too small, unless the 'single' argument is specified.
        Calls the `generate_sphere` function that creates spheres as slices.
        Radius Mean is sampled from normal distribution or was specified when instantiating a Sphere object.
        Sphere radius is sampled from a normal distribution around the Radius Mean. Center is sampled from the uniform distribution or was specified for a single sphere.
        Arguments:
            single (bool): puts one or many cylinders into the box
            nonoverlapping(bool): option for hard spheres
        """
        if self.rMean is None:
            self.rMean = -1
            while self.rMean<=0:
                self.rMean = np.random.normal(loc = self.box_size*0.005, scale= self.box_size*0.02 )
        
        if single:
            if self.center is None:
                self.center = np.random.uniform(low = -self.box_size/2 + self.rMean, high = self.box_size/2 - self.rMean, size = 3)
            _ = self.__generate_sphere(self.rMean, self.center, nonoverlapping)
            self.shapes=1
            print('volume fraction is {vf:.5f}, radius is {r:.2f}, center at ({cx:.1f},{cy:.1f},{cz:.1f}) '.format(vf = self.volume_fraction, r = self.rMean, cx=self.center[0], cy = self.center[1], cz = self.center[2]))
        else:
            while self.volume_fraction<self.volume_fraction_threshold:
                radius = -1
                while radius <= 0:
                    radius = np.random.normal(loc = self.rMean, scale= self.rWidth*self.rMean)
                # generate a position that is definitely inside the box
                center = np.random.uniform(low = -self.box_size/2 + radius, high = self.box_size/2 - radius, size = 3)
                success = self.__generate_sphere(radius, center, nonoverlapping)
                if success:
                    self.shapes+=1
                    #print('volume fraction is {vf:.5f}, radius is {r:.2f}, center at ({cx:.1f},{cy:.1f},{cz:.1f}) '.format(vf = self.volume_fraction, r = radius, cx=center[0], cy = center[1], cz = center[2]))
                else:
                    self.declined_shapes+=1

                print('volume fraction now:{vf:.3f}'.format(vf=self.volume_fraction/self.volume_fraction_threshold*100), end = '\r')
            print('spheres accepted:{ac} and declined: {dc}'.format(ac = self.shapes, dc = self.declined_shapes))
            
    def __generate_sphere(self, radius, center, nonoverlapping):
        """
        Create a 3D sphere as 2D slices. The slices are placed at the same distance to each other
        as the grid inside the slices. Updates  the class attributes `box` and `density`.
        Arguments:
            radius (float): the radius of sphere in nm
            center ([float, float, float]): points to the center of the sphere in cartesian coordinates system
            nonoverlapping (bool) : controls overlap of spheres
        Returns:
            bool: if a sphere was placed  
        """
        success = True
        if nonoverlapping:
            empty_box = torch.zeros_like(self._box)
        #working on slices: slice through x
        x2y = self.grid[:,None]
        x2z = self.grid[None,:]
        # the next few lines are required for nonoverlapping instances only, st the comparison on overlap is only done on a smaller sub-box containing sphere 
        center_ind_y = torch.argwhere(self.grid>center[1])[0,0]
        center_ind_z = torch.argwhere(self.grid>center[2])[0,0]
        y_r_min = center_ind_y - int(radius//self.grid_space)-2
        y_r_max = center_ind_y + int(radius//self.grid_space)+2
        z_r_min = center_ind_z - int(radius//self.grid_space)-2
        z_r_max = center_ind_z + int(radius//self.grid_space)+2

        # if central slice on grid:
        if len(self.grid[self.grid == center[0]])==1:
            central_slice = torch.argwhere(self.grid==center[0])[0,0] # start with the central slice
            x_r_min = central_slice-int(radius//self.grid_space)
            x_r_max = central_slice+int(radius//self.grid_space)+1
            for i in range(int(radius//self.grid_space)): # last grid point fully covering the radius is considered , sphere is symmetric so work in both directions
                d = self.grid_space*i # calculate the distance grom the center to the slice
                radius_at_d = np.sqrt(radius**2-d**2) # calculate the radius of circle at slice using Pythagoras Theorem
                circle_at_d = (x2y-center[1])**2 + (x2z-center[2])**2 < radius_at_d**2 # mask the circle location
                if nonoverlapping:
                    empty_box[central_slice+i,circle_at_d] = 1 # density inside sphere
                    empty_box[central_slice-i,circle_at_d] = 1
                else:
                    self._box[central_slice+i,circle_at_d] = 1 # density inside sphere
                    self._box[central_slice-i,circle_at_d] = 1
        else:
            # if the center of the sphere in between of two grid points, find those points and do the same in both dierections
            nearest_bigger_ind = torch.argwhere(self.grid>center[0])[0,0]
            x_r_min = nearest_bigger_ind-int(radius//self.grid_space)-1
            x_r_max = nearest_bigger_ind+int(radius//self.grid_space)+1
            for i in range(int(radius//self.grid_space)): # last grid point fully covering the radius is considered  
                d1 = self.grid_space*i + center[0] - self.grid[nearest_bigger_ind-1]
                d2 = self.grid_space*i + self.grid[nearest_bigger_ind]-center[0]
                radius_at_d1 = torch.sqrt(radius**2-d1**2)
                radius_at_d2 = torch.sqrt(radius**2-d2**2)
                circle_at_d1 = (x2y-center[1])**2 + (x2z-center[2])**2 < radius_at_d1**2
                circle_at_d2 = (x2y-center[1])**2 + (x2z-center[2])**2 < radius_at_d2**2
                if nonoverlapping:
                    empty_box[nearest_bigger_ind+i,circle_at_d1] = 1
                    empty_box[nearest_bigger_ind-1-i,circle_at_d2] = 1
                else:
                    self._box[nearest_bigger_ind+i,circle_at_d1] = 1
                    self._box[nearest_bigger_ind-1-i,circle_at_d2] = 1
        if nonoverlapping:
            # check if main simulation box overlaps with sphere
            if int(torch.logical_and(self._box[x_r_min:x_r_max, y_r_min: y_r_max, z_r_min:z_r_max],\
                 empty_box[x_r_min:x_r_max, y_r_min: y_r_max, z_r_min:z_r_max]).sum()) == 0:
                self._box[x_r_min:x_r_max, y_r_min: y_r_max, z_r_min:z_r_max] += empty_box[x_r_min:x_r_max, y_r_min: y_r_max, z_r_min:z_r_max]
            else:
                success = False
        return success


    def save_data(self, uncertainty = "ISigma", directory='.', for_SasView = True):
        """
        Saves .dat file. If slice  of 3D Fourier Transform was created only, operates on that slice, otherwise on whole data.
        Arguments:
            uncertainty (str): the uncertainty estimates that will be saved
            path (str): path to save
            for_SasView (bool): if True converts Q and I to SASView compartible values: Armstrong^-1 for Q and (m*sr)^-1.
        """
        if 'binned_slice' in dir(self):
            data = self.binned_slice
        else:
            data = self.binned_data

        if for_SasView:
            data.assign(Q = data.Q/10, I = data.I/100, ISigma = data.ISigma/100).to_csv(directory+'/polydispersed_spheres_{r}.dat'.
            format(r = int(self.rMean*1000)), header=None, index=None, columns=["Q", "I", uncertainty])
        else:
            data.to_csv(directory+'/polydispersed_spheres_{r}.dat'.
            format(r = int(self.rMean*1000)), header=None, index=None, columns=["Q", "I", uncertainty])