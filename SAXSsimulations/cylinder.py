import numpy as np
import pandas as pd
import itertools
import torch
import math
import time
from SAXSsimulations.create_form import Simulation

  
class Cylinder(Simulation):
    """
    Puts polydispersed cylinders into the box.
    """
    def __init__(self,size, nPoints, volFrac = 0.05):
        Simulation.__init__(self, size, nPoints, volFrac)  
        self.hWidth = None
        self.hMean = None
        self.rWidth = None
        self.rMean = None
        self.theta = None
        self.phi = None
        self.center = None  
        self.shape = 'cylinder'
        self.thetaWidth = 5
        self.shapes=0

    def place_shape(self, single = False, **kwargs):
        """
        Updates the Cylinder attributes and places one or many cylinders into the box. If the standard deviation for hight and radius are 
        not defined, sets the default values.
        """
        keys = ['rMean', 'rWidth', 'hMean', 'hWidth', 'center', 'theta']
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in keys)
        if self.rWidth is None:
            self.rWidth = 0.1
        if self.hWidth is None:
            self.hWidth = 0.15
        if self.volume_fraction_threshold != 0:
            self.__cylinder_in_box(single)
        
    
    def __cylinder_in_box(self,single):
        """
        
        Given a box fill it with cylinder(s). Cylinder must fit into box, otherwise it is discarded until a fitting one is sampled.
        Box might fit many cylinder if the volume fractuion is too small, unless the 'single' argument is specified.
        Calls the generate_cylinder function that creates cylinders as slices.
        Radius Mean and Hight Mean are sampled from normal distribution or were specified when instantiating a Cylinder object.
        Cylinder radius and height are sampled from a normal distribution around the respective Means. 
        Center is sampled from the uniform distribution or was specified for a single cylinder.
        Theta angle is sampled to indicate rotation between the cylinder axis to the beam  and Phi to indicate the  rotation about the beam.
        input:
            single[boolean] : create a single cylinder in a box?        
        """
        if self.theta is None:
            self.theta = 0
            self.theta_distribution = 'gaussian'
            self.phi_distribution = 'uniform'

        if self.rMean is None:
            self.rMean = -1
            while self.rMean<=0:
                self.rMean = np.random.normal(loc = self.box_size*0.005, scale= self.box_size*0.02 )

        if self.hMean is None:
            self.hMean = -1
            while self.hMean<=0:
                self.hMean = np.random.normal(loc = self.box_size*0.3, scale= self.box_size*0.1 )
        attempt = 0
        if single:
            phi = int(np.random.uniform(low = 0, high = 360))
            success = False
            while success == False and attempt <100:
                attempt+=1
                if self.center is not None and (not ((self.center >self.box_size/2)|(self.center<-self.box_size/2) == True).any()):
                    continue # center was passed and is inside box
                else:
                    self.center = np.zeros(3)
                    self.center[0:2] = np.random.uniform(low = -self.box_size/2 + self.rMean, high = self.box_size/2 - self.rMean, size = 2)
                    self.center[2] =  np.random.uniform(low = -self.box_size/2 + self.rMean, high =self.box_size/2 - self.hMean/np.cos(np.deg2rad(self.theta)))
                self.__generate_cylinder(self.rMean, self.hMean, self.center, self.theta, phi)
                success = self._pbc                  
                if success ==False:
                    self.center = None
                    print('attempt:', attempt, end = '\r')
                else:
                    self.shapes=1
                    print('volume fraction is {vf:.5f}, height is {h:.2f}, radius is {r:.2f}, center at ({cx:.1f},{cy:.1f},{cz:.1f}), rotation phi is {phi}, rotation theta is {theta} '
                        .format(vf = self.volume_fraction, h = self.hMean, r = self.rMean, cx=self.center[0], cy = self.center[1], cz = self.center[2], phi = phi, theta = self.theta))
        else:
            self.theta_all = []
            self.phi_all = []
            st = time.time()
            while self.volume_fraction<self.volume_fraction_threshold and (time.time()-st)<1200: # 20 min max
                success = False
                while success == False:
                    height = np.random.normal(loc = self.hMean, scale= self.hWidth)
                    radius = np.random.normal(loc = self.rMean, scale= self.rWidth )
                    theta = int(np.random.normal(loc = self.theta, scale= self.thetaWidth ) )
                    phi = int(np.random.uniform(low = 0, high = 360))
                    center = np.zeros(3)
                    center[0:2] = np.random.uniform(low = -self.box_size/2, high = self.box_size/2, size = 2)
                    center[2] =  np.random.uniform(low = -self.box_size/2 + self.rMean, high = self.box_size/2 -height/np.cos(np.deg2rad(theta)))

                    if ((center >self.box_size/2)|(center<-self.box_size/2) == True).any() or (radius <0) or (height <0):
                        continue # center is outside of box or radius or height is negatibve
                    self.__generate_cylinder(radius, height, center, theta, phi)
                    success = self._pbc    
                    if success:
                        self.shapes+=1
                        self.phi_all.append(phi)
                        self.theta_all.append(theta)
                        print('volume fraction is {vf:.5f}, height is {h:.3f}, radius is {r:.3f}, center at ({cx:.1f},{cy:.1f},{cz:.1f}), rotation phi is {phi}, rotation theta is {theta} '
                        .format(vf = self.volume_fraction, h = height, r = radius, cx=center[0], cy = center[1], cz = center[2], phi = phi, theta = theta), end = '\r')
                    attempt+=1


    def __points_in_cylinder(self, pt1, pt2, r, q):
        vec = pt2 - pt1
        const = r * np.linalg.norm(vec)
        return np.array((np.dot(q - pt1, vec) >= 0) * (np.dot(q - pt2, vec) <= 0) *(np.linalg.norm(np.cross(q - pt1, vec), axis=1) <= const), dtype=float)

    def __cylinder_pbc(self,mask):
        self._pbc =  not((mask[0,:,:] == True).any() or (mask[-1,:,:] == True).any() or (mask[:,0,:] == True).any() or (mask[:, -1,:] == True).any() or (mask[:,:,0] == True).any() or (mask[:,:,-1] == True).any())
            

    def __generate_cylinder(self, radius, height, center, theta, phi):
        cylinder_end = [center[0]+height*np.sin(np.deg2rad(theta))* np.cos(np.deg2rad(phi)) , 
                center[1]+height*np.sin(np.deg2rad(theta))* np.sin(np.deg2rad(phi)), 
                center[2]+np.cos(np.deg2rad(theta))*height]
        # check the points where the potential cylinder will take place - we need a meshgrid
        # to not check the whole box, we will only be checking in between the cylinder ends + radius at slice(for x,y) or capping (for z) plus some extra few points for safety
        grid_x = self.grid[(self.grid>=min(cylinder_end[0], center[0])-radius/np.cos(np.deg2rad(theta))- 2*self.grid_space ) & (self.grid<=max(cylinder_end[0], center[0])+radius/np.cos(np.deg2rad(theta))+ 2*self.grid_space) ]
        grid_y = self.grid[(self.grid>=min(cylinder_end[1], center[1])-radius/np.cos(np.deg2rad(theta))- 2*self.grid_space ) & (self.grid<=max(cylinder_end[1], center[1])+radius/np.cos(np.deg2rad(theta))+ 2*self.grid_space) ]
        grid_z = self.grid[(self.grid>=min(cylinder_end[2], center[2])-radius*np.tan(np.deg2rad(theta))- 2*self.grid_space ) & (self.grid<=max(cylinder_end[2], center[2])+radius*np.tan(np.deg2rad(theta))+ 2*self.grid_space) ] 
        coords = np.array(np.meshgrid(grid_x, grid_y, grid_z))

        #coords = np.array(np.meshgrid(self.grid, self.grid, self.grid))
        coordsArr = np.vstack(coords).reshape(3,-1).T
        cylinder = self.__points_in_cylinder(pt1 = center, pt2 = cylinder_end, r=radius, q=coordsArr)
        cylinder = np.array(cylinder).reshape(len(grid_y), len(grid_x), len(grid_z)).transpose(1,0,2)
        #cylinder = np.array(cylinder).reshape(self.nPoints, self.nPoints, self.nPoints)
        
        self.__cylinder_pbc(cylinder)
        if  self._pbc:
            # now assign the checked values to the box: need determine indices of grid_i start and end (by comparison), +1 at the end of slice 
            self._box[int((self.grid == grid_x[0]).nonzero(as_tuple=True)[0]): int((self.grid == grid_x[-1]).nonzero(as_tuple=True)[0])+1, \
                      int((self.grid == grid_y[0]).nonzero(as_tuple=True)[0]): int((self.grid == grid_y[-1]).nonzero(as_tuple=True)[0])+1, \
                      int((self.grid == grid_z[0]).nonzero(as_tuple=True)[0]): int((self.grid == grid_z[-1]).nonzero(as_tuple=True)[0])+1] +=  torch.from_numpy(cylinder)
            #self._box +=torch.from_numpy(cylinder)
            #self._box = torch.clamp(self._box, max=1)
  
    def save_data(self, uncertainty = "ISigma", directory='.', for_SasView = True):
        """
        Saves .dat file. If slice  of 3D Fourier Transform was created only, operates on that slice, otherwise on whole data.
        input:
            directory to save
            for_SasView: boolean, if True converts Q and I to SASView compartible values: Armstrong^-1 for Q and (m*sr)^-1.
        """
        Q = self.Q[self.nPoints//2+1,:,:].numpy()
        data = pd.DataFrame({'Qx': self.binned_slice['qy'], 
                             'Qy': self.binned_slice['qz'],
                             'I': self.binned_slice['I'], 
                             'ISigma': self.binned_slice[uncertainty]})
        if for_SasView:
            data.assign(Qx = data.Qx/10, Qy = data.Qy/10, I = data.I/100, ISigma = data.ISigma/100).to_csv(directory+'/polydispersed_cylinders_{r}_{h}.dat'.
            format(r = int(self.rMean*1000), h = int(self.hMean*1000)), header=None, index=None, columns=["Qx", "Qy", "I", uncertainty])
        else:
            data.to_csv(directory+'/polydispersed_cylinders_{r}_{h}.dat'.
            format(r = int(self.rMean*1000), h = int(self.hMean*1000)), header=None, index=None, columns=["Qx", "Qy", "I", uncertainty])

    
