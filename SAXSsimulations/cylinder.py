import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from SAXSsimulations.create_form import Simulation
  
  
class Cylinder(Simulation):
    def __init__(self,size, nPoints, volFrac = 0.05):
        Simulation.__init__(self, size, nPoints, volFrac)  
        self.hWidth = None
        self.hMean = None
        self.rWidth = None
        self.rMean = None
        self.center = None  
        self.shape = 'cylinder'

    def place_shape(self, single = False, **kwargs):
        """
        Updates the Cylinder attributes and places one or many cylinders into the box. If the standard deviation for hight and radius are not defined, sets the default values.
        """
        keys = ['rMean', 'rWidth', 'hMean', 'hWidth', 'center']
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in keys)
        if self.rWidth is None:
            self.rWidth = 0.1
        if self.hWidth is None:
            self.hWidth = 0.15
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
            single[boolean] : create a single sphere in a box?        
        """
        self.theta = np.random.uniform(low = 0, high = 45)
        self.phi = np.random.uniform(low = 0, high = 45)
        
        if self.rMean is None:
            self.rMean = -1
            while self.rMean<=0:
                self.rMean = np.random.normal(loc = self.box_size*0.02, scale= self.box_size*0.05 )
                print('rMean',self.rMean)

        if self.hMean is None:
            self.hMean = -1
            while self.hMean<=0:
                self.hMean = np.random.normal(loc = self.box_size*0.65, scale= self.box_size*0.1 )
        if single:
            success = False
            while success == False:
                if self.center is not None and (not ((self.center >self.box_size/2)|(self.center<-self.box_size/2) == True).any()):
                    continue # center was passed and is inside box
                else:
                    self.center = np.random.uniform(low = -self.box_size/2 + self.rMean, high = self.box_size/2 - self.rMean, size = 3)
                success = self.__generate_cylinder(self.rMean, self.hMean, self.center)
                if success ==False:
                    self.center = None
            print('volume fraction is {vf:.5f}, height is {h:.2f}, radius is {r:.2f}, center at ({cx:.1f},{cy:.1f},{cz:.1f}) '
                .format(vf = self.volume_fraction, h = self.hMean, r = self.rMean, cx=self.center[0], cy = self.center[1], cz = self.center[2]))
        else:
            while self.volume_fraction<self.volume_fraction_threshold:
                success = False
                while success == False:
                    height = np.random.normal(loc = self.hMean, scale= self.hWidth)
                    radius = np.random.normal(loc = self.rMean, scale= self.rWidth )
                    center = np.random.uniform(low = -self.box_size/2, high = self.box_size/2, size = 3)
                    if ((center >self.box_size/2)|(center<-self.box_size/2) == True).any() or (radius <0) or (height <0):
                        continue # center is outside of box or radius or height is negatibve
                    success = self.__generate_cylinder(radius, height, center)
                    if success:
                        print('volume fraction is {vf:.5f}, height is {h:.3f}, radius is {r:.3f}, center at ({cx:.1f},{cy:.1f},{cz:.1f}) '
                        .format(vf = self.volume_fraction, h = height, r = radius, cx=center[0], cy = center[1], cz = center[2]))

    
    def __create_slice(self, height, r_theta,r_phi, center, d, capping, cap_overhead, direction_right, check):
        """
        FIXME description
        """
        if direction_right:
            center_y = center[1] + d*np.tan(np.deg2rad(self.phi)) # because of the theta rotation the y-coordinate of center at fixed x  slice shifts
            center_z = center[2] + d*np.tan(np.deg2rad(self.theta)) # because of phi rotation the z-coordinate of center at fixed x shifts

        else:
            center_y = center[1] - d*np.tan(np.deg2rad(self.phi))
            center_z = center[2] - d*np.tan(np.deg2rad(self.theta))
        x2y = self.grid[None,:]
        x2z = self.grid[:,None]
        mask = ((x2y-center_y)**2/r_phi**2 + (x2z-center_z)**2/r_theta**2 <=1).type(torch.bool)
        mask_sum = mask.sum()
        if check:
            if float(mask.sum()) ==0 or (mask[0,:] == True).any() or (mask[-1,:] == True).any() or (mask[:,0] == True).any() or (mask[:, -1] == True).any():
                return torch.zeros_like(mask), True # check_failed
        if capping:
            cap_theta = r_theta + (height/2-d/np.cos(np.deg2rad(self.theta))) / np.sin(np.deg2rad(self.theta))
            cap_phi = r_phi + (height/2-d/np.cos(np.deg2rad(self.phi))) / np.sin(np.deg2rad(self.phi))
            if not cap_overhead:
                cap_theta = r_theta*2 - cap_theta 
                cap_phi = r_phi*2 - cap_phi
            # point A is first on the line = (cap_phi, cap_theta) now figure out point 2
            # say, point B lies on cap_phi and on diameter of ellipse r_phi -> it has the coord (cap_phi, center_z) -> AB = center_z - cap_theta
            #second point on the line is through triangle ABC, C outside of the ellipse, perp to AB: tan (theta) = BC/AB -> 
            # coordinates of C =(cap_phi - tan(theta)*(center_z - cap_theta), center_z)
            #isn't there a mix up in phi and theta?
            first_index_y = center_y - r_theta #int(torch.argwhere(mask.any(axis=0))[0][0])
            last_index_y = center_y + r_theta # int(torch.argwhere(mask.any(axis=0))[0][-1])
            first_index_z = center_z - r_phi # int(torch.argwhere(mask.any(axis=1))[0][0])
            last_index_z = center_z + r_phi #int(torch.argwhere(mask.any(axis=1))[0][-1])
            print('f_y:({f_y:.2f},{c_z:.2f}), l_y:({l_y:.2f},{c_z:.2f}), f_z:({c_y:.2f},{f_z:.2f}), l_z:({c_y:.2f},{l_z:.2f})'.format(f_y = first_index_y, f_z = first_index_z, l_y = last_index_y, l_z  =last_index_z, c_y = center_y, c_z = center_z))
            A = (float(first_index_y + cap_theta), float(last_index_z-cap_phi))
            C = (float(first_index_y + cap_theta - np.tan(np.deg2rad(self.theta + self.phi))*(last_index_z - cap_phi-center_z)), float(center_z ))
            print('A: ({a0:.2f},{a1:.2f}), C: ({c0:.2f},{c1:.2f})'.format(a0 = A[0], a1  =A[1], c0 = C[0], c1 = C[1])) 
            line = np.linalg.solve(np.array([[A[0], 1],[C[0],1]]), np.array([A[1],C[1]]))
            print("Ellipse equation: (x-({y:+.2f}))^2/{r_theta:.2f}^2 +(y -({z:+.2f}))^2/{r_phi:.2f}^2<1, line equation: y = {a:.2f}x{b:+.2f}, cap at ({cap_phi:.2f},{cap_theta:.2f})".
            format(r_theta = float(r_theta), r_phi = float(r_phi), cap_theta = float(cap_theta), cap_phi = float(cap_phi), y = float(center_y), z = float(center_z), a = line[0], b = line[1]))
            cap_mask = (x2y *line[0]+line[1]>x2z).type(torch.bool)
            if direction_right:
                mask = torch.logical_and(mask, cap_mask)
            else:
                mask = torch.logical_and(mask, torch.logical_not(cap_mask))
        return mask, False


    def __generate_cylinder(self, radius, height, center):
        """
        Create a 3D cylinder as 2D slices. The slices are placed at the same distance to each other 
        as the grid inside the slices. Updates  the class attributes `box` and `density`.
        input:
            radius: the radius of cylinder in nm
            center: triple that points to the center of the cylinder
            height: the hight of the cylinder, height/2 when counting from the center
        output:
            boolean: True if a cylinder was  placed in a box, otherwise, if constraints were not met returns False
        """

        #working on slices: slice through x
        central_axis_cylinder_projection_on_x = (height/2*np.cos(np.deg2rad(self.theta))*np.cos(np.deg2rad(self.phi)))//self.grid_space # projection of central cylinder axis on x-axis
        cylinder_rest_projection_on_x =  (np.sin(np.deg2rad(self.theta))*np.sin(np.deg2rad(self.phi))*radius)//self.grid_space # projection of the rest of the cylinder after the central axis on x-axis
        cylinder_projection_on_x = math.ceil(central_axis_cylinder_projection_on_x+ cylinder_rest_projection_on_x) # projection of whole cylinder on x-axis
        print(cylinder_projection_on_x, central_axis_cylinder_projection_on_x, cylinder_rest_projection_on_x, self.grid_space)
        radius_at_theta = radius/np.cos(np.deg2rad(self.theta)) # calculate the radius of ellipse at slice at both rotations
        radius_at_phi = radius/np.cos(np.deg2rad(self.phi))
        # if central slice on grid:
        if len(self.grid[self.grid == center[0]])==1:
            central_slice = int(torch.argwhere(self.grid==center[0])[0,0])# start with the central slice
            if central_slice+cylinder_projection_on_x >self.box.shape[0] or central_slice-cylinder_projection_on_x <0:
                #print('--->outside of x plane')
                return False
            # check if circles at the ends of cylinder are inside the box  
            d = self.grid_space*cylinder_projection_on_x
            capping = True
            cap_overhead = True
            circle_1, check_1 = self.__create_slice(height, radius_at_theta,radius_at_phi, center,d, capping, cap_overhead, direction_right = True, check = True)# mask the ellipse location
            circle_2, check_2 = self.__create_slice(height, radius_at_theta,radius_at_phi, center,d, capping, cap_overhead, direction_right = False, check = True)

            if  check_1 or check_2:
                #print('--->outside on yz-plane')
                return  False
            # all checks done, the latest sphere is not touching the edge of the box or is completelyoutside of it:  cylinder fits the box, fill the densities by slices
            for i in range(cylinder_projection_on_x): # last grid point fully covering the radius is considered , cylinder is symmetric so work in both directions
                d = self.grid_space*i # calculate the distance grom the center to the slice
                capping = i> central_axis_cylinder_projection_on_x-cylinder_rest_projection_on_x 
                if capping:
                    cap_overhead = i > central_axis_cylinder_projection_on_x
                else:
                    cap_overhead = False
                mask,_ =self.__create_slice(height, radius_at_theta,radius_at_phi, center, d, capping, cap_overhead, direction_right = True, check = False)
                self.box[central_slice+i,mask] = 1 # density inside cylinder
                mask,_ = self.__create_slice(height, radius_at_theta,radius_at_phi, center, d, capping, cap_overhead, direction_right = False, check = False)
                self.box[central_slice-i,mask] = 1

        else:
            # if the center of the cylinder in between of two grid points, find those points and do the same in both dierections
            nearest_bigger_ind = int(torch.argwhere(self.grid>center[0])[0,0])
            if nearest_bigger_ind+cylinder_projection_on_x >self.box.shape[0] or nearest_bigger_ind-1-cylinder_projection_on_x <0:
                #print('--->outside of x plane')
                return  False
            # check if circles at the ends of cylinder are inside the box  
            d1 = self.grid_space*cylinder_projection_on_x + self.grid[nearest_bigger_ind]-center[0]
            d2 = self.grid_space*cylinder_projection_on_x + center[0] - self.grid[nearest_bigger_ind-1]
            capping = True
            cap_overhead = True
            circle_1,check_1 = self.__create_slice(height, radius_at_theta,radius_at_phi, center, d1, capping, cap_overhead, direction_right = True, check = True)
            circle_2,check_2 = self.__create_slice(height, radius_at_theta,radius_at_phi, center, d2, capping, cap_overhead, direction_right = False, check = True)        
            if  check_1 or check_2:
                #print('--->outside on yz-plane')
                return False
            # all checks done, cylinder fitsd the box, fill the densities by slices
            for i in range(cylinder_projection_on_x):
                d1 = self.grid_space*i + self.grid[nearest_bigger_ind]-center[0]
                capping = i> central_axis_cylinder_projection_on_x-cylinder_rest_projection_on_x 
                if capping:
                    cap_overhead = i > central_axis_cylinder_projection_on_x
                    print('index:', i, cap_overhead)
                else:
                    cap_overhead = False
                mask,_ = self.__create_slice(height, radius_at_theta,radius_at_phi, center, d1, capping, cap_overhead, direction_right = True, check = False)
                self.box[nearest_bigger_ind+i,mask] = 1 # density inside cylinder
                d2 = self.grid_space*i + center[0] - self.grid[nearest_bigger_ind-1]
                mask,_ = self.__create_slice(height, radius_at_theta,radius_at_phi, center, d2, capping, cap_overhead, direction_right = False, check = False)
                self.box[nearest_bigger_ind-1-i,mask] = 1
        return True

    def save_data(self,  directory='.', for_SasView = True):
        """
        Saves .dat file. If slice  of 3D Fourier Transform was created only, operates on that slice, otherwise on whole data.
        input:
            directory to save
            for_SasView: boolean, if True converts Q and I to SASView compartible values: Armstrong^-1 for Q and (m*sr)^-1.
        """
        if 'binned_slice' in dir(self):
            data = self.binned_slice
        else:
            data = self.binned_data

        if for_SasView:
            data.assign(Q = data.Q/10, I = data.I/100, ISigma = data.ISigma/100).to_csv(directory+'/polydispersed_cylinders_{r}_{h}.dat'.
            format(r = int(self.rMean*1000), h = int(self.hMean*1000)), header=None, index=None, columns=["Q", "I", "ISigma"])
        else:
            data.to_csv(directory+'/polydispersed_cylinders_{r}_{h}.dat'.
            format(r = int(self.rMean*1000), h = int(self.hMean*1000)), header=None, index=None, columns=["Q", "I", "ISigma"])

