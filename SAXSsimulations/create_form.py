import numpy as np
import pandas as pd
import torch


class Simulation:
    def __init__(self, size, nPoints, volFrac = 0.05):
        self.box_size = size
        self.nPoints = nPoints
        self.volume_fraction_threshold = volFrac
        self.__initialize_box()
      
    ################################   The geometry functions   ################################
    @property
    def volume_fraction(self):
        """ 
        Volume fraction as a proportion of non-zero voxels to all voxels in a box
        """
        return int(self.box.sum())/self.nPoints**3
    
    @property
    def density(self):
        """
        Because the box might be changed at any moment density is the copy of it 
        """
        return self.box

    @property
    def FTI(self):
        """
        Set the Fourier transform variable to be the copy of the FT calulated the custom way
        """
        return self.FTI_custom

    def __initialize_box(self):
        """
        Creates a 1D gridding and expands it into a 3D box filled with voxels of size 'grid_space'.
        The box is symmetric around 0 on all axes with the predefined grid points. 
        Calls a function to specify the 3D scattering angle Q

        """
        self.grid = torch.linspace(-self.box_size /2 , self.box_size /2, self.nPoints) 
        self.__expand_Q()
        self.box  = torch.zeros((self.nPoints,self.nPoints,self.nPoints), dtype = torch.float32)
        self.grid_space = self.box_size/(self.nPoints-1)
    
    def __expand_Q(self):
        """
        convert a 1D array to the array of scattering angles and expand it 
        to 3D reciprocal(?) space to get the scattering angle Q in nm-1

        """

        self.qx = torch.linspace(torch.pi/self.grid.min(), torch.pi/self.grid.max(), self.nPoints)
        #qy = qx.clone()
        #qz = qx.clone()

        q3x = self.qx + 0 * self.qx[None,:,None] + 0 * self.qx[:,None,None]
        q3y = 0 * self.qx + self.qx[None,:,None] + 0 * self.qx[:,None,None]
        q3z = 0 * self.qx + 0 * self.qx[None,:,None] + self.qx[:,None,None]

        self.Q = torch.sqrt(q3x**2 + q3y**2 + q3z**2)

    def place_shape(self, shape, single = False, radius = None, center = None):
        self.shape = shape
        if self.shape == 'cylinder':
            self.__cylinder_in_box()
        elif self.shape == 'sphere':
            if not single :
                self.__sphere_in_box()
            else:
                self.__one_sphere_in_box(radius, center)
        else:
            raise  NotImplementedError('Other shapes are not supportd yet')

    def __cylinder_in_box(self):
        """
        Given a box, fills it with cylinder(s) at random center, radius and hight. 
        Cylinder must fit into box, otherwise it is discarded until a fitting one is sampled.
        Box might fit many cylinder if the volume fractuion is too small. 
        Calls the generate_cylinder function that creates cylinders as slices.
        Radius and Hight are sampled from normal distribution. After the first placed cylinder 
        both variables are sampled from a normal distribution centered around the radius/hight of the first one.
        Center is sampled from the uniform distribution
        """
        theta = np.random.uniform(low = 0, high = 90)
        phi = np.random.uniform(low = 0, high = 90)
        self.radius_global = 0
        self.height_global = 0
        while self.volume_fraction<self.volume_fraction_threshold:
            success = False
            while success == False:
                if self.height_global == 0:
                    height = np.random.normal(loc = self.box_size*0.15, scale = self.box_size*0.5 )
                    radius = np.random.normal(loc = self.box_size*0.05, scale= self.box_size*0.1 )
                else:
                    height = np.random.normal(loc = self.height_global, scale= self.height_global*0.2 )
                    radius = np.random.normal(loc = self.radius_global, scale= self.radius_global*0.1 )
                center = np.random.uniform(low = -self.box_size/2, high = self.box_size/2, size = 3)
                if ((center >self.box_size/2)|(center<-self.box_size/2) == True).any() or (radius <0) or (height <0):
                    continue # center is outside of box or radius is bigger than one fitting the box
                success = self.__generate_cylinder(radius, height, center, theta, phi)
                if success:
                    if self.height_global == 0: # set polydispersity arounf first created cylinder
                        self.height_global = height
                        self.radius_global = radius
                    print('volume fraction is {vf:.5f}, height is {h:.2f}, radius is {r:.2f}, center at ({cx:.1f},{cy:.1f},{cz:.1f}) '
                      .format(vf = self.volume_fraction, h = height, r = radius, cx=center[0], cy = center[1], cz = center[2]))

            

    def __generate_cylinder(self, radius, height, center, theta, phi):
        """
        Create a 3D cylinder as 2D slices. The slices are placed at the same distance to each other 
        as the grid inside the slices.
        input:
            radius: the radius of cylinder in nm
            center: triple that points to the center of the cylinder
            height: the hight of the cylinder, height/2 when counting from the center
            theta: angle in xy-plane
            phi: angle in xz-plane
        output:
            boolean: True if a cylinder was  placed in a box, otherwise, if constraints were not met returns False
        """

        #working on slices: slice through x
        x2y = self.grid[None,:]
        x2z = self.grid[:,None]
        cylinder_projection_on_x = int(height/2//self.grid_space*np.sin(np.deg2rad(theta))*np.sin(np.deg2rad(phi))) #in one direction
        radius_at_slice = np.cos(np.deg2rad(theta))*radius* np.cos(np.deg2rad(phi)) # calculate the radius of circle at slice at both rotations

        # if central slice on grid:
        if len(self.grid[self.grid == center[0]])==1:
            central_slice = torch.argwhere(self.grid==center[0])[0,0] # start with the central slice
            if central_slice+cylinder_projection_on_x >self.box.shape[0] or central_slice-cylinder_projection_on_x <0:
                print('--->outside of x plane')
                return False
            # check if circles at the ends of cylinder are inside the box  
            d = self.grid_space*cylinder_projection_on_x
            c_y_1 = center[1] + d*np.tan(np.deg2rad(theta)) # because of the theta rotation the y-coordinate of center at fixed x  slice shifts
            c_y_2 = center[1] - d*np.tan(np.deg2rad(theta)) 
            c_z_1 = center[2] + d*np.tan(np.deg2rad(phi)) # because of phi rotation the z-coordinate of center at fixed x shifts
            c_z_2 = center[2] - d*np.tan(np.deg2rad(phi)) 
            circle_1 = (x2y-c_y_1)**2 + (x2z-c_z_1)**2 < radius_at_slice**2 # mask the circle location
            circle_2 = (x2y-c_y_2)**2 + (x2z-c_z_2)**2 < radius_at_slice**2             

            if (circle_1[0,:] == True).any() or (circle_1[-1,:] == True).any() or (circle_1[:,0] == True).any() or (circle_1[:, -1] == True).any() or (circle_2[0,:] == True).any() or (circle_2[-1,:] == True).any() or (circle_2[:,0] == True).any() or (circle_2[:, -1] == True).any():
                print('--->outside on yz-plane')
                return  False
            # all checks done, cylinder fits the box, fill the densities by slices
            for i in range(cylinder_projection_on_x): # last grid point fully covering the radius is considered , cylinder is symmetric so work in both directions
                d = self.grid_space*i # calculate the distance grom the center to the slice
                c_y_1 = center[1] + d*np.tan(np.deg2rad(theta)) 
                c_y_2 = center[1] - d*np.tan(np.deg2rad(theta)) 
                c_z_1 = center[2] + d*np.tan(np.deg2rad(phi)) 
                c_z_2 = center[2] - d*np.tan(np.deg2rad(phi)) 

                circle_1 = (x2y-c_y_1)**2 + (x2z-c_z_1)**2 < radius_at_slice**2
                circle_2 = (x2y-c_y_2)**2 + (x2z-c_z_2)**2 < radius_at_slice**2 

                self.box[central_slice+i,circle_1] = 1 # density inside cylinder
                self.box[central_slice-i,circle_2] = 1
        else:
            # if the center of the cylinder in between of two grid points, find those points and do the same in both dierections
            nearest_bigger_ind = torch.argwhere(self.grid>center[0])[0,0]
            if nearest_bigger_ind+cylinder_projection_on_x >self.box.shape[0] or nearest_bigger_ind-1-cylinder_projection_on_x <0:
                print('--->outside of x plane')
                return  False
            # check if circles at the ends of cylinder are inside the box  
            d = self.grid_space*cylinder_projection_on_x
            c_y_1 = center[1] + d*np.tan(np.deg2rad(theta)) # because of the theta rotation the y-coordinate of center at fixed x  slice shifts
            c_y_2 = center[1] - d*np.tan(np.deg2rad(theta)) 
            c_z_1 = center[2] + d*np.tan(np.deg2rad(phi)) # because of phi rotation the z-coordinate of center at fixed x shifts
            c_z_2 = center[2] - d*np.tan(np.deg2rad(phi)) 
            circle_1 = (x2y-c_y_1)**2 + (x2z-c_z_1)**2 < radius_at_slice**2 # mask the circle location
            circle_2 = (x2y-c_y_2)**2 + (x2z-c_z_2)**2 < radius_at_slice**2             

            if (circle_1[0,:] == True).any() or (circle_1[-1,:] == True).any() or (circle_1[:,0] == True).any() or (circle_1[:, -1] == True).any() or (circle_2[0,:] == True).any() or (circle_2[-1,:] == True).any() or (circle_2[:,0] == True).any() or (circle_2[:, -1] == True).any():
                print('--->outside on yz-plane')
                return False
            # all checks done, cylinder fitsd the box, fill the densities by slices
            for i in range(cylinder_projection_on_x):
                d1 = self.grid_space*i + center[0] - self.grid[nearest_bigger_ind-1]
                d2 = self.grid_space*i + self.grid[nearest_bigger_ind]-center[0]

                c_y_1 = center[1] + d1*np.tan(np.deg2rad(theta)) 
                c_y_2 = center[1] - d2*np.tan(np.deg2rad(theta)) 
                c_z_1 = center[2] + d1*np.tan(np.deg2rad(phi)) 
                c_z_2 = center[2] - d2*np.tan(np.deg2rad(phi)) 

                circle_at_d1 = (x2y-c_y_1)**2 + (x2z-c_z_1)**2 < radius_at_slice**2
                circle_at_d2 = (x2y-c_y_2)**2 + (x2z-c_z_2)**2 < radius_at_slice**2

                self.box[nearest_bigger_ind+i,circle_at_d1] = 1
                self.box[nearest_bigger_ind-1-i,circle_at_d2] = 1
        return True

    def __sphere_in_box(self):
        """
        Given a box fill it with sphere(s) at random center and radius.
        Sphere must fit into box, otherwise it's discarded until a fitting one is sampled.
        Box might fit many spheres if the volume fractuion is too small. 
        Calls the generate_sphere function that creates spheres as slices.
        Radius is sampled from normal distribution. After the first placed sphere radius is sampled from a normal distribution 
        around the radius of the first one. Center is sampled from the uniform distribution.
        """
        self.radius_global = 0
        while self.volume_fraction<self.volume_fraction_threshold:
            success = False
            while success == False:
                if self.radius_global == 0:
                    radius = np.random.normal(loc = self.box_size*0.05, scale= self.box_size*0.1 )
                else: 
                    radius = np.random.normal(loc = self.radius_global, scale= self.radius_global*0.1 )
                # center is inside of the box minus the radius, s.t. the sphere fits inside the simulation box
                center = np.random.uniform(low = -self.box_size/2 +radius, high = self.box_size/2-radius, size = 3)
                if ((center >self.box_size/2)|(center<-self.box_size/2) == True).any() or (radius <0):
                    continue # center is outside of box or radius is bigger than one fitting the box
                self.__generate_sphere(radius, center)
                success = True
                if success:
                    if self.radius_global == 0: # set polydispersity around first created sphere
                        self.radius_global = radius
                    print('volume fraction is {vf:.5f}, radius is {r:.2f}, center at ({cx:.1f},{cy:.1f},{cz:.1f}) '
                           .format(vf = self.volume_fraction, r = radius, cx=center[0], cy = center[1], cz = center[2]))


    def __one_sphere_in_box(self, radius = None, center = None):
        """
        Given a box, fill it with sphere at apecified or random center and radius, sphere must fit into box, otherwise it's discarded until a fitting one is sampled.
        Calls the generate_sphere function that creates sphere as slices.
        Radius sampled from normal distribution, center from the uniform
        input:
            radius: a value for radius of the sphere
            center: 3 values specifying the center of the sphere
        """
        success = False
        while success == False:
            # center is inside of the box minus the radius, s.t. the sphere fits inside the simulation box
            if radius is None:
                radius = np.random.normal(loc = self.box_size*0.05, scale= self.box_size*0.1 )
            if center is None:
                center = np.random.uniform(low = -self.box_size/2 +radius, high = self.box_size/2-radius, size = 3)
            if ((center >self.box_size/2)|(center<-self.box_size/2) == True).any() or (radius <0):
                continue # center is outside of box or radius is bigger than one fitting the box
            box = self.__generate_sphere(radius, center)
            success = True
            if success:
                self.radius_global = radius
                vol_frac = 4/3*torch.pi*radius**3/self.box_size**3
                print('volume fraction is {vf:.5f}, radius is {r:.3f}, center at ({cx:.1f},{cy:.1f},{cz:.1f})'.format(vf = vol_frac, r = radius, cx=center[0], cy = center[1], cz = center[2]))

            
    def __generate_sphere(self, radius, center):
        """
        Create a 3D sphere as 2D slices. The slices are placed at the same distance to each other
        as the grid inside the slices.
        input:
            radius: the radius of sphere in nm
            center: triple that points to the center of the sphere
        output:
            box: box with a new sphere
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

                
    ################################   The Fourier Transformation functions   ################################
    def calculate_torch_FTI_3D(self, device = 'cuda'):
        """
        Calculates Fourier transform of a 3D box with torch fftn and shifts to nyquist frequency
        """
        density = self.density.type(torch.complex64).to(device)
        FT = torch.fft.fftn(density)
        FT = torch.fft.fftshift(FT)
        FTI = torch.abs(FT)**2
        self.FTI_torch = FTI.cpu().detach().numpy()
    
    def calculate_custom_FTI_3D(self, device = 'cuda'):
        """
        calculate the 3D FFT via 2D FFT and then the last 1D FFT
        """
        density = self.density.type(torch.complex64).to(device)
        for k in range(density.shape[0]):
            if density[k,:,:].any():
                density[k,:,:] = torch.fft.fft2(density[k,:,:])
        for i in range(density.shape[1]):
            for j in range(density.shape[2]):                
                density[:,i,j] = torch.fft.fft(density[:,i,j])

        density = torch.fft.fftshift(density)
        FTI = torch.abs(density)**2
        self.FTI_custom = FTI.cpu().detach().numpy()
    

    
    ################################   The rebinnning functions   ################################
    
    def __determine_Error(self,row):
        """ Given pandas.Series of Intesity I or scattering angle Q return the Error calculated as sum of 
        squared root of sum of squared values normaized by number of measurments. If a bin is empty returns NaN."""
        if row.empty:
            return np.nan
        else:
            return np.sqrt((row**2).sum())/ len(row)

    def __determine_std(self,row):
        """ Given pandas.Series of Intesity I or scattering angle Q return the standard deviation of the Series 
        or a value itself if only one values falls into the bin """
        if len(row) == 1:
            return row.iloc[0]
        else:
            return row.std(ddof = 1, skipna = True)

    def __determine_sem(self, row):
        """ Given pandas.Series of Intesity I or scattering angle Q return the standard error of the mean of the Series 
        or a value itself if only one values falls into the bin """
        if len(row) == 1:
            return row.iloc[0]
        else:
            return row.sem(ddof = 1, skipna = True)

    def __determine_ISigma(self, df, IEMin):
        """ Given pandas.Series of Intesity I return the maximum  within the Error of I, standard error of the mean of I or 
        the value of I multiplie by some coefficient IEmin"""
        return np.max([df.ISEM, df.IError, df.I*IEMin])

    def __determine_QSigma(self, df, QEMin):
        """ Given pandas.Series of scattering angle Q return the maximum  within the Error of Q(if presen), standard error 
        of the mean of Q or the value of Q multiplie by some coefficient QEmin"""
        if 'QError' in df.index:
            return np.max([df.QSEM, df.QError, df.Q*QEMin])
        else:
            return np.max([df.QSEM, df.Q*QEMin])

    def __reBinSlice(self, df, binEdges, IEMin=0.01, QEMin=0.01):

        """
        Unweighted rebinning funcionality with extended uncertainty estimation, adapted from the datamerge methods,
        as implemented in Paulina's notebook of spring 2020.
        Aggregates the values of the Data Frame into the predefined bins and calls the helper functions on it
        for desired properties.
        input: 
            df: pandas.DataFrame with the Fourier Transformed values
            IEMin: coefficent used to determine ISigma
            QEMin: coefficent used to determine QSigma
        output:
            binned_data: pandas.DataFrame with the values of intensity I and scattering angle Q rebinned 
            in specified intervals and other calculated statistics on it
        """
        bins = pd.cut(df['Q'], binEdges, right = False, include_lowest = True)

        if "QSigma" in df.keys(): 
            binned_data = df.groupby(bins).agg({'I':['mean', self.__determine_std, self.__determine_sem], 
                                                'ISigma': self.__determine_Error, 
                                                'Q':['mean', self.__determine_std, self.__determine_sem], 
                                                'QSigma': self.__determine_Error})
            binned_data.columns = ['I', 'IStd','ISEM', 'IError', 'Q','QStd', 'QSEM', 'QError']
        else: #  propagated uncertainties in Q if available
            binned_data = df.groupby(bins).agg({'I':['mean', self.__determine_std, self.__determine_sem], 
                                                'ISigma': self.__determine_Error, 
                                                'Q':['mean', self.__determine_std, self.__determine_sem]})
            binned_data.columns = ['I', 'IStd','ISEM', 'IError', 'Q','QStd', 'QSEM']

        binned_data['ISigma'] = binned_data.apply(lambda x: self.__determine_ISigma(x, IEMin), axis = 1) 
        # if Qerror ad QSEm are same(when not specified separately), save space and not copy it, Qsigma is fixed to account for it
        # or comment out next line to copy
        # binned_data['QError'] = binned_data['QSEM']
        binned_data['QSigma'] = binned_data.apply(lambda x: self.__determine_QSigma(x, QEMin), axis = 1) 

        # remove empty bins
        binned_data.dropna(thresh=4, inplace=True)
        return binned_data[['Q', 'I', 'IStd', 'ISEM', 'IError', 'ISigma', 'QStd', 'QSEM', 'QSigma']]

    def reBin(self, nbins, IEMin=0.01, QEMin=0.01):
        """
        Unweighted rebinning funcionality with extended uncertainty estimation, adapted from the datamerge methods,
        as implemented in Paulina's notebook of spring 2020.
        Calls the function on each sliced and then aggregates again for the all-slices statistics.
        input: 
            ft: 3D Fourier Transformed values
            Q: 3D values of the scattering angle
            nbins: how many bins shoould there be in a new data
            IEMin: coefficent used to determine ISigma
            QEMin: coefficent used to determine QSigma
        output:
            binned_data: pandas.DataFrame with the values of intensity I and scattering angle Q rebinned 
            in specified intervals and other calculated statistics on it
        """
        qMin = float(self.Q[self.Q!=0].min())
        qMax = float(self.Q.max())

        # prepare bin edges:
        binEdges = np.logspace(np.log10(qMin ), np.log10(qMax), num=nbins + 1)

        # add a little to the end to ensure the last datapoint is captured:
        binEdges[-1] = binEdges[-1] + 1e-3 * (binEdges[-1] - binEdges[-2])
        # accumulate bins of slices in a new Data Frame
        binned_slices = pd.DataFrame()
        for nSlice in range(self.FTI.shape[0]):
            df = pd.DataFrame({'Q':self.Q[nSlice,:,:].flatten(), 
                               'I':self.FTI[nSlice,:,:].flatten(), 
                               'ISigma':0.01 * self.FTI[nSlice,:,:].flatten()})
            binned_slices = pd.concat([binned_slices, self.__reBinSlice(df, binEdges, IEMin=0.01, QEMin=0.01)], axis=0) 

        # another aggregation round, now group by index, which represents bins already 
        if "QSigma" in binned_slices.keys():
            self.binned_data = binned_slices.groupby(binned_slices.index).agg({'I':['mean', self.__determine_std, self.__determine_sem], 
                                                                          'ISigma': self.__determine_Error, 
                                                                          'Q':['mean', self.__determine_std, self.__determine_sem], 
                                                                          'QSigma': self.__determine_Error})
            self.binned_data.columns = ['I', 'IStd','ISEM', 'IError', 'Q','QStd', 'QSEM', 'QError']
        else: #  propagated uncertainties in Q if available
            self.binned_data = binned_slices.groupby(binned_slices.index).agg({'I':['mean', self.__determine_std, self.__determine_sem],
                                                                               'ISigma': self.__determine_Error, 
                                                                               'Q':['mean', self.__determine_std, self.__determine_sem]})
            self.binned_data.columns = ['I', 'IStd','ISEM', 'IError', 'Q','QStd', 'QSEM']
        self.binned_data['ISigma'] = self.binned_data.apply(lambda x: self.__determine_ISigma(x, IEMin), axis = 1) 
        self.binned_data['QSigma'] = self.binned_data.apply(lambda x: self.__determine_QSigma(x, QEMin), axis = 1) 
        self.binned_data.dropna(thresh=4, inplace=True) # empty rows appear because of groupping by index in accumulated data, which in turn consisted of binEdges, which are sometimes empty


