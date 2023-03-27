import numpy as np
import pandas as pd
import warnings
import torch
import scipy.optimize 
import scipy.stats
from SAXSsimulations.SAXSforward.utils import Intensity_func, tensor_informative
import sasmodels
import sasmodels.core as core
import sasmodels.direct_model as direct_model

__all__ = ['DensityData', 'Simulation']



class DensityData:
    """
    A class with Fourier Transform functionality. You can set any calculated density as an attribute and calculate FT with different  methods. 
    """
    def __init__(self):
        pass
    
    def set_density(self, density):
        """
        set density matrix from numpy array. The number of points is taken from the first dimension
        Arguments:
            density (np.array): a tensor with elctron density values
        """
        if type(density) != torch.Tensor:
            raise TypeError(" The dennsity data should be passed in a torch tensor")
        self.density = density
        self.nPoints = max(self.density.shape)
        self.grid = np.linspace(-self.nPoints/2,self.nPoints/2,self.nPoints)
        self.box_size= self.nPoints
        self.grid_space = self.box_size/(self.nPoints-1)

    @property
    def FTI(self):
        """
        Set the Fourier transform variable to be the copy of the FTI calulated the custom way
        """
        return self._FTI_custom

    ################################   The Fourier Transformation functions   ################################


    def pin_memory(self):
        """
        Copies the tensor to pinned memory
        """
        self.density.pin_memory()


    def calculate_torch_FTI(self, device = 'cuda', slice = None, dtype = torch.complex64, memory_stats = False):
        """
        Calculates Fourier transform of a 3D box with torch fftn and shifts to nyquist frequency
        Arguments:
            device ['cuda' or 'cpu']:  for use of the GPU, can optionaly specify the gpu id 'cuda:1'
            slice (int/None): set the slice number to be assigned to the attribute FTI_slice_torch. If None, the central slice will be assigned
            dtype (torch.complex__): set the floating points precision of FT matrix. Must be of complex type
            memory_stats (bool): specify to output statistics for how much memory is required to place density data and how much to calculate the FT of this data.
        Returns:
            float: memory usage to store density data in GB,
            flot:  memory usage to calculate FT for this density data in GB
            
        Warning: UserWarning: Casting complex values to real discards the imaginary part usually appears, it is however not affecting anything, becausethe tensor was just casted into complex one and in this check Can only have real values
        """
        FT = self.density.type(dtype)
        try:
            FT = FT.to(device)
            matrix_to_cuda = torch.cuda.memory_allocated()
            FT = torch.fft.fftn(FT, norm = 'forward')
            matrix_fft = torch.cuda.memory_reserved()
            FT = torch.fft.fftshift(FT).cpu().detach()
            FT = torch.abs(FT)**2
            self.FTI_torch = FT
            if slice is None:
                slice = self.nPoints//2+1
            self.FTI_slice_torch = self.FTI_torch[slice,:,:]
            if memory_stats:
                return matrix_to_cuda/1024**3, matrix_fft/1024**3
        except RuntimeError:
            del FT
            print("The simulation is too big to fit into GPU memory. The custom fft method uuuuuuuuuioshould be used ")
            if memory_stats:
                return 0, 0
        
    def __smallest_memory_FTI(self, dtype = torch.complex64):
        """
        Calculate FTI in form of 1D arrays. Is slow but very low in memory usage. Only used for benchmarking.
        Argumnents:
            dtype (torch.complex__): set the floating points precision of FT matrix. Must be of complex type
        Returns:
            torch.tensor: FT'ed density 
            float: memory usage to store density data in GB,
            flot:  memory usage to calculate FT for this density data in GB
        """
        FT = self.density.type(dtype)
        for i in range(FT.shape[1]):
            for j in range(FT.shape[2]):  
                FT_1D = FT[:,i,j].to('cuda')
                matrix_to_cuda = torch.cuda.memory_allocated() 
                FT_1D = torch.fft.fft(FT_1D, norm = 'forward')
                matrix_fft = torch.cuda.memory_reserved()
                FT[:,i,j] = FT_1D.cpu()
                del FT_1D
        for k in range(FT.shape[0]):
            for j in range(FT.shape[2]):
                FT_1D = FT[k,:,j].to('cuda')
                FT_1D = torch.fft.fft(FT_1D, norm = 'forward')
                FT[k, :,j] = FT_1D.cpu()
                del FT_1D
        for k in range(FT.shape[0]):
            for i in range(FT.shape[1]):
                FT_1D = FT[k,i, :].to('cuda')
                FT_1D = torch.fft.fft(FT_1D, norm = 'forward')
                FT[k, i,:] = FT_1D.cpu()
                del FT_1D
        return FT, matrix_to_cuda, matrix_fft

  
    



    def calculate_custom_FTI(self, three_d = False,  device = 'cuda', slice = None, dtype = torch.complex64, smallest_memory = False, memory_stats = False):
        """
        Calculate FTI with custom algorihtm with less memory requirments. 
        Arguments:
            three_d(bool): calculate full 3D FT or just a slice
            device ['cuda' or 'cpu']:  for use of the GPU, can optionaly specify the gpu id 'cuda:1'
            slice (int/None): set the slice number to be assigned to the attribute FTI_slice_torch. If None, the central slice will be assigned
            dtype (torch.complex__): set the floating points precision of FT matrix. Must be of complex type
            smallest_memory (bool):  break down FT caculation into 3 consecutive 1D operations (very slow) or 2D slice +1D array option
            memory_stats (bool): specify to output statistics for how much memory is required to place density data and how much to calculate the FT of this data
        Returns:
            float: memory usage to store density data in GB,
            flot:  memory usage to calculate FT for this density data in GB

        """
        FT = self.density.type(dtype)
        if three_d: 
            if smallest_memory:
                FT, matrix_to_cuda, matrix_fft = self.__smallest_memory_FTI(dtype)

            elif  device == 'cuda':
                for k in range(FT.shape[0]):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if tensor_informative(FT[k,:,:]):
                            FT_2D = FT[k,:,:].to(device) 
                            matrix_to_cuda = torch.cuda.memory_allocated() # this is the biggest it gets as it's 2D
                            FT_2D = torch.fft.fft2(FT_2D, norm = 'forward')
                            matrix_fft = torch.cuda.memory_reserved()
                            FT[k,:,:] = FT_2D.cpu()
                            del FT_2D
                for i in range(FT.shape[1]):
                    for j in range(FT.shape[2]):  
                        FT_1D = FT[:,i,j].to(device)
                        FT_1D = torch.fft.fft(FT_1D, norm = 'forward')
                        FT[:,i,j] = FT_1D.cpu()
                        del FT_1D
                
            else: 
                matrix_to_cuda = 0
                matrix_fft = 0
                for k in range(FT.shape[0]):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if tensor_informative(FT[k,:,:]):
                            FT[k,:,:] = torch.fft.fft2(FT[k,:,:], norm = 'forward')
                for i in range(FT.shape[1]):
                    for j in range(FT.shape[2]):                
                        FT[:,i,j] = torch.fft.fft(FT[:,i,j], norm = 'forward')
                FT = FT.cpu().detach()

            FT = torch.fft.fftshift(FT)
            FT = torch.abs(FT)**2
            self._FTI_custom = FT
            del FT
            if memory_stats:
                return matrix_to_cuda/1024**3, matrix_fft/1024**3
        else:
            if slice is None:
                slice = self.nPoints//2+1 # is slice is None get  central slice
        
            if device == 'cuda':
                for i in range(FT.shape[1]):
                    for j in range(FT.shape[2]):  
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if tensor_informative(FT[:,i,j]):
                                FT_1D = FT[:,i,j].to(device)
                                FT_1D = torch.fft.fft(FT_1D, norm = 'forward')
                                FT[:,i,j] = FT_1D.cpu()
                                del FT_1D
            else:
                for i in range(FT.shape[1]):
                    for j in range(FT.shape[2]):      
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if tensor_informative(FT[:,i,j]):      
                                FT[:,i,j] = torch.fft.fft(FT[:,i,j], norm = 'forward')
                FT = FT.cpu().detach()

                    
            FT = torch.fft.fftshift(FT) # shifts in 1 direction 
            FT_slice = torch.fft.fft2(FT[slice,:,:].to(device), norm = 'forward') # ft2 at slice
            FT_slice = torch.fft.fftshift(FT_slice) # shift the slice
            FT = torch.abs(FT_slice)**2
            self._FTI_custom = FT.cpu().detach()


    def mask_FT_to_sphere(self, instance, box_bins = None, fill_value = np.nan):
        """
        The values in the matrix have circular symmetry and the values in the corners of the matrix are underrepresented and will not be considered in the rebinning.
        Arguments: 
            instance (torch.tensor): an instance of the class to be masked. Might be Q vectors or FTI
            box_bins (int): number of points in the box
            fill_value: values to mask by, by default set to nan
        Returns:
            torch.tensor: masked instance
        """
                
        if box_bins is None:
            radius = self.nPoints/2
            x2x = self.grid[None,:]
            x2y = self.grid[:,None]
        else:
            radius = box_bins/2
            grid = np.arange( box_bins)
            x2x = grid[None,:]
            x2y = grid[:,None]
        if len(instance.shape)==2:
            x2x,x2y  = np.ogrid[:instance.shape[0], :instance.shape[1]]
            center = (int(instance.shape[0]//2+1), int(instance.shape[1]//2+1))
            mask = torch.from_numpy((x2x-center[0])**2 + (x2y-center[1])**2 < radius**2) # center = 0
            return torch.where(mask, instance, fill_value)
        else:
            mask = DensityData()
            mask.set_density(torch.zeros_like(instance))
            mask.grid = torch.arange(instance.shape[0])
            x2y,x2z  = np.ogrid[:instance.shape[1], :instance.shape[2]]
            center = (int(instance.shape[0]//2+1), int(instance.shape[1]//2+1),  int(instance.shape[2]//2+1))
            if len(mask.grid[mask.grid == center[0]])==1:
                for i in range(int(radius)): 
                    d = mask.grid_space*i # calculate the distance grom the center to the slice
                    radius_at_d = np.sqrt(radius**2-d**2) # calculate the radius of circle at slice using Pythagoras Theorem
                    circle_at_d = (x2y-center[1])**2 + (x2z-center[2])**2 < radius_at_d**2 # mask the circle location
                    mask.density[center[0]+i,circle_at_d] = 1
                    mask.density[center[0]-i,circle_at_d] = 1
            return torch.where(mask.density.type(torch.bool),instance, fill_value)




class Simulation(DensityData):

    """ A proper simulation class inheriting FT functions from DensityData. Given size and the number of points in the simulation
     it initializes empty simulation box and the scattering angle for each pixel. When volume fraction argument is passed, the shapes
     will be placed into the box until that threshold is reached."""
    def __init__(self, size, nPoints, volFrac = 0.05):
        self.box_size = size
        self.nPoints = nPoints
        self.volume_fraction_threshold = volFrac
        self.__initialize_box()
        super(Simulation, self).__init__()

      
    ################################   The geometry functions   ################################
    @property
    def volume_fraction(self):
        """ 
        Calculate volume fraction as a proportion of non-zero voxels to all voxels in a box
        Returns:
            float: proportion of pixels carrying density information  
        """
        return int(self._box.sum())/self.nPoints**3
    
    @property
    def density(self):
        """
        Density is the copy of the simulated box, because the box might be changed at any moment 
        Returns: 
            torch.tensor: density matrix
        """
        return self._box

    @property
    def FTI_sinc(self):
        """
        Set the sinc'ed FTI
        Returns:
            torch.tensor: sinc'ed smoothed out density 
        """
        return self.__sinc(self._FTI_custom)

    def __initialize_box(self):
        """
        Creates a 1D gridding and expands it into a 3D box filled with voxels of size 'grid_space'.
        The box is symmetric around 0 on all axes with the predefined grid points. 
        Calls a function to specify the 3D scattering angle Q

        """
        self.grid = torch.linspace(-self.box_size /2 , self.box_size /2, self.nPoints) 
        self._box  = torch.zeros((self.nPoints,self.nPoints,self.nPoints), dtype = torch.float32)
        self.grid_space = self.box_size/(self.nPoints-1)
        self.__set_q()
    
    def __set_q(self):
        """
        Sets a scattering vector Q in 1D in nm-1. because of the symmetry, they are the same and each direction, and scattering vectors in 2D/3D are spared for the sake of memory.
        """
        voxel_centers_dist = self.box_size/self.nPoints
        self.qx = torch.linspace(-torch.pi/voxel_centers_dist, torch.pi/voxel_centers_dist, self.nPoints)

        #self._q2y = self.qx[None,:]
        #self._q2z = self.qx[:,None]

        #self._q3x = self.qx + 0 * self.qx[None,:,None] + 0 * self.qx[:,None,None]
        #self._q3y = 0 * self.qx + self.qx[None,:,None] + 0 * self.qx[:,None,None]
        #self._q3z = 0 * self.qx + 0 * self.qx[None,:,None] + self.qx[:,None,None]

        #self.Q = torch.sqrt(self._q3x**2 + self._q3y**2 + self._q3z**2)

    def __Q_at_slice(self, slice):
        """ 
        Get vector of scattering angles Q in 3D reciprocal space at the certain slice
        Arguments:
            slice(int): the number of slice to calculate Q for
        Returns:
            torch.tensor: scattering angles in 2D
        """
        q2y = self.qx[None,:]
        q2z = self.qx[:,None]
        Q =  torch.sqrt(self.qx[slice]**2 + q2y**2 + q2z**2)
        return Q

    def __sinc(self, FTI):
        """
        Applies the sinc function to the voxel and multiplies the result with the Fourier Transformed Structure. New attribute is 
        created as a convolution of sinc'ed voxel with the Fourier Transform of the structure
        Argumnets:
            FTI (torch.tensor): the instance that will be sinc'ed
        Returns:
            torch.tensor: sinc'ed version of the instance
        """    
        if len(FTI.shape)==2:
            self.sinc = torch.abs(torch.special.sinc(self.qx[None,:]*self.grid_space/2/np.pi)*torch.special.sinc(self.qx[:,None]*self.grid_space/2/np.pi))**2
        else:
            self.sinc = torch.abs(torch.special.sinc(self.qx[:,None,None]*self.grid_space/2/np.pi)*torch.special.sinc(self.qx[None,:,None]*self.grid_space/2/np.pi)**torch.special.sinc( self.qx[None,None,:]*self.grid_space/2/np.pi))**2
        FTI_sinced = FTI * self.sinc
        return FTI_sinced   

    def calculate_FTI_sinc_masked(self):
        """
        Mask the sinc'ed FTI to the sphere/circle
        """    
        try:
            self.FTI_sinc_masked = self.mask_FT_to_sphere(self.FTI_sinc, box_bins = self.nPoints)
        except AttributeError:
            print ('compute sinc function `FTI_sinc()` first!' )



    ################################   The rebinnning functions   ################################
    
    def __determine_Error(self,row):
        """ 
        Calculate the Error as a sum of squared root of sum of squared values normaized by number of measurments. If a bin is empty returns NaN.
        Arguments:
            row (pandas.Series): the I or Q measurmets falling into one bin
        Returns:
            float: error    
        """
        if row.empty:
            return np.nan
        else:
            return np.sqrt((row**2).sum())/ len(row)

    def __determine_std(self,row):
        """ 
        Calulate the standard deviation.
        Arguments:
            row (pandas.Series): the I or Q measurmets falling into one bin
        Returns:
            float: Standard deviation    
        """
        if len(row) == 1:
            return row.iloc[0]
        else:
            return row.std(ddof = 1, skipna = True)

    def __determine_sem(self, row):
        """ 
        Calculate  the standard error of the mean.
        Arguments:
            row (pandas.Series): the I or Q measurmets falling into one bin
        Returns:
            float: SEM   
        """
        if len(row) == 1:
            return row.iloc[0]
        else:
            return row.sem(ddof = 1, skipna = True)

    def __determine_ISigma(self, df, IEMin):
        """ 
        Calculate the maximum  within the Error of I, standard error of the mean of I or the value of I multiplie by some coefficient IEmin
        Arguments:
            df (pandas.DataFrame): the subset of the data frame falling into one bin
            IEMin (float): a constant
        Returns:
            float: ISigma    
        """
        return np.max([df.ISEM, df.IError, df.I*IEMin])

    def __determine_QSigma(self, df, QEMin):
        """ 
        Calculate the maximum  within the Error of Q(if present), standard error of the mean of Q or the value of Q multiplie by some coefficient QEmin
        Arguments:
            df (pandas.DataFrame): the subset of the data frame falling into one bin
            QEMin (float): a constant
        Returns:
            float: QSigma    
        """
        if 'QError' in df.index:
            return np.max([df.QSEM, df.QError, df.Q*QEMin])
        else:
            return np.max([df.QSEM, df.Q*QEMin])

    def __reBinSlice(self, df, bins, IEMin, QEMin):

        """
        Unweighted rebinning funcionality with extended uncertainty estimation, adapted from the datamerge methods,
        as implemented in Paulina's notebook of spring 2020.
        Aggregates the values of the Data Frame into the predefined bins and calls the helper functions on it
        for desired properties.
        Arguments: 
            df (pandas.DataFrame): with the Fourier Transformed values
            bins (int): either predefined intervals on Q ( for sphere ) or the group (for cylinders, each group corresponds to interval in qy and qz)
            IEMin (float): coefficent used to determine ISigma
            QEMin (float): coefficent used to determine QSigma
        Returns:
            binned_dat (pandas.DataFrame): data frame with the values of intensity I and scattering angle Q rebinned in specified intervals and other calculated statistics on it
        """
        if self.shape == 'sphere':
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
        elif self.shape == 'cylinder':
            if "QSigma" in df.keys(): 
                binned_data = df.groupby(bins).agg({'I':['mean', self.__determine_std, self.__determine_sem], 
                                                    'ISigma': self.__determine_Error, 
                                                    'Q':['mean', self.__determine_std, self.__determine_sem], 
                                                    'QSigma': self.__determine_Error,
                                                    'qy':['mean', self.__determine_sem],
                                                    'qz':['mean', self.__determine_sem]})
                binned_data.columns = ['I', 'IStd','ISEM', 'IError', 'Q','QStd', 'QSEM', 'QError', 'qy', 'qy_sem', 'qz', 'qz_sem']
            else: #  propagated uncertainties in Q if available
                binned_data = df.groupby(bins).agg({'I':['mean', self.__determine_std, self.__determine_sem], 
                                                    'ISigma': self.__determine_Error, 
                                                    'Q':['mean', self.__determine_std, self.__determine_sem],
                                                    'qy':['mean', self.__determine_sem],
                                                    'qz':['mean', self.__determine_sem]})
                binned_data.columns = ['I', 'IStd','ISEM', 'IError', 'Q','QStd', 'QSEM', 'qy', 'qy_sem', 'qz', 'qz_sem']
        binned_data['ISigma'] = binned_data.apply(lambda x: self.__determine_ISigma(x, IEMin), axis = 1) 
        # if Qerror ad QSEm are same(when not specified separately), save space and not copy it, Qsigma is fixed to account for it
        # or comment out next line to copy
        # binned_data['QError'] = binned_data['QSEM']
        binned_data['QSigma'] = binned_data.apply(lambda x: self.__determine_QSigma(x, QEMin), axis = 1) 

        # remove empty bins
        binned_data.dropna(thresh=4, inplace=True)
        if self.shape == 'sphere':
            return binned_data[['Q', 'I', 'IStd', 'ISEM', 'IError', 'ISigma', 'QStd', 'QSEM', 'QSigma']]
        elif self.shape == 'cylinder':
            return binned_data[['Q', 'I', 'IStd', 'ISEM', 'IError', 'ISigma', 'QStd', 'QSEM', 'QSigma','qy', 'qy_sem', 'qz', 'qz_sem']]


    def __reBin_sphere(self,  IEMin, QEMin,for_sas):
        """
        For a sphere we consider a 1D rebinned curve. This function masks the relevant pixels (a sphere for a 3D instance and circle for a 2D)
        to rebin with the general rebinning function.
        Arguments: 
            IEMin (float): coefficent used to determine ISigma
            QEMin (float): coefficent used to determine QSigma
            for_sas (bool): when True the central slice is only used for rebinning
        
        """
        if for_sas and len(self.FTI_sinc.shape) ==3:
            FTI_sinc = self.FTI_sinc[self.nPoints//2+1, :, :]
            Q = self.__Q_at_slice(self.nPoints//2+1)
            self.FTI_sinc_masked = self.mask_FT_to_sphere(FTI_sinc, box_bins = self.nPoints)
            self.Q_masked = self.mask_FT_to_sphere(Q,box_bins = self.nPoints)


        if len(self.FTI_sinc.shape) ==3:
            self.FTI_sinc_masked = self.mask_FT_to_sphere(self.FTI_sinc, box_bins = self.nPoints)
            Q = torch.zeros_like(self.FTI_sinc)
            for i in range(self.FTI_sinc.shape[0]):
                Q[i,:,:] = self.__Q_at_slice(i)
            self.Q_masked = self.mask_FT_to_sphere(Q,box_bins = self.nPoints)
        else:
            self.FTI_sinc_masked = self.mask_FT_to_sphere(self.FTI_sinc)
            self.Q_masked =  self.mask_FT_to_sphere(self.__Q_at_slice(self.nPoints//2+1))
        
        Q_masked_no_nan = self.Q_masked[~self.Q_masked.isnan()]
        FTI_masked_no_nan = self.FTI_sinc_masked[~self.FTI_sinc_masked.isnan()]
        qMin = float(Q_masked_no_nan[Q_masked_no_nan!=0].min())
        qMax = float(Q_masked_no_nan.max())

        # prepare bin edges:
        self.binEdges = np.logspace(np.log10(qMin ), np.log10(qMax), num=self.nBins + 1)

        # add a little to the end to ensure the last datapoint is captured:
        self.binEdges[-1] = self.binEdges[-1] + 1e-3 * (self.binEdges[-1] - self.binEdges[-2])
        if for_sas:
            df = pd.DataFrame({'Q':Q_masked_no_nan, 
                            'I': FTI_masked_no_nan, 
                            'ISigma':0.01 * FTI_masked_no_nan})
            bins = pd.cut(df['Q'], self.binEdges, right = False, include_lowest = True)
            self.binned_slice = self.__reBinSlice(df, bins, IEMin, QEMin)
            
        else:
            # accumulate bins of slices in a new Data Frame
            binned_slices = pd.DataFrame()
            for n_slice in range(self.FTI_sinc.shape[0]):
                Q = self.__Q_at_slice(n_slice).flatten()                    
                I = self.FTI_sinc[n_slice,:,:][~self.FTI_sinc[n_slice,:,:].isnan()]
                df = pd.DataFrame({'Q':Q, 
                                'I':I,
                                'ISigma':0.01 * I})
                bins = pd.cut(df['Q'], self.binEdges, right = False, include_lowest = True)
                binned_slices = pd.concat([binned_slices, self.__reBinSlice(df, bins, IEMin, QEMin)], axis=0) 

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
            self.binned_data.dropna(thresh=4, inplace=True) # empty rows appear because of groupping by index in accumulated data, which in turn consisted of self.binEdges, which are sometimes empty

    def __reBin_cylinder(self, IEMin, QEMin):
        """
        In the cylinder the final output is the 2D scattering pattern, because otherwise the angular data is lost. 
        Only option to rebin the central slice exists.  Data will be rebinned into [nbins x nbins] matrix 
        Arguments: 
            IEMin (float): coefficent used to determine ISigma
            QEMin (float): coefficent used to determine QSigma
        """
        
        Q_central_slice = self.__Q_at_slice(self.nPoints//2+1)
        FTI = self.FTI_sinc[self.nPoints//2+1,:,:].numpy()if len(self.FTI_sinc.shape)==3 else self.FTI_sinc.numpy()
        qMin = float(self.qx[self.qx!=0].min())
        qMax = float(self.qx.max())

        # prepare bin edges:
        self.binEdges = np.linspace(qMin, qMax, num=self.nBins + 1)
        self.binEdges[-1] = self.binEdges[-1] + 1e-3 * (self.binEdges[-1] - self.binEdges[-2])
                
        df = pd.DataFrame({'Q':Q_central_slice.flatten(), 
                           'I':FTI.flatten(),
                           'ISigma':0.01 * FTI.flatten()})
        df = df.assign(qy = self.qx[df.index//len(self.qx)], qz = self.qx[df.index%len(self.qx)])

        bins_qy = pd.cut(df['qy'], self.binEdges, right = False, include_lowest = True)
        bins_qz = pd.cut(df['qz'], self.binEdges, right = False, include_lowest = True)
        bins = pd.concat([bins_qy,bins_qz],axis=1)
        bins_id = bins.groupby(['qy','qz']).ngroup()
        
        self.binned_slice = self.__reBinSlice( df, bins_id, IEMin, QEMin)

    def reBin(self, nbins,IEMin=0.01, QEMin=0.01,  for_sas = True):
        """
        depending on the shape of an object the proper function will be called
        Arguments:
            nbins (int): how many bins should there be in a new data
            IEMin (float): coefficent used to determine ISigma
            QEMin (float): coefficent used to determine QSigma
            for_sas (bool): when True the central slice is only used for rebinning
        """
        self.nBins = nbins
        if self.shape == 'sphere':
            self.__reBin_sphere( IEMin, QEMin,  for_sas)
        elif self.shape == 'cylinder':
            self.__reBin_cylinder( IEMin, QEMin)



    def drop_first_bin(self ):
        """
        Drops first bin in the rebinned data frime, because it's not needed for our purposes and has way too small scattering angle Q compared to the rest of bins.
        If slice  of 3D Fourier Transform was created only, operates on that slice, otherwise on whole data.
        """
        if 'binned_slice' in dir(self):
            self.binned_slice = self.binned_slice.iloc[1:]
        else:
            self.binned_data = self.binned_data.iloc[1:]
    ################################   The SasModels functions   ################################

    def init_sas_model(self):
        """
        Initialize an analytical SasModels simulation for a shape simulated before with matching parameters
        """
        try:
            if self.shape =='sphere' and self.hardsphere:
                print('simulating hardsphere')
                self.model = core.load_model('sphere@hardsphere')
            else:
                self.model = core.load_model(self.shape)

        except AttributeError:
            print("First create a manual simultion!")
        
        if self.shape == 'sphere':
            self.qx_sas = self.binned_slice['Q'].values if 'binned_slice' in dir(self) else (self.binned_data['Q'].values)
            self.Q_sas = np.array(self.qx_sas[np.newaxis, :])/10
        elif self.shape =='cylinder':
            qMin = float(self.qx[self.qx!=0].min())
            qMax = float(self.qx.max())
            self.qx_sas = np.linspace(qMin, qMax, num=self.nBins)/10
        self.modelParameters_sas = self.model.info.parameters.defaults.copy()
        if self.shape == 'sphere':
            self.modelParameters_sas.update({
                'radius': self.rMean*10, 
                'background':0., 
                'sld':1.,
                'sld_solvent':0.,
                'radius_pd': self.rWidth, 
                'radius_pd_type': 'gaussian', 
                'radius_pd_n': 35
                })
            if self.hardsphere:
                self.modelParameters_sas.update({
                    'radius_effective' : self.rMean*10,
                    'volfraction' : self.volfraction
                })
        elif self.shape =='cylinder':
            r = np.array(self.theta_all)

            self.modelParameters_sas.update({
                'radius': self.rMean*10, 
                'background':0., 
                'sld':1.,
                'sld_solvent':0.,
                'radius_pd': self.rWidth, #0.1
                'radius_pd_type': 'gaussian', 
                'radius_pd_n': 35, 
                'length': self.hMean*10, 
                'length_pd': self.hWidth, #0.15
                'length_pd_type': 'gaussian', 
                'length_pd_n': 35,          
                'theta':90,    
                'phi': 0,
                'phi_pd': self.theta, #self.thetaWidth,  
                'phi_pd_type':'uniform',
                'phi_pd_n':35
                })
        self.__create_sas_model()
        # mask edges end center
        if self.shape =='cylinder':
            self.__mask_2d_pattern()

    def __mask_2d_pattern(self):    
        """
        Mask the edges and the central circle of both simulations.
        """    
        q2x = np.arange(self.nBins) + 0* np.arange(self.nBins)[:,np.newaxis]
        q2y = np.arange(self.nBins)[:,np.newaxis] + 0* np.arange(self.nBins)
        mask = (q2x-(self.nBins//2+1))**2 + (q2y-(self.nBins//2+1))**2 > np.round(self.nBins/50)**2 # on the right side is the radius of sphere(in pixel) to mask in center 1% of box size 
        mask = self.mask_FT_to_sphere(torch.from_numpy(mask),box_bins =  self.nBins, fill_value=False).numpy()
        self.I_sas = np.where(mask, self.I_sas, np.nan)
        self.binned_slice["I"] = np.where(mask, self.binned_slice.I.values.reshape(self.nBins, self.nBins), np.nan).flatten()

    def __create_sas_model(self):
        """
        The core SasModel function to create a simulation. creates a kernel for scatterig angles represented in the simulation before.
        """
        if self.shape == 'sphere':
            self.kernel=self.model.make_kernel(self.Q_sas)
            self.I_sas = direct_model.call_kernel(self.kernel, self.modelParameters_sas)
        elif self.shape == 'cylinder':
            q2y = self.qx_sas + 0* self.qx_sas[:,np.newaxis]
            q2z = self.qx_sas[:,np.newaxis] + 0* self.qx_sas
            q2y = q2y.reshape(q2y.size)
            q2z = q2z.reshape(q2z.size)
            self.kernel=self.model.make_kernel([q2y, q2z])
            self.I_sas = direct_model.call_kernel(self.kernel, self.modelParameters_sas)
            self.I_sas = self.I_sas.reshape(self.nBins, self.nBins)*100                             # bring values to the (m_sr)-1 scale 
        self.model.release()


    def update_scaling(self, value):
        """
        Updates the scaling factor of SasModels simulation
        Arguments:
            value (float): the updated scaling factor
        """
        self.modelParameters_sas.update({'scale':value})
        self.__create_sas_model()
        # mask edges end center
        if self.shape =='cylinder':
            self.__mask_2d_pattern()
    

    def Chi_squared_norm(self, uncertainty):
        """
        Calculate the Chi squared error between analytical SasModels simulation and the manual 3D one.
        Arguments:
            uncertainty (str): the uncertainty metrics to be used
        """
        if self.shape == 'sphere':
            chi_squared = ((self.I_sas - self.binned_slice['I'])**2/self.binned_slice[uncertainty]**2).sum() 
            return chi_squared / (len(self.I_sas) - 1)
        elif self.shape == 'cylinder':
            chi_squared = ((self.I_sas.flatten() - self.binned_slice['I'])**2/self.binned_slice[uncertainty]**2).sum() 
            return chi_squared / ( self.I_sas.shape[0]*self.I_sas.shape[1] - 1)

    def optimize_scaling(self):
        """
        Optimizes the scaling factor of the SasModel simulation. Uses Scikit-learn least squares method, Levenberg-Marquardt algorithm
        with linear loss(not the chi squared as i sasModels because this combination is no possible in Scikit, but still ives good results.
        The function to optimize is simply the difference between SasModels and manual 3D simulations.  )
        """
        sol = scipy.optimize.least_squares(fun = Intensity_func, 
                                    x0 = self.modelParameters_sas['scale'], 
                                    method = 'lm',
                                    args = [self])
        self.update_scaling(sol.x)
     


  