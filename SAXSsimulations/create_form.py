import numpy as np
import pandas as pd
import torch
import scipy.optimize 
import scipy.stats
from SAXSsimulations.utils import Intensity_func
import sasmodels
import sasmodels.core as core
import sasmodels.direct_model as direct_model



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
        return int(self._box.sum())/self.nPoints**3
    
    @property
    def density(self):
        """
        Because the box might be changed at any moment density is the copy of it 
        """
        return self._box

    @property
    def FTI(self):
        """
        Set the Fourier transform variable to be the copy of the FT calulated the custom way
        """
        return self._FTI_custom

    @property
    def FTI_sinc(self):
        """
        Set the sinc'ed FTI
        """
        if 'FTI_slice_custom' in dir(self):
            return self.__sinc(self.FTI_slice_custom)
        else:
            return self.__sinc(self.FTI_custom)


    def __initialize_box(self):
        """
        Creates a 1D gridding and expands it into a 3D box filled with voxels of size 'grid_space'.
        The box is symmetric around 0 on all axes with the predefined grid points. 
        Calls a function to specify the 3D scattering angle Q

        """
        self.grid = torch.linspace(-self.box_size /2 , self.box_size /2, self.nPoints) 
        self.__expand_Q()
        self._box  = torch.zeros((self.nPoints,self.nPoints,self.nPoints), dtype = torch.float32)
        self.grid_space = self.box_size/(self.nPoints-1)
    
    def __expand_Q(self):
        """
        convert a 1D array to the array of scattering angles and expand it 
        to 3D reciprocal(?) space to get the scattering angle Q in nm-1

        """
        divLength = self.box_size/self.nPoints
        self.qx = torch.linspace(-torch.pi/divLength, torch.pi/divLength, self.nPoints)
        #qy = qx.clone()
        #qz = qx.clone()
        self._q2y = self.qx[None,:]
        self._q2z = self.qx[:,None]

        self._q3x = self.qx + 0 * self.qx[None,:,None] + 0 * self.qx[:,None,None]
        self._q3y = 0 * self.qx + self.qx[None,:,None] + 0 * self.qx[:,None,None]
        self._q3z = 0 * self.qx + 0 * self.qx[None,:,None] + self.qx[:,None,None]

        self.Q = torch.sqrt(self._q3x**2 + self._q3y**2 + self._q3z**2)


    ################################   The Fourier Transformation functions   ################################


    def pin_memory(self):
        self.density.pin_memory()


    def calculate_torch_FTI_3D(self, device = 'cuda', slice = None):
        """
        Calculates Fourier transform of a 3D box with torch fftn and shifts to nyquist frequency
        input:
            device: to use the GPU
            slice: if None, the central slice will be assigned to the attribute FTI_slice_torch
        """
        density = self.density.to(device)
        FT = torch.fft.fftn(density, norm = 'forward')
        FT = torch.fft.fftshift(FT)
        FTI = torch.abs(FT)**2
        self.FTI_torch = FTI.cpu().detach()
        if slice is None:
            slice = self.nPoints//2+1
        self.FTI_slice_torch = self.FTI_torch[slice,:,:]
    
    def calculate_custom_FTI_3D(self, device = 'cuda'):
        """
        calculate the 3D FFT via 2D FFT and then the last 1D FFT
        """
        FT = self.density.type(torch.complex128).to(device)
        for k in range(FT.shape[0]):
            if FT[k,:,:].any():
                FT[k,:,:] = torch.fft.fft2(FT[k,:,:], norm = 'forward')
        for i in range(FT.shape[1]):
            for j in range(FT.shape[2]):                
                FT[:,i,j] = torch.fft.fft(FT[:,i,j], norm = 'forward')

        FT = torch.fft.fftshift(FT)
        FTI = torch.abs(FT)**2
        self._FTI_custom = FTI.cpu().detach()

    def calculate_custom_FTI_3D_slice(self, device = 'cuda', slice = None):
        """
        calculate the 3D FFT AT CERTAIN SLICE via 2D FFT and then the last 1D FFT
        input:
            device: to use the GPU
            slice: if None, the central will be returned
        """
        if slice is None:
            slice = self.nPoints//2+1
        FT = self.density.type(torch.complex128).to(device)
        for i in range(FT.shape[1]):
            for j in range(FT.shape[2]):      
                if FT[:,i,j].any():          
                    FT[:,i,j] = torch.fft.fft(FT[:,i,j], norm = 'forward')
        FT = torch.fft.fftshift(FT) # shifts in 1 direction 
        FT_slice = torch.fft.fft2(FT[slice,:,:], norm = 'forward') # ft2 at slice
        FT_slice = torch.fft.fftshift(FT_slice) # shift the slice
        FTI = torch.abs(FT_slice)**2
        self.FTI_slice_custom = FTI.cpu().detach()
        
    def __sinc(self, FTI):
        """
        Applies the sinc function to the voxel and multiplies the result with the Fourier Transformed Structure. New attribute is 
        created as a convolution of sinc'ed voxel with the Fourier Transform of the structure
        """    
        if len(FTI.shape)==2:
            self.sinc = torch.abs(torch.special.sinc(self._q2y*self.grid_space/2/np.pi)*torch.special.sinc(self._q2z*self.grid_space/2/np.pi))**2
        else:
            self.sinc = torch.abs(torch.special.sinc(self._q3x*self.grid_space/2/np.pi)*torch.special.sinc(self._q3y*self.grid_space/2/np.pi)**torch.special.sinc(self._q3z*self.grid_space/2/np.pi))**2
        FTI_sinced = FTI * self.sinc
        return FTI_sinced
    
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

    def __reBinSlice(self, df, bins, IEMin, QEMin):

        """
        Unweighted rebinning funcionality with extended uncertainty estimation, adapted from the datamerge methods,
        as implemented in Paulina's notebook of spring 2020.
        Aggregates the values of the Data Frame into the predefined bins and calls the helper functions on it
        for desired properties.
        input: 
            df: pandas.DataFrame with the Fourier Transformed values
            bins: either predefined intervals on Q ( for sphere ) or the group (for cylinders, each group corresponds to interval in qy and qz)
            IEMin: coefficent used to determine ISigma
            QEMin: coefficent used to determine QSigma
        output:
            binned_data: pandas.DataFrame with the values of intensity I and scattering angle Q rebinned 
            in specified intervals and other calculated statistics on it
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

    def __mask_FT_to_sphere(self):
        x2x = self.grid[None,:]
        x2y = self.grid[:,None]
        radius = self.box_size//2
        if len(self.FTI_sinc.shape)==2:
            mask = (x2x)**2 + (x2y)**2 < radius**2 # center = 0
            self.FTI_sinc_masked = torch.where(mask, self.FTI_sinc, 0)
            self.Q_masked = torch.where(mask, self.Q[self.nPoints//2+1,:,:], 0)
        else:
            mask = Simulation(self.box_size, self.nPoints)
            if len(mask.grid[mask.grid == 0])==1:
                central_slice = torch.argwhere(mask.grid==0)[0,0] # start with the central slice
                for i in range(int(radius//mask.grid_space)): # last grid point fully covering the radius is considered , sphere is symmetric so work in both directions
                    d = mask.grid_space*i # calculate the distance grom the center to the slice
                    radius_at_d = torch.sqrt(radius**2-d**2) # calculate the radius of circle at slice using Pythagoras Theorem
                    circle_at_d = (x2x)**2 + (x2y)**2 < radius_at_d**2 # mask the circle location
                    mask._box[central_slice+i,circle_at_d] = 1 # density inside sphere
                    mask._box[central_slice-i,circle_at_d] = 1
            else:
                # if the center of the sphere in between of two grid points, find those points and do the same in both dierections
                nearest_bigger_ind = torch.argwhere(mask.grid>0)[0,0]
                for i in range(int(radius//self.grid_space)): # last grid point fully covering the radius is considered  
                    d1 = self.grid_space*i  - self.grid[nearest_bigger_ind-1]
                    d2 = self.grid_space*i + self.grid[nearest_bigger_ind]
                    radius_at_d1 = torch.sqrt(radius**2-d1**2)
                    radius_at_d2 = torch.sqrt(radius**2-d2**2)
                    circle_at_d1 = (x2x)**2 + (x2y)**2 < radius_at_d1**2
                    circle_at_d2 = (x2x)**2 + (x2y)**2 < radius_at_d2**2
                    mask._box[nearest_bigger_ind+i,circle_at_d1] = 1
                    mask._box[nearest_bigger_ind-1-i,circle_at_d2] = 1
            
            self.FTI_sinc_masked = torch.where(mask.density, self.FTI_sinc, np.nan)
            self.Q_masked = torch.where(mask.density, self.Q, np.nan)


    def __reBin_sphere(self, nbins, IEMin, QEMin, slice = 'center'):
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
            slice: if 'center' the central slice computed, if other integer, the slice at the integer is computed otherwise the 3D version is computed, 
        output:
            binned_data: pandas.DataFrame with the values of intensity I and scattering angle Q rebinned 
            in specified intervals and other calculated statistics on it
        """
        self.__mask_FT_to_sphere() # mask the Q to the sphere
        
        Q_masked_no_nan = self.Q_masked[~self.Q_masked.isnan()]
        FTI_masked_no_nan = self.FTI_sinc_masked[~self.FTI_sinc_masked.isnan()]
        qMin = float(Q_masked_no_nan[Q_masked_no_nan!=0].min())
        qMax = float(Q_masked_no_nan.max())

        # prepare bin edges:
        self.binEdges = np.logspace(np.log10(qMin ), np.log10(qMax), num=nbins + 1)

        # add a little to the end to ensure the last datapoint is captured:
        self.binEdges[-1] = self.binEdges[-1] + 1e-3 * (self.binEdges[-1] - self.binEdges[-2])
        if slice is not None:
            if slice=='center':
                slice = self.nPoints//2+1
            else:
                raise IndexError('Only rebins the central slice or the whole box')
            df = pd.DataFrame({'Q':Q_masked_no_nan, 
                            'I': FTI_masked_no_nan, 
                            'ISigma':0.01 * FTI_masked_no_nan})
            bins = pd.cut(df['Q'], self.binEdges, right = False, include_lowest = True)
            self.binned_slice = self.__reBinSlice(df, bins, IEMin, QEMin)
            
        else:
            # accumulate bins of slices in a new Data Frame
            binned_slices = pd.DataFrame()
            for nSlice in range(self.FTI_sinc.shape[0]):
                Q = self.Q[nSlice,:,:][~self.Q[nSlice,:,:].isnan()]
                I = self.FTI_sinc[nSlice,:,:][~self.FTI_sinc[nSlice,:,:].isnan()]
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

    def __reBin_cylinder(self, nbins, IEMin, QEMin):
        
        Q_central_slice = self.Q[self.nPoints//2+1,:,:]
        qMin = float(self.qx[self.qx!=0].min())
        qMax = float(self.qx.max())

        # prepare bin edges:
        self.binEdges = np.linspace(qMin, qMax, num=nbins + 1)
        self.binEdges[-1] = self.binEdges[-1] + 1e-3 * (self.binEdges[-1] - self.binEdges[-2])
                
        df = pd.DataFrame({'Q':Q_central_slice.flatten(), 
                        'I':self.FTI_sinc.numpy().flatten(),
                        'ISigma':0.01 * self.FTI_sinc.numpy().flatten()})
        df = df.assign(qy = self.qx[df.index//len(self.qx)], qz = self.qx[df.index%len(self.qx)])

        bins_qy = pd.cut(df['qy'], self.binEdges, right = False, include_lowest = True)
        bins_qz = pd.cut(df['qz'], self.binEdges, right = False, include_lowest = True)
        bins = pd.concat([bins_qy,bins_qz],axis=1)
        bins_id = bins.groupby(['qy','qz']).ngroup()
        
        self.binned_slice = self.__reBinSlice( df, bins_id, IEMin, QEMin)

    def reBin(self, nbins,IEMin=0.01, QEMin=0.01, slice = None):
        self.nBins = nbins
        if self.shape == 'sphere':
            self.__reBin_sphere(nbins, IEMin, QEMin, slice)
        elif self.shape == 'cylinder':
            self.__reBin_cylinder(nbins, IEMin, QEMin)



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

    def init_sas_model(self, IEmin=0.01):
        try:
            self.model = core.load_model(self.shape)

        except AttributeError:
            print("First create a manual simultion!")
        
        if self.shape == 'sphere':
            self.qx_sas = self.binned_slice['Q'].values if 'binned_slice' in dir(self) else (self.binned_data['Q'].values)
            self.Q_sas = np.array(self.qx_sas[np.newaxis, :])
        elif self.shape =='cylinder':
            qMin = float(self.qx[self.qx!=0].min())
            qMax = float(self.qx.max())
            self.qx_sas = np.linspace(qMin, qMax, num=self.nBins)
        self.modelParameters_sas = self.model.info.parameters.defaults.copy()
        if self.shape == 'sphere':
            self.modelParameters_sas.update({
                'radius': self.rMean, 
                'background':0., 
                'sld':1.,
                'sld_solvent':0.,
                'radius_pd': self.rWidth/self.rMean, 
                'radius_pd_type': 'gaussian', 
                'radius_pd_n': 35
                })
        elif self.shape =='cylinder':
            self.modelParameters_sas.update({
                'radius': self.rMean, 
                'background':0., 
                'sld':1.,
                'sld_solvent':0.,
                'radius_pd': self.rWidth/self.rMean, 
                'radius_pd_type': 'gaussian', 
                'radius_pd_n': 35, 
                'length': self.hMean, 
                'length_pd': self.hWidth/self.hMean, 
                'length_pd_type': 'gaussian', 
                'length_pd_n': 35,          
                'theta':self.theta,    
                'theta_pd': self.rotWidth,
                'theta_pd_type':self.theta_distribution,
                'theta_pd_n':10,
                'phi':90-self.phi,
                'phi_pd': self.rotWidth,
                'phi_pd_type':self.phi_distribution,
                'phi_pd_n':10
                })
        self.__create_sas_model()
                
    def __create_sas_model(self):
        if self.shape == 'sphere':
            self.kernel=self.model.make_kernel(self.Q_sas)
            self.I_sas = direct_model.call_kernel(self.kernel, self.modelParameters_sas)
        elif self.shape == 'cylinder':
            q2y = self.qx_sas + 0* self.qx_sas[:,np.newaxis]
            q2z = self.qx_sas[:,np.newaxis] + 0* self.qx_sas
            q2y = q2y.reshape(q2y.size)
            q2z = q2z.reshape(q2z.size)
            print('start creating', q2y.shape)
            self.kernel=self.model.make_kernel([q2y, q2z])
            print('kernel done')
            self.I_sas = direct_model.call_kernel(self.kernel, self.modelParameters_sas)
            print('model done')
            self.I_sas = self.I_sas.reshape(self.nBins, self.nBins)
        self.model.release()


    def update_scaling(self, value):
        self.modelParameters_sas.update({'scale':value})
        self.__create_sas_model()
    

    def Chi_squared_norm(self, uncertainty):
        if self.shape == 'sphere':
            chi_squared = ((self.I_sas - self.binned_slice['I'])**2/self.binned_slice[uncertainty]**2).sum() 
            return chi_squared / (len(self.I_sas) - 1)
        elif self.shape == 'cylinder':
            chi_squared = ((self.I_sas.flatten() - self.binned_slice['I'])**2/self.binned_slice[uncertainty]**2).sum() 
            return chi_squared / ( self.I_sas.shape[0]*self.I_sas.shape[1] - 1)

    def optimize_scaling(self):
        sol = scipy.optimize.least_squares(fun = Intensity_func, 
                                    x0 = self.modelParameters_sas['scale'], 
                                    method = 'lm',
                                    args = [self])
        self.update_scaling(sol.x)
                          

'''
def noisy(noise_type,image):
    if noise_type == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

'''