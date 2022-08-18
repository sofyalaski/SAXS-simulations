import numpy as np
import pandas as pd
import torch

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
        divLength = self.box_size/self.nPoints
        self.qx = torch.linspace(-torch.pi/divLength, torch.pi/divLength, self.nPoints)
        #qy = qx.clone()
        #qz = qx.clone()

        q3x = self.qx + 0 * self.qx[None,:,None] + 0 * self.qx[:,None,None]
        q3y = 0 * self.qx + self.qx[None,:,None] + 0 * self.qx[:,None,None]
        q3z = 0 * self.qx + 0 * self.qx[None,:,None] + self.qx[:,None,None]

        self.Q = torch.sqrt(q3x**2 + q3y**2 + q3z**2)


    ################################   The Fourier Transformation functions   ################################
    def calculate_torch_FTI_3D(self, device = 'cuda', slice = None):
        """
        Calculates Fourier transform of a 3D box with torch fftn and shifts to nyquist frequency
        input:
            device: to use the GPU
            slice: if None, the central slice will be assigned to the attribute FTI_slice_torch
        """
        density = self.density.type(torch.complex64).to(device)
        FT = torch.fft.fftn(density, norm = 'forward')
        FT = torch.fft.fftshift(FT)
        FTI = torch.abs(FT)**2
        self.FTI_torch = FTI.cpu().detach().numpy()
        if slice is None:
            slice = self.nPoints//2+1
        self.FTI_slice_torch = self.FTI_torch[slice,:,:]
    
    def calculate_custom_FTI_3D(self, device = 'cuda'):
        """
        calculate the 3D FFT via 2D FFT and then the last 1D FFT
        """
        FT = self.density.type(torch.complex64).to(device)
        for k in range(FT.shape[0]):
            if FT[k,:,:].any():
                FT[k,:,:] = torch.fft.fft2(FT[k,:,:], norm = 'forward')
        for i in range(FT.shape[1]):
            for j in range(FT.shape[2]):                
                FT[:,i,j] = torch.fft.fft(FT[:,i,j], norm = 'forward')

        FT = torch.fft.fftshift(FT)
        FTI = torch.abs(FT)**2
        self.FTI_custom = FTI.cpu().detach().numpy()

    def calculate_custom_FTI_3D_slice(self, device = 'cuda', slice = None):
        """
        calculate the 3D FFT AT CERTAIN SLICE via 2D FFT and then the last 1D FFT
        input:
            device: to use the GPU
            slice: if None, the central will be returned
        """
        if slice is None:
            slice = self.nPoints//2+1
        FT = self.density.type(torch.complex64).to(device)
        for i in range(FT.shape[1]):
            for j in range(FT.shape[2]):      
                if FT[:,i,j].any():          
                    FT[:,i,j] = torch.fft.fft(FT[:,i,j], norm = 'forward')
        FT = torch.fft.fftshift(FT) # shifts in 1 direction 
        FT_slice = torch.fft.fft2(FT[slice,:,:], norm = 'forward') # ft2 at slice
        FT_slice = torch.fft.fftshift(FT_slice) # shift the slice
        FTI = torch.abs(FT_slice)**2
        self.FTI_slice_custom = FTI.cpu().detach().numpy()
        
    def sinc(self):
        """
        Applies the sinc function to the voxel and multiplies the result with the Fourier Transformed Structure. New attribute is 
        created as a convolution of sinc'ed voxel with the Fourier Transform of the structure
        """
        q_lengthbox = float(self.Q[0,0,0])-float(self.Q[0,0,1])
        self.FTI_sinc = self.FTI_slice_custom*np.sinc(q_lengthbox/np.pi)
    
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

    def reBin(self, nbins, IEMin=0.01, QEMin=0.01, slice = 'center'):
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
        qMin = float(self.Q[self.Q!=0].min())
        qMax = float(self.Q.max())

        # prepare bin edges:
        binEdges = np.logspace(np.log10(qMin ), np.log10(qMax), num=nbins + 1)

        # add a little to the end to ensure the last datapoint is captured:
        binEdges[-1] = binEdges[-1] + 1e-3 * (binEdges[-1] - binEdges[-2])
        if slice is not None:
            if slice=='center':
                slice = self.nPoints//2+1
            elif slice<0 or slice > (self.nPoints-1):
                raise IndexError('Desired slice is out of boundaries of the simulation box')
            df = pd.DataFrame({'Q':self.Q[slice,:,:].flatten(), 
                            'I':self.FTI_sinc[:,:].flatten(), 
                            'ISigma':0.01 * self.FTI_sinc[:,:].flatten()})
            self.binned_slice = self.__reBinSlice(df, binEdges, IEMin=0.01, QEMin=0.01)
            
        else:
            # accumulate bins of slices in a new Data Frame
            binned_slices = pd.DataFrame()
            for nSlice in range(self.FTI_sinc.shape[0]):
                df = pd.DataFrame({'Q':self.Q[nSlice,:,:].flatten(), 
                                'I':self.FTI_sinc[nSlice,:,:].flatten(), 
                                'ISigma':0.01 * self.FTI_sinc[nSlice,:,:].flatten()})
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
        try:
            self.model = sasmodels.core.load_model(self.shape)

        except AttributeError:
            print("First create a manual simultion!")
        
        #q = np.geomspace(float(self.binned_data['Q'].min()), float(self.binned_data['Q'].max()), 501)
        self.qx_sas = self.binned_data['Q'].values if 'binned_slice' in dir(self) else (self.binned_slice['Q'].values)
        self.Q_sas = np.array(self.qx_sas[np.newaxis, :])
        self.modelParameters_sas = self.model.info.parameters.defaults.copy()
        if self.shape == 'sphere':
            self.modelParameters_sas.update({
                'radius': self.rMean, 
                'background':0., 
                'sld':1.,
                'sld_solvent':0.,
                'radius_pd': self.rWidth, 
                'radius_pd_type': 'gaussian', 
                'radius_pd_n': 35
                })
        elif self.shape =='cylinder':
            self.modelParameters_sas.update({
                'radius': self.rMean, 
                'background':0., 
                'sld':1.,
                'sld_solvent':0.,
                'radius_pd': self.rWidth, 
                'radius_pd_type': 'gaussian', 
                'radius_pd_n': 35, 
                'length': self.hMean, 
                'length_pd': self.hWidth, 
                'length_pd_type': 'gaussian', 
                'length_pd_n': 35, 
                'theta': self.theta,
                'phi': self.phi
                })
        self.__create_sas_model()
                
    def __create_sas_model(self):
        self.kernel=self.model.make_kernel(self.Q_sas)
        self.I_sas = sasmodels.direct_model.call_kernel(self.kernel, self.modelParameters_sas)

    def update_scaling(self, value):
        self.modelParameters_sas.update({'scale':value})
        self.__create_sas_model()
    
