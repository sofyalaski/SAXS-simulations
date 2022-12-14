import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from SAXSsimulations.utils import compute_error


def plot_slices(density,grid, direction  = 'x', path = None):
    """ 
    plots 5 evenly spaced slices of the 3D entity
    input: 
        density: the 3D entity to plot
        grid: grid points of the entity to plot, used for labeling axes
        direction: choose between 'x/y/z' to plots slices on specific diension of the 3D entity
        path: path to save figure 
    """
    nPoints = density.shape[0]
    fig,axs = plt.subplots(1,5,figsize = (20,5))
    
    for i, sl in enumerate([1,3,5,7,9]):
        ax = axs[i]
        if direction == 'x':
            im = ax.imshow(density[nPoints//10*sl,:,:], extent = [np.round(float(grid.min()),2), np.round(float(grid.max()),2),np.round(float(grid.min()),2), np.round(float(grid.max()),2)])
        elif direction =='y':
            im = ax.imshow(density[:,nPoints//10*sl,:], extent = [np.round(float(grid.min()),2), np.round(float(grid.max()),2),np.round(float(grid.min()),2), np.round(float(grid.max()),2)])
        else:
            im = ax.imshow(density[:,:,nPoints//10*sl], extent = [np.round(float(grid.min()),2), np.round(float(grid.max()),2),np.round(float(grid.min()),2), np.round(float(grid.max()),2)])
        ax.set_title('slice {s}'.format(s = int(nPoints//10*sl)))
    plt.tight_layout()
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.suptitle('Realspace density')
        plt.show()
    
def plot_3D_structure(entity, grid, realspace = True, path = None):
    """ 
    plots the full 3D structure
    input: 
        entity: the 3D entity to plot
        grid: grid points of the entity to plot, used for labeling axes
        realspace[boolean]: if True considered to be density, if false, the Fourier Transform, values are log-values
        path: path to save figure 
    """
    nPoints = entity.shape[0]
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection ='3d')
    values = entity.nonzero()
    xx, yy, zz = values[:,0],values[:,1],values[:,2]
    img = ax.scatter3D(xx, yy, zz, zdir='z', c= entity[xx,yy,zz],  cmap="coolwarm" if realspace else 'Greys', s = 2, marker = 'o')
    ticks = np.linspace(0,nPoints,11)
    labels = np.round(np.linspace(grid.min(), grid.max(),11)).astype('int')  #labels = np.linspace(self.q3x.numpy().min(), self.q3x.numpy().max(),11)
    if realspace:
        img = ax.scatter3D(xx, yy, zz, zdir='z', c= entity[xx,yy,zz],  cmap="coolwarm" , s = 2, marker = 'o', vmin = 1, vmax =3)
        ax.set_xlabel(r'$X (nm)$')
        ax.set_ylabel(r'$Y (nm)$')
        ax.set_zlabel(r'$Z (nm)$')
        title = 'The density in 3D'
    else: # reciprocal
        img = ax.scatter3D(xx, yy, zz, zdir='z', c= np.log(entity[xx,yy,zz]),  cmap="Greys" , s = 2, marker = 'o')
        ax.set_xlabel('r$Q_x nm^{-1}$')
        ax.set_ylabel(r'$Q_y nm^{-1}$')
        ax.set_zlabel(r'$Q_z nm^{-1}$')
        title = 'Fourier transformed density in 3D'
        
    cbar = plt.colorbar(img, shrink = 0.8)
    cbar.set_label('log FFT')# only include colorbar when showing a FT
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_zticks(ticks, labels)
    ax.set_xlim(0,nPoints)
    ax.set_ylim(0,nPoints)
    ax.set_zlim(0,nPoints)
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.title(title)# only set title when not saving a file
        plt.show()

def plot_FTI_version(custom_version, torch_version, qx, slice_number = None, path = None):
    """ 
    plots custom FTI version versus the torch's and prints out the highest pixel-wise difference
    input: 
        custom_version: the instance with the FT calculatd a custom way
        torch_version: the instance with the FT calculatd with torch
        qx: scattering angles grid points of the entity to plot, used for labeling axes
        slice_number: the slice to plot, if not specified - central
        path: path to save figure 
    """
    # just another round to compare
    custom_version = custom_version.numpy()
    torch_version = torch_version.numpy()

    vmin = min(myround(np.log(custom_version).min()),myround(np.log(torch_version).min()))
    vmax = max(myround(np.log(custom_version).max()), myround(np.log(torch_version).max()))

    difference = compute_error(custom_version,torch_version)
    print('the maximal difference between the implementation of the FTI is {}'.format(difference.max()))
    if not slice_number:
        slice_number = custom_version.shape[0]//2+1
    
    fig,axs = plt.subplots(1,2,figsize = (20,10))
    ax = axs[0]
    if custom_version.ndim==3:
        im = ax.imshow(np.log(custom_version[slice_number,:,:]), cmap = 'Greys',  extent = [qx.min(), qx.max(), qx.min(), qx.max()], vmin = vmin, vmax = vmax)
    else:
        im = ax.imshow(np.log(custom_version), cmap = 'Greys',  extent = [qx.min(), qx.max(), qx.min(), qx.max()], vmin = vmin, vmax = vmax)
    ax.set_title('custom fft ')
    ax = axs[1]
    if torch_version.ndim ==3:
        im = ax.imshow(np.log(torch_version[slice_number,:,:]), cmap = 'Greys',  extent = [qx.min(), qx.max(), qx.min(), qx.max()], vmin = vmin, vmax = vmax)
    else:
        im = ax.imshow(np.log(torch_version), cmap = 'Greys',  extent = [qx.min(), qx.max(), qx.min(), qx.max()], vmin = vmin, vmax = vmax)
    ax.set_title('fftn')
    
    cbar = plt.colorbar(im, ax = axs.ravel().tolist(), shrink=0.8, location = 'left')
    cbar.set_label('log FFT')
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.suptitle("Comparison of the Fourier Transforms at slice {sl} calculated a custom way\nby 2D slices + final 1D FT and the torch's fftn".format(sl = slice_number))
        plt.show()

    
def plot_Q_vs_I(binned_data, yerr = 'ISigma', xlim = None, path = None):
    """
    After the rebinning procedure plot Intensity I vs the scattering angle Q on log-scale
    input:
        pandas DataFrame with calculated values
        yerr(optional) : the type of uncertainty to show on plot. Must be calculated and present in the df
        
        xlim(optional) : the limit of the x-axis
        path(optional) : path to save figure

    """
    binned_data.plot('Q', 'I', yerr = yerr, 
            figsize=(10,10),logx = True, logy = True, xlabel = 'q', ylabel = 'I', title = 'rebinned Q vs I')
    if xlim:
        plt.xlim(xlim[0], xlim[1]) # e.g. (10**(-2), 10**(-1))
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
    

def plot_simulation_vs_sas(simulation, uncertainty = 'ISigma'):
    """
    Plot Intensity I vs the scattering angle Q on log-scale for manual 3D simulation and an analytical SasModels
    input:
        simulation: a class instance with calculated attributes
        uncertainty: The value to plot as uncertainty
    """
    if simulation.shape == 'sphere':
        plt.figure(figsize = (10,10))
        print("relative error: {:.2f}".format(compute_error(simulation.I_sas, simulation.binned_slice.I.values)))
        plt.plot(simulation.qx_sas, simulation.I_sas, '-', color = 'red', label = 'SasView')
        if 'binned_slice' in dir(simulation):
            plt.plot(simulation.binned_slice.Q, simulation.binned_slice.I, color = 'blue', label = 'Simulation')
        else:
            plt.plot(simulation.binned_data.Q, simulation.binned_data.I, color = 'blue', label = 'Simulation') 
        plt.xlabel(r"q $nm^{-1}$", fontsize = 16)
        plt.ylabel(r"I $(m sr)^{-1}$", fontsize = 16)
        plt.xscale('log') 
        plt.yscale('log') 
        plt.title(r'$\chi^2$ error: {error}'.format(error = simulation.Chi_squared_norm(uncertainty) ))
        plt.legend()
    else:
        print("relative error: {:.2f}".format(compute_error(simulation.I_sas, simulation.binned_slice.I.values.reshape(simulation.nBins, simulation.nBins))))
        fig,axs = plt.subplots(1,3,figsize = (20,10))
        ax = axs[0]
        binned_FTI = simulation.binned_slice['I'].values.reshape(simulation.nBins, simulation.nBins)
        vmin = min(np.nanmin(binned_FTI),np.nanmin(simulation.I_sas))
        vmax =  max(np.nanmax(binned_FTI), np.nanmax(simulation.I_sas))
        im = ax.imshow(np.log(binned_FTI), vmin = np.log(vmin), vmax = np.log(vmax), extent = [simulation.qx.min(), simulation.qx.max(), simulation.qx.min(), simulation.qx.max()])
        ax.set_xlabel(r"q $nm^{-1}$", fontsize = 16)
        ax.set_ylabel(r"q $nm^{-1}$", fontsize = 16)
        #ax.set_title('manual simulation')
        ax = axs[1]
        im = ax.imshow(np.log(simulation.I_sas), vmin = np.log(vmin), vmax = np.log(vmax), extent = [simulation.qx.min(), simulation.qx.max(), simulation.qx.min(), simulation.qx.max()])
        ax.set_xlabel(r"q $nm^{-1}$", fontsize = 16)
        ax.set_ylabel(r"q $nm^{-1}$", fontsize = 16)
        #ax.set_title('SasModels simulation')
        ax = axs[2]
        im = ax.imshow(np.log(np.abs(binned_FTI-simulation.I_sas)) , vmin = np.log(vmin), vmax = np.log(vmax), extent = [simulation.qx.min(), simulation.qx.max(), simulation.qx.min(), simulation.qx.max()] )
        ax.set_xlabel(r"q $nm^{-1}$", fontsize = 16)
        ax.set_ylabel(r"q $nm^{-1}$", fontsize = 16)
        #ax.set_title('logged difference')
        
        cbar = plt.colorbar(im, ax = axs.ravel().tolist(), shrink=0.8, location = 'bottom',orientation='horizontal')
        cbar.set_label('log FFT')
        plt.suptitle(r'$\chi^2$ error: {error}'.format(error = simulation.Chi_squared_norm(uncertainty) ))


def plt_slices_sum(simulation):
    """
    Plots the simulated density summed onto one of the dimensions x/y/z.
    input:
        simulationa : class instance with calculated attributes
    """
    fig,axs = plt.subplots(1,3,figsize = (15,5))
    ax = axs[0]
    im = ax.imshow(simulation.density.sum(axis=0).T) # when sum on axis = x, the figure will be [y,z],  but i want z to be on y axis of the plot
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax = axs[1]
    im = ax.imshow(simulation.density.sum(axis=1).T) # same here z on the y-axis, x on the x axis
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax = axs[2]
    im = ax.imshow(simulation.density.sum(axis=2).T)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()    

def plot_slices_at_interval(interval,i,simulation, direction):
    """
    Plots 10 slices of the simulated density on given interval in given direction.
    input:
        interval : interval to plot slices
        i: start intervals from the slice #i
        simulation : class instance with calculated attributes
        direction: x/y/z axis
    """
    fig,axs = plt.subplots(1,10,figsize = (20,10))
    for t in range(10):
        ax = axs[t]
        if direction == 'x':
            ax.imshow(simulation.density[i +t*interval ,:,:])
        elif direction == 'y':
            ax.imshow(simulation.density[:,i +t*interval ,:])
        else:
            ax.imshow(simulation.density[:,:,i +t*interval ])
    

def plt_slices_ft(simulation):
    """
    Plots central slices of the Fourier transform given all dimensions x/y/z.
    input:
        simulation : class instance with calculated attributes
    """
    vmin = myround(np.log(simulation._FTI_custom).min())
    vmax = myround(np.log(simulation._FTI_custom).max())

    fig,axs = plt.subplots(1,3,figsize = (20,10))
    ax = axs[0]
    im = ax.imshow(np.log(simulation._FTI_custom[simulation._FTI_custom.shape[0]//2+1,:,:]), cmap = 'Greys', vmin = vmin, vmax = vmax, extent = [simulation.qx.min(), simulation.qx.max(), simulation.qx.min(), simulation.qx.max()])
    ax.set_ylabel(r"$Q_z\,nm^{-1}$", fontsize = 16)
    ax.set_xlabel(r"$Q_y\,nm^{-1}$", fontsize = 16)
    ax = axs[1]
    im = ax.imshow(np.log(simulation._FTI_custom[:,simulation._FTI_custom.shape[1]//2+1,:]), cmap = 'Greys', vmin = vmin, vmax = vmax, extent = [simulation.qx.min(), simulation.qx.max(), simulation.qx.min(), simulation.qx.max()])
    ax.set_ylabel(r"$Q_z\,nm^{-1}$", fontsize = 16)
    ax.set_xlabel(r"$Q_x\,nm^{-1}$", fontsize = 16)
    ax = axs[2]
    im = ax.imshow(np.log(simulation._FTI_custom[:,:,simulation._FTI_custom.shape[2]//2+1]), cmap = 'Greys', vmin = vmin, vmax = vmax, extent = [simulation.qx.min(), simulation.qx.max(), simulation.qx.min(), simulation.qx.max()])
    ax.set_ylabel(r"$Q_y\,nm^{-1}$", fontsize = 16)
    ax.set_xlabel(r"$Q_x\,nm^{-1}$", fontsize = 16)
    cbar = plt.colorbar(im, ax = axs.ravel().tolist(), shrink=0.8, location = 'bottom',orientation='horizontal')
    cbar.set_label('log FFT')
    plt.show()    

def plot_2d_sas(Intensity_2D_sas):
    """ plot 2D SasModels simulation"""
    plt.imshow(np.log10(Intensity_2D_sas))
    cbar = plt.colorbar()
    cbar.set_label('log Intensity')
def myround(x, base=5):
    d = (x/base)
    return base * np.ceil(d) if d >0 else base * np.floor(d) 