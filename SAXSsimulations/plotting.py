import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from SAXSsimulations.utils import compute_error


def plot_slices(density,grid, file = None, file_format = None):
    nPoints = density.shape[0]
    fig,axs = plt.subplots(1,5,figsize = (20,10))
    
    for i, sl in enumerate([1,3,5,7,9]):
        ax = axs[i]
        im = ax.imshow(density[nPoints//10*sl,:,:], extent = [np.round(float(grid.min()),2), np.round(float(grid.max()),2),np.round(float(grid.min()),2), np.round(float(grid.max()),2)])
        ax.set_title('slice {s}'.format(s = int(nPoints//10*sl)))
    plt.tight_layout()
    if file:
        if not file_format:
            file_format = 'png'
        plt.savefig(file, format = file_format)
    else:
        plt.suptitle('Realspace density')
        plt.show()
    
def plot_3D_structure(entity, grid, realspace = True, file = None, file_format = None):
    nPoints = entity.shape[0]
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection ='3d')
    values = entity.nonzero()
    xx, yy, zz = values[:,0],values[:,1],values[:,2]
    img = ax.scatter3D(xx, yy, zz, zdir='z', c= entity[xx,yy,zz],  cmap="coolwarm" if realspace else 'Greys', s = 2, marker = 'o')
    ticks = np.linspace(0,nPoints,11)
    labels = np.round(np.linspace(grid.min(), grid.max(),11),2)  #labels = np.linspace(self.q3x.numpy().min(), self.q3x.numpy().max(),11)
    if realspace:
        img = ax.scatter3D(xx, yy, zz, zdir='z', c= entity[xx,yy,zz],  cmap="coolwarm" , s = 2, marker = 'o')
        ax.set_xlabel('nm')
        ax.set_ylabel('nm')
        ax.set_zlabel('nm')
        title = 'The density in 3D'
    else: # reciprocal
        img = ax.scatter3D(xx, yy, zz, zdir='z', c= np.log(entity[xx,yy,zz]),  cmap="Greys" , s = 2, marker = 'o')
        ax.set_xlabel('$nm^{-1}$')
        ax.set_ylabel('$nm^{-1}$')
        ax.set_zlabel('$nm^{-1}$')
        title = 'Fourier transformed density in 3D'
        plt.colorbar(img) # only include colorbar when showing a FT
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_zticks(ticks, labels)
    ax.set_xlim(0,nPoints)
    ax.set_ylim(0,nPoints)
    ax.set_zlim(0,nPoints)
    if file:
        if not file_format:
            file_format = 'png'
        plt.savefig(file, format = file_format)
    else:
        plt.title(title)# only set title when not saving a file
        plt.show()
    
def plot_FTI_version(custom_version, torch_version, qx, slice_number = None, file = None, file_format = None):
    # just another round to compare
    difference = compute_error(custom_version,torch_version)
    print('the maximal difference between the implementation of the FTI is {}'.format(difference.max()))
    if not slice_number:
        slice_number = np.random.randint(low = 0, high = custom_version.shape[0])
    
    print(slice_number)
    fig,axs = plt.subplots(1,2,figsize = (20,10))
    ax = axs[0]
    im = ax.imshow(np.log(custom_version[slice_number,:,:]), cmap = 'Greys', vmin = 0, vmax = 20, extent = [qx.min(), qx.max(), qx.min(), qx.max()])
    ax.set_title('custom fft ')
    ax = axs[1]
    im = ax.imshow(np.log(torch_version[slice_number,:,:]), cmap = 'Greys', vmin = 0, vmax = 20,  extent = [qx.min(), qx.max(), qx.min(), qx.max()])
    ax.set_title('fftn')
    plt.colorbar(im, ax = axs.ravel().tolist(), shrink=0.8, location = 'left')
    if file:
        if not file_format:
            file_format = 'png'
            print(slice_number)
        plt.savefig(file, format = file_format)
    else:
        plt.suptitle("Comparison of the Fourier Transforms at slice {sl} calculated a custom way\nby 2D slices + final 1D FT and the torch's fftn".format(sl = slice_number))
        plt.show()
    
    
def plot_Q_vs_I(binned_data,xlim = None, file = None, file_format = None):
    plt.figure(figsize=(8,3))
    plt.plot(binned_data.Q, binned_data.I)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('q')
    plt.ylabel('I')
    plt.title("rebinned Q vs I");
    if xlim:
        plt.xlim(xlim[0], xlim[1]) # e.g. (10**(-2), 10**(-1))
    if file:
        if not file_format:
            file_format = 'png'
        plt.savefig(file, format = file_format)
    else:
        plt.show()