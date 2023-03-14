import numpy as np


def compute_error(d1, d2):   
    """
    compute relative error
    Arguments:
        d1 (float) first value
        d2 (float) second value
    Returns:
        float: relative error
    """
    return np.nanmax(np.abs(d1-d2) / np.mean((np.abs(d1), np.abs(d2)), axis=0))


def tensor_informative( tensor):
    """
    Tests if any element in input evaluates to True.
    Arguments:
        tensor(torch.tensor)
    Returns:
        bool:  any element in input evaluates to True
    """
    return tensor.any()  


def Intensity_func(scale, simulation):
    """
    The function to compare Intensity of the 3D simulation and SAS-generated one.
    Arguments:
        scale (flaot): the scale of the intensity to be multiplied with SAS-gnerated intensity
        simulation (Simulation instance): the instance of class simulation
    Returns: 
        np.array difference between intensity vlues in simulated simulation and SAS-generated one
    """
    if simulation.shape == 'sphere':
        if 'binned_slice' in dir(simulation):
            return simulation.binned_slice.I.values - simulation.I_sas*scale
        else:
            return simulation.binned_data.I.values - simulation.I_sas*scale
    else:
        values =  simulation.binned_slice.I.dropna() - (simulation.I_sas[~np.isnan(simulation.I_sas)]*scale).flatten()
        return values[~np.isnan(values)]

