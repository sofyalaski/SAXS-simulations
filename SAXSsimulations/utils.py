import numpy as np


def compute_error(d1, d2):   
    return np.max(np.abs(d1-d2) / np.mean((np.abs(d1), np.abs(d2)), axis=0))
    
def Intensity_func(scale, simulation):
    if simulation.shape == 'sphere':
        if 'binned_slice' in dir(simulation):
            return simulation.binned_slice.I.values - simulation.I_sas*scale
        else:
            return simulation.binned_data.I.values - simulation.I_sas*scale
    else:
        values =  simulation.binned_slice.I - (simulation.I_sas*scale).flatten()
        return values[~np.isnan(values)]

