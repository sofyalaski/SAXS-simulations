import numpy as np


def compute_error(d1, d2):   
    return np.max(np.abs(d1-d2) / np.mean((np.abs(d1), np.abs(d2)), axis=0))


def safe_multiplier(*args):
    product = 1
    for factor in args:
        if factor !=0:
            product *=factor
    return product 


def safe_dividend( *args):
    fraction = args[0]
    for factor in args[1:]:
        if factor !=0:
            fraction /=factor
    return fraction 

    
def Intensity_func(scale, simulation):
    if 'binned_slice' in dir(simulation):
        return simulation.binned_slice.I.values - simulation.I_sas*scale
    else:
        return simulation.binned_data.I.values - simulation.I_sas*scale
