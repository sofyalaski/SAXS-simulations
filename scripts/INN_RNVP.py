#!/home/slaskina/.conda/envs/ma/bin/python

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data

import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from time import time

import FrEIA.framework  as Ff
import FrEIA.modules as Fm

from SAXSsimulations.SAXSinverse.utils import ScatteringProblem
from SAXSsimulations.SAXSinverse.visualizations import plot_outcomes_identified, describe_false_shapes, describe_positive_shapes


def main():

    # this parameters work when I/100
    filename_out    = '../results/ML_output/inn.pt'
    # Compute device to perform the training on, 'cuda' or 'cpu'
    device          = 'cuda'

    #######################
    #  Training schedule  #
    #######################

    # Initial learning rate
    lr_init         = 2e-3
    #Batch size
    batch_size      = 128
    # Total number of epochs to train for
    n_epochs        = 200

    # End the epoch after this many iterations (or when the train loader is exhausted)
    n_its_per_epoch = 200
    # For the first n epochs, train with a much lower learning rate. This can be
    # helpful if the model immediately explodes.
    pre_low_lr      = 0
    # Decay exponentially each epoch, to final_decay*lr_init at the last epoch.
    final_decay     = 0.02
    # L2 weight regularization of model parameters
    l2_weight_reg   = 1e-5
    # Parameters beta1, beta2 of the Adam optimizer
    adam_betas = (0.9, 0.95)

    #####################
    #  Data dimensions  #
    #####################
    ndim_pad_x = 522

    ndim_y     = 512
    ndim_z     = 2
    ndim_pad_zy = 18 


    ############
    #  Losses  #
    ############

    train_reconstruction = False


    lambd_fit_forw         = 0.01 
    lambd_mmd_forw         = 100
    lambd_mmd_back         = 500
    lambd_reconstruct      = 1.

    # Both for fitting, and for the reconstruction, perturb y with Gaussian 
    # noise of this sigma
    add_y_noise     = 0 
    # For reconstruction, perturb z 
    add_z_noise     = 2e-2
    # In all cases, perturb the zero padding
    add_pad_noise   = 1e-2

    # For noisy forward processes, the sigma on y (assumed equal in all dimensions).
    # This is only used if mmd_back_weighted of train_max_likelihoiod are True.
    y_uncertainty_sigma = 0.12 * 4


    mmd_forw_kernels = [(0.2, 1/2), (1.5, 1/2), (3.0, 1/2)]
    mmd_back_kernels = [(0.2, 1/2), (0.2, 1/2), (0.2, 1/2)]
    mmd_back_weighted = True

    ###########
    #  Model  #
    ###########

    # Initialize the model parameters from a normal distribution with this sigma
    init_scale = 0.10
    #
    N_blocks   = 5
    #
    exponent_clamping = 2
    #
    hidden_layer_sizes = 32
    #
    use_permutation = True
    #
    verbose_construction = False
    #
    flow = 'inverse'


    lp = ScatteringProblem(filename_out, 
                            device, 
                            lr_init, 
                            batch_size, 
                            n_epochs, 
                            n_its_per_epoch, 
                            pre_low_lr, 
                            final_decay, 
                            l2_weight_reg, 
                            adam_betas, 
                            ndim_pad_x, 
                            ndim_y, 
                            ndim_z, 
                            ndim_pad_zy , 
                            train_reconstruction, 
                            lambd_fit_forw , 
                            lambd_mmd_forw, 
                            lambd_mmd_back, 
                            lambd_reconstruct, 
                            add_y_noise , 
                            add_z_noise, 
                            add_pad_noise, 
                            y_uncertainty_sigma,
                            mmd_forw_kernels, 
                            mmd_back_kernels, 
                            mmd_back_weighted, 
                            init_scale, 
                            flow)

    lp.read_data('../data/simulations',shapes = 3, input_keys = ['shape', 'radius', 'radius_pd','length', 'length_pd',  'volfraction'])
    lp.normalize_inputs()

    lp.create_loaders()

    def subnet(dims_in, dims_out):
        return nn.Sequential(nn.Linear(dims_in, hidden_layer_sizes*2), nn.ReLU(),
                            nn.Linear(hidden_layer_sizes*2,  hidden_layer_sizes), nn.ReLU(),
                            nn.Linear(hidden_layer_sizes,  dims_out))

    input = Ff.InputNode(lp.ndim_x + ndim_pad_x, name='input')


    nodes = [input]
    for i in range(N_blocks):
        nodes.append(Ff.Node(nodes[-1].out0, Fm.RNVPCouplingBlock, {'subnet_constructor':subnet, 'clamp':exponent_clamping}, name = 'coupling_{}'.format(i)))
        if use_permutation:
            nodes.append(Ff.Node([nodes[-1].out0], Fm.PermuteRandom, {'seed':i}, name='permute_{}'.format(i)))

    nodes.append(Ff.OutputNode([nodes[-1].out0], name='output'))
    model = Ff.GraphINN(nodes, verbose=verbose_construction)

    lp.set_model(model)
    lp.set_optimizer()

    lp.train()

    df_test = lp.create_table_from_outcomes_test(lp.make_prediction_test(lp.test_indices), lp.test_indices)



    latent_v= lp.predict_latent(lp.test_indices)
    a = pd.DataFrame({'latent space': 0, 'value' : latent_v[:,0]})
    b = pd.DataFrame({'latent space': 1, 'value' : latent_v[:,1]})
    latent = pd.concat((a,b))
    sns.displot(data=latent, x='value', hue = 'latent space', kde=True, height = 4, aspect = 1.6)
    plt.savefig('../results/RNVP_latent.png')

    plot_outcomes_identified(df_test, 'Test', filename_out.split('.pt')[0]+'_shapes.txt', '../results/RNVP_test.png' )


if __name__ == '__main__':
    main()