#!/home/slaskina/.conda/envs/ma/bin/python

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data

import os
import h5py
import matplotlib.pyplot as plt
import seaborn
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from time import time

import FrEIA.framework  as Ff
import FrEIA.modules as Fm

from SAXSsimulations.SAXSinverse.utils import ScatteringProblemForward
from SAXSsimulations.SAXSinverse.visualizations import plot_outcomes_identified, describe_false_shapes, describe_positive_shapes


# look at the latent space
def predict_laten(lp, data_subset):
    x = lp.inputs_norm[data_subset].to(device)
    x = torch.cat((x, lp.add_pad_noise * torch.randn(1500, lp.ndim_pad_x).to(lp.device)), dim=1)
    out_y, _ = lp.model(x, jac  = True)
    return out_y[:,:2].cpu().detach()

def main():
    filename_out    = '../results/ML_output/feed_forward_resnet.pt'
    # Model to load and continue training. Ignored if empty string
    filename_in     = ''
    # Compute device to perform the training on, 'cuda' or 'cpu'
    device          = 'cuda'

    #######################
    #  Training schedule  #
    #######################

    # Initial learning rate
    lr_init         = 1.0e-3
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
    ndim_x_class     = 3
    ndim_x_features     = 7
    ndim_y     = 512

    ############
    #  Losses  #
    ############

    train_forward_mmd    = True

    lambd_mmd_for_class    = 10
    lambd_mmd_for_feature  = 200

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

    mmd_forw_kernels = [(0.2, 2), (1.5, 2), (3.0, 2)]
    mmd_back_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]
    mmd_back_weighted = True

    ###########
    #  Model  #
    ###########

    # Initialize the model parameters from a normal distribution with this sigma
    init_scale = 0.10
    #
    N_blocks   = 25
    #
    hidden_layer_sizes = 1024
    #
    use_permutation = True
    #
    verbose_construction = False
    #
    flow = 'forward'

    lp = ScatteringProblemForward( filename_out, 
                            device, 
                            lr_init, 
                            batch_size, 
                            n_epochs, 
                            n_its_per_epoch, 
                            pre_low_lr, 
                            final_decay, 
                            l2_weight_reg, 
                            adam_betas, 
                            ndim_y, 
                            lambd_mmd_for_class , 
                            lambd_mmd_for_feature, 
                            y_uncertainty_sigma,
                            mmd_forw_kernels, 
                            mmd_back_kernels, 
                            mmd_back_weighted, 
                            init_scale, 
                            flow)

    lp.read_data('../data/simulations',shapes = 3, input_keys = ['shape', 'radius', 'radius_pd','length', 'length_pd',  'volfraction'])
    lp.normalize_inputs()

    lp.create_loaders()

    class ForwardScatteringResNet(nn.Module):
        def __init__(self, N_blocks, ResNet_block) -> None:
            super().__init__()
            self.N_blocks = N_blocks
            self.resnet_blocks = ResNet_block
            self.res_block = nn.Sequential(
                nn.Conv1d(512, 512, kernel_size = 5,padding = 2, stride=1),
                nn.ELU(),
                nn.AvgPool1d(5, padding=2, stride=1),
            )
            self.s1 = nn.Sequential(
                nn.Conv1d(in_channels = 512, out_channels = 128 , kernel_size = 11, padding=5, stride=3),
                nn.ELU()
            )
            self.linear_out = nn.Sequential(    
                nn.Linear(128, len(lp.shapes_dict.keys())*2 + len(lp.input_features)-2),
                nn.ELU()
            )
            self.activation = nn.ELU()
            self.flatten = nn.Flatten()
        def forward(self, x):
            x = x.reshape(len(x), -1, 1)
            for _ in range(self.N_blocks):
                residual = x
                for _ in range(self.resnet_blocks):
                    x =  self.res_block(x)
                x += residual
                x = self.activation(x)

            x = self.s1(x)
            x = self.flatten(x)
            x = self.linear_out(x)
            return x.reshape(-1,len(lp.shapes_dict.keys())*2 + len(lp.input_features)-2)
        
    resnet_model = ForwardScatteringResNet(4,3)


    lp.set_model(resnet_model)
    lp.set_optimizer()

    lp.train()

    df_train = lp.create_table_from_outcomes_test(lp.make_prediction_test(lp.train_indices), lp.train_indices)
    df_val = lp.create_table_from_outcomes_test(lp.make_prediction_test(lp.val_indices), lp.val_indices) 
    df_test = lp.create_table_from_outcomes_test(lp.make_prediction_test(lp.test_indices), lp.test_indices)




    plot_outcomes_identified(df_test, 'Test', '../results/ff_test.png' )


if __name__ == '__main__':
    main()