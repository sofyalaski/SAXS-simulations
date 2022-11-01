from time import time

import torch
from torch.autograd import Variable

from FrEIA.framework import *
from FrEIA.modules import *

import config as c

import losses
import model
import monitoring
import numpy as np
assert c.train_loader and c.test_loader, "No data loaders supplied"

def noise_batch(ndim):
    return torch.randn(c.batch_size, ndim).to(c.device)

def loss_max_likelihood(out, y):
    jac = model.model.jacobian(run_forward=False)

    neg_log_likeli = ( 0.5 / c.y_uncertainty_sigma**2 * torch.sum((out[:, -ndim_y:]       - y[:, -ndim_y:])**2, 1)
                     + 0.5 / c.zeros_noise_scale**2   * torch.sum((out[:, ndim_z:-ndim_y] - y[:, ndim_z:-ndim_y])**2, 1)
                     + 0.5 * torch.sum(out[:, :ndim_z]**2, 1)
                     - jac)

    return c.lambd_max_likelihood * torch.mean(neg_log_likeli)

def loss_forward_mmd(out, y):
    # Shorten output, and remove gradients wrt y, for latent loss
    # [z,y] only
    output_block_grad = torch.cat((out[:, :c.ndim_z],
                                   out[:, -c.ndim_y:].data), dim=1) 
    y_short = torch.cat((y[:, :c.ndim_z], y[:, -c.ndim_y:]), dim=1)

    l_forw_fit = c.lambd_fit_forw * losses.l2_fit(out[:, c.ndim_z:], y[:, c.ndim_z:])
    l_forw_mmd = c.lambd_mmd_forw  * torch.mean(losses.forward_mmd(output_block_grad, y_short))

    return l_forw_fit, l_forw_mmd

def loss_backward_mmd(x_class, x_features, y):
    [x_samples_class, x_samples_features], x_samples_jac = model.model(y, rev=True) 
    MMD = losses.backward_mmd(x_class, x_samples_class) + losses.backward_mmd(x_features, x_samples_features)
    if c.mmd_back_weighted:
        MMD *= torch.exp(- 0.5 / c.y_uncertainty_sigma**2 * losses.l2_dist_matrix(y, y))
    return c.lambd_mmd_back * torch.mean(MMD)

def loss_reconstruction(out_y, x):
    cat_inputs = [out_y[:, :c.ndim_z] + c.add_z_noise * noise_batch(c.ndim_z)] # list with 1 tensor
    
    if c.ndim_pad_zy:
        cat_inputs.append(out_y[:, c.ndim_z:-c.ndim_y] + c.add_pad_noise * noise_batch(c.ndim_pad_zy)) # list with 2 tensor
    cat_inputs.append(out_y[:, -c.ndim_y:] + c.add_y_noise * noise_batch(c.ndim_y)) # list with 3 tensors
    x_reconstructed_class, x_reconstructed_features, x_reconstructed_jac = model.model(torch.cat(cat_inputs, 1), rev=True) # concatenate list elements along axis 1
    return c.lambd_reconstruct * losses.l2_fit(x_reconstructed, x) # needs fix

def train_epoch(i_epoch, test=False):
    if not test:
        model.model.train()
        loader = c.train_loader

    if test:
        model.model.eval()
        loader = c.test_loader
        #nograd = torch.no_grad()
        #nograd.__enter__()


    batch_idx = 0
    loss_history = []

    for x, y in loader:

        if batch_idx > c.n_its_per_epoch:
            break
        batch_losses = []

        batch_idx += 1

        #x, y = Variable(x).to(c.device), Variable(y).to(c.device)
        x, y = x.to(c.device), y.to(c.device)

        if c.add_y_noise > 0:
            y += c.add_y_noise * noise_batch(c.ndim_y)

        if c.ndim_pad_x_class:
            # x = [x, pad_x]
            x_class = torch.cat((x[:,:3], c.add_pad_noise * noise_batch(c.ndim_pad_x_class)), dim=1)
        
        if c.ndim_pad_x_features:
            # x = [x, pad_x]
            x_features = torch.cat((x[:,3:], c.add_pad_noise * noise_batch(c.ndim_pad_x_features)), dim=1)
        if c.ndim_pad_zy:
            y = torch.cat((c.add_pad_noise * noise_batch(c.ndim_pad_zy), y), dim=1)
        # y = [z, pad_yz, y]
        y = torch.cat((noise_batch(c.ndim_z), y), dim=1)

        out_y, out_y_jac = model.model([x_class,x_features])
        # tuple with output[0] and jacobian[1]
        if c.train_max_likelihood:
            batch_losses.append(loss_max_likelihood(out_y, y))

        if c.train_forward_mmd:
            batch_losses.extend(loss_forward_mmd(out_y, y))

        if c.train_backward_mmd:
            batch_losses.append(loss_backward_mmd(x_class,x_features, y))

        if c.train_reconstruction:
            batch_losses.append(loss_reconstruction(out_y.data, x))

        l_total = sum(batch_losses)
        loss_history.append([l.item() for l in batch_losses]) # lisr of lists: list for each batch

        if not test:
            l_total.backward()
            model.optim_step()

    if test:
        monitoring.show_hist(out_y[:, :c.ndim_z])
        monitoring.show_cov(out_y[:, :c.ndim_z])

        if c.test_time_functions:
            out_x_class, out_x_features, out_x_jac = model.model(y, rev=True) 
            for f in c.test_time_functions:
                f(out_x_class, out_x_features, out_y, x_class,x_features, y)

        #nograd.__exit__(None, None, None)
    return np.mean(loss_history, axis=0)

def main():
    monitoring.restart()

    try:
        monitoring.print_config()
        t_start = time()
        for i_epoch in range(-c.pre_low_lr, c.n_epochs):

            if i_epoch < 0:
                for param_group in model.optim.param_groups:
                    param_group['lr'] = c.lr_init * 1e-1

            train_losses = train_epoch(i_epoch) # mean over batches
            test_losses  = train_epoch(i_epoch, test=True)
            t = np.concatenate([train_losses, test_losses])
            monitoring.show_loss(t)
            model.scheduler_step() 

    except:
        model.save(c.filename_out + '_ABORT')
        raise

    finally:
        print("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.))
        model.save(c.filename_out)

if __name__ == "__main__":
    main()