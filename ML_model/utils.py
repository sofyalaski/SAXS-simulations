import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import os
import h5py
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from time import time

import FrEIA.framework  as Ff
import FrEIA.modules as Fm

import losses
import monitoring
from sklearn.preprocessing import StandardScaler



class ScatteringProblemSplit:
    def __init__(self, device,  batch_size, ndim_x_class, ndim_x_features, ndim_y, ndim_z, ndim_pad_x_class, ndim_pad_x_features,
                       ndim_pad_zy, init_scale, final_decay, n_epochs, lr_init, adam_betas, l2_weight_reg, lambd_fit_forw, 
                       lambd_mmd_forw, lambd_mmd_back_class, lambd_mmd_back_feature, lambd_reconstruct, mmd_back_weighted, 
                       y_uncertainty_sigma, add_pad_noise, add_z_noise, add_y_noise, n_its_per_epoch, pre_low_lr, filename_out, mmd_forw_kernels, mmd_back_kernels):
        self.device = device
        self.init_scale = init_scale
        self.final_decay = final_decay
        self.n_epochs = n_epochs
        self.lr_init = lr_init
        self.adam_betas = adam_betas
        self.l2_weight_reg = l2_weight_reg
        self.batch_size = batch_size
        self.ndim_x_class = ndim_x_class
        self.ndim_x_features = ndim_x_features
        self.ndim_y = ndim_y
        self.ndim_z = ndim_z 
        self.ndim_pad_x_class = ndim_pad_x_class
        self.ndim_pad_x_features =  ndim_pad_x_features
        self.ndim_pad_zy = ndim_pad_zy
        self.lambd_fit_forw = lambd_fit_forw
        self.lambd_mmd_forw = lambd_mmd_forw
        self.lambd_mmd_back_class = lambd_mmd_back_class
        self.lambd_mmd_back_feature = lambd_mmd_back_feature
        self.lambd_reconstruct = lambd_reconstruct
        self.mmd_back_weighted = mmd_back_weighted
        self.y_uncertainty_sigma = y_uncertainty_sigma
        self.add_pad_noise = add_pad_noise
        self.add_z_noise = add_z_noise
        self.add_y_noise = add_y_noise
        self.n_its_per_epoch = n_its_per_epoch
        self.pre_low_lr = pre_low_lr
        self.filename_out = filename_out
        self.mmd_forw_kernels = mmd_forw_kernels
        self.mmd_back_kernels = mmd_back_kernels

    def update_hyperparameters(self, device,  batch_size, ndim_x_class, ndim_x_features, ndim_y, ndim_z, ndim_pad_x_class, ndim_pad_x_features,
                       ndim_pad_zy, init_scale, final_decay, n_epochs, lr_init, adam_betas, l2_weight_reg, lambd_fit_forw, 
                       lambd_mmd_forw, lambd_mmd_back_class, lambd_mmd_back_feature, lambd_reconstruct, mmd_back_weighted, 
                       y_uncertainty_sigma, add_pad_noise, add_z_noise, add_y_noise, n_its_per_epoch, pre_low_lr, filename_out, mmd_forw_kernels, mmd_back_kernels):
        self.device = device
        self.init_scale = init_scale
        self.final_decay = final_decay
        self.n_epochs = n_epochs
        self.lr_init = lr_init
        self.adam_betas = adam_betas
        self.l2_weight_reg = l2_weight_reg
        self.batch_size = batch_size
        self.ndim_x_class = ndim_x_class
        self.ndim_x_features = ndim_x_features
        self.ndim_y = ndim_y
        self.ndim_z = ndim_z 
        self.ndim_pad_x_class = ndim_pad_x_class
        self.ndim_pad_x_features =  ndim_pad_x_features
        self.ndim_pad_zy = ndim_pad_zy
        self.lambd_fit_forw = lambd_fit_forw
        self.lambd_mmd_forw = lambd_mmd_forw
        self.lambd_mmd_back_class = lambd_mmd_back_class
        self.lambd_mmd_back_feature = lambd_mmd_back_feature
        self.lambd_reconstruct = lambd_reconstruct
        self.mmd_back_weighted = mmd_back_weighted
        self.y_uncertainty_sigma = y_uncertainty_sigma
        self.add_pad_noise = add_pad_noise
        self.add_z_noise = add_z_noise
        self.add_y_noise = add_y_noise
        self.n_its_per_epoch = n_its_per_epoch
        self.pre_low_lr = pre_low_lr
        self.filename_out = filename_out
        self.mmd_forw_kernels = mmd_forw_kernels
        self.mmd_back_kernels = mmd_back_kernels

    def shape_condition(self,true_shape):
        a,b=1,1
        if true_shape=='hardsphere':
            while a+b>=1:
                b=np.random.normal(0.8, 0.2)
                if b<1:
                    a = np.random.uniform(0, (1-b)/2)
            c = 1-a-b    
            return torch.tensor([a,b,c])
        else:
            # if sphere or cylinder probs are calculated similarly
            while a+b>=1:
                a,b=np.random.normal(0.55, 0.1),np.random.normal(0.05, 0.05)
            c = 1-a-b
            if true_shape=='sphere':
                return torch.tensor([a,b,c])  
            else:
                return torch.tensor([c,b,a])    

    def read_data(self, path, conditioned = False):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
        self.labels = torch.zeros(len(files), self.ndim_y, dtype = torch.float32)
        try:
            self.inputs = torch.zeros(len(files), self.ndim_x_class+self.ndim_x_features, dtype = torch.float32)
        except AttributeError:
            self.inputs = torch.zeros(len(files), self.ndim_x, dtype = torch.float32)
        if conditioned:
            conditions = torch.zeros(len(files), 3, dtype = torch.float32)
        input_keys = ['shape', 'radius', 'radius_pd','length', 'length_pd',  'volfraction']
        for i,f in enumerate(files):
            with  h5py.File(os.path.join(path,f),'r') as file:
                print(file, end = '\r')
                self.labels[i,:] = torch.from_numpy(file['entry/I'][()].flatten())#I_noisy
                for i_k, key in enumerate(input_keys):
                    try:
                        if key == 'shape':
                            shape = file['properties'][key][()].decode("utf-8")
                            if shape == 'sphere':
                                self.inputs[i, i_k:i_k+3] = torch.tensor([1,0,0]) 
                                #conditions[i, i+k]
                            elif shape == 'hardsphere':
                                self.inputs[i, i_k:i_k+3] = torch.tensor([0,1,0])
                            else:
                                self.inputs[i, i_k:i_k+3] = torch.tensor([0,0,1])
                            if conditioned:
                                conditions[i, i_k:i_k+3] = self.shape_condition(shape)
                        elif key =='radius':
                            if shape == 'sphere':
                                self.inputs[i, i_k+2:i_k+5] = torch.tensor([file['properties'][key][()],0,0]) 
                            elif shape == 'hardsphere':
                                self.inputs[i, i_k+2:i_k+5] = torch.tensor([0,file['properties'][key][()],0])
                            else:
                                self.inputs[i, i_k+2:i_k+5] = torch.tensor([0,0,file['properties'][key][()]])
                        else:
                            self.inputs[i, i_k+4] = file['properties'][key][()]
                    except KeyError:
                        # spheres don't have all of the properties a cylinder does
                        pass
        if conditioned:
            self.inputs = torch.cat((self.inputs,conditions),1)

    def normalize_inputs(self):
        self.scaler = StandardScaler()
        self.scaler.fit(self.inputs[:,3:])
        self.inputs_norm = self.scaler.transform(self.inputs[:,3:])
        self.inputs_norm = torch.concatenate((self.inputs[:,:3],torch.from_numpy(self.inputs_norm) ), axis=1).type(torch.float32)


    def create_loaders(self):
        # Creating data indices for training and validation splits:
        dataset_size = len(self.labels)
        indices = list(range(dataset_size))
        test_size = 0.1
        val_size = 0.2
        test_split = int(np.floor(test_size * dataset_size))
        val_split = int(np.floor(val_size * dataset_size))
        np.random.seed(1234)
        np.random.shuffle(indices)
        self.train_indices, self.val_indices, self.test_indices = indices[:dataset_size-test_split-val_split], indices[dataset_size-test_split-val_split:dataset_size-test_split], indices[dataset_size-test_split:]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(self.train_indices)
        val_sampler = SubsetRandomSampler(self.val_indices)

        try:
            self.test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(self.inputs_norm, self.labels), batch_size=self.batch_size, drop_last=True, sampler = val_sampler)

            self.train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(self.inputs_norm, self.labels), batch_size=self.batch_size, drop_last=True, sampler = train_sampler)
        except AttributeError:
            self.test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(self.inputs, self.labels), batch_size=self.batch_size, drop_last=True, sampler = val_sampler)

            self.train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(self.inputs, self.labels), batch_size=self.batch_size, drop_last=True, sampler = train_sampler)

    def set_model(self, model):
        self.model = model
        self.model.to(self.device)


    def set_optimizer(self):
        self.params_trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        for p in self.params_trainable:
            p.data = self.init_scale * torch.randn(p.data.shape).to(self.device)

        gamma = (self.final_decay)**(1./self.n_epochs)
        self.optim = torch.optim.Adam(self.params_trainable, lr=self.lr_init, betas=self.adam_betas, eps=1e-6, weight_decay=self.l2_weight_reg)
        self.weight_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=gamma)


    def optim_step(self):
        #for p in params_trainable:
            #print(torch.mean(torch.abs(p.grad.data)).item())
        self.optim.step()
        self.optim.zero_grad()

    def scheduler_step(self):
        #self.weight_scheduler.step()
        pass

    def save(self,name):
        torch.save({'opt':self.optim.state_dict(),
                    'net':self.model.state_dict()}, name)

    def load(self,name):
        state_dicts = torch.load(name)
        self.model.load_state_dict(state_dicts['net'])
        try:
            self.optim.load_state_dict(state_dicts['opt'])
        except ValueError:
            print('Cannot load optimizer for some reason or other')

    def noise_batch(self,ndim):
        return torch.randn(self.batch_size, ndim).to(self.device)

    def loss_forward_mmd(self, out, y):
        # Shorten output, and remove gradients wrt y, for latent loss
        # [z,y] only
        output_block_grad = torch.cat((out[:, :self.ndim_z],
                                    out[:, -self.ndim_y:].data), dim=1) 
        y_short = torch.cat((y[:, :self.ndim_z], y[:, -self.ndim_y:]), dim=1)

        l_forw_fit = self.lambd_fit_forw * losses.l2_fit(out[:, self.ndim_z:], y[:, self.ndim_z:], self.batch_size)
        l_forw_mmd = self.lambd_mmd_forw  * torch.mean(losses.forward_mmd(output_block_grad, y_short, self.mmd_forw_kernels, self.device))

        return l_forw_fit, l_forw_mmd


    def loss_backward_mmd(self,x_class, x_features, y):
        [x_samples_class, x_samples_features], x_samples_jac = self.model(y, rev=True, jac = True) 
        MMD_class = losses.backward_mmd(x_class, x_samples_class, self.mmd_back_kernels, self.device) 
        MMD_features = losses.backward_mmd(x_features, x_samples_features, self.mmd_back_kernels, self.device)
        if self.mmd_back_weighted:
            MMD_class *= torch.exp(- 0.5 / self.y_uncertainty_sigma**2 * losses.l2_dist_matrix(y, y))
            MMD_features *= torch.exp(- 0.5 / self.y_uncertainty_sigma**2 * losses.l2_dist_matrix(y, y))
        return self.lambd_mmd_back_class * torch.mean(MMD_class)+self.lambd_mmd_back_feature * torch.mean(MMD_features)


    def loss_reconstruction(self, out_y, x_class, x_features):
        cat_inputs = [out_y[:, :self.ndim_z] + self.add_z_noise * self.noise_batch(self.ndim_z)] # list with 1 tensor
        
        if self.ndim_pad_zy:
            cat_inputs.append(out_y[:, self.ndim_z:-self.ndim_y] + self.add_pad_noise * self.noise_batch(self.ndim_pad_zy)) # list with 2 tensor
        cat_inputs.append(out_y[:, -self.ndim_y:] + self.add_y_noise * self.noise_batch(self.ndim_y)) # list with 3 tensors
        [x_reconstructed_class, x_reconstructed_features], x_reconstructed_jac = self.model(torch.cat(cat_inputs, 1), rev=True, jac = True) # concatenate list elements along axis 1
        return self.lambd_reconstruct * (losses.l2_fit(x_reconstructed_class[:, :self.ndim_pad_x_class], x_class[:,:self.ndim_pad_x_class], self.batch_size) + losses.l2_fit(x_reconstructed_features[:,:self.ndim_pad_x_features], x_features[:,:self.ndim_pad_x_features], self.batch_size) )

    def train_epoch(self,i_epoch, test=False):
        if not test:
            self.model.train()
            loader = self.train_loader

        if test:
            self.model.eval()
            loader = self.test_loader
            nograd = torch.no_grad()
            nograd.__enter__()


        batch_idx = 0
        loss_history = []

        for x, y in loader:

            if batch_idx > self.n_its_per_epoch:
                break
            batch_losses = []

            batch_idx += 1

            x, y = x.to(self.device), y.to(self.device)

            if self.add_y_noise > 0:
                y += self.add_y_noise * self.noise_batch(self.ndim_y)
            if self.ndim_pad_x_class:
                x_class = torch.cat((x[:,:3], self.add_pad_noise * self.noise_batch(self.ndim_pad_x_class)), dim=1)
            
            if self.ndim_pad_x_features:
                x_features = torch.cat((x[:,3:], self.add_pad_noise * self.noise_batch(self.ndim_pad_x_features)), dim=1)
            if self.ndim_pad_zy:
                y = torch.cat((self.add_pad_noise * self.noise_batch(self.ndim_pad_zy), y), dim=1)
            y = torch.cat((self.noise_batch(self.ndim_z), y), dim=1)

            out_y, out_y_jac = self.model([x_class,x_features], jac  = True)
            # tuple with output[0] and jacobian[1]

            batch_losses.extend(self.loss_forward_mmd(out_y, y))
            batch_losses.append(self.loss_backward_mmd(x_class,x_features, y))
            batch_losses.append(self.loss_reconstruction(out_y.data, x_class, x_features))

            l_total = sum(batch_losses)
            loss_history.append([l.item() for l in batch_losses]) # lisr of lists: list for each batch

            if not test:
                l_total.backward()
                self.optim_step()

        if test:
            monitoring.show_hist(out_y[:, :self.ndim_z])
            monitoring.show_cov(out_y[:, :self.ndim_z])
            nograd.__exit__(None, None, None)
        return np.mean(loss_history, axis=0)

    def train(self):
        monitoring.restart()

        try:
            t_start = time()
            for i_epoch in range(-self.pre_low_lr, self.n_epochs):

                if i_epoch < 0:
                    for param_group in self.optim.param_groups:
                        param_group['lr'] = self.lr_init * 1e-1

                train_losses = self.train_epoch(i_epoch) # mean over batches
                test_losses  = self.train_epoch(i_epoch, test=True)
                t = np.concatenate([train_losses, test_losses])
                monitoring.show_loss(t)
                self.scheduler_step() 

        except:
            self.save(self.filename_out + '_ABORT')
            raise

        finally:
            print("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.))
            self.save(self.filename_out)


    def make_prediction(self, data_subset, cond = None):
        y = torch.cat((self.add_pad_noise * torch.randn( len(self.labels[data_subset]), self.ndim_pad_zy).to(self.device), self.labels[data_subset].to(self.device)), dim=1)
        y = torch.cat((torch.randn( len(self.labels[data_subset]), self.ndim_z).to(self.device), y), dim=1)
        if cond is None:
            [pred_class, pred_features], _ = self.model(y, rev = True)  # output of NN without padding
        else:
            [pred_class, pred_features], _ = self.model(y, rev = True, c=cond.to(self.device))  # output of NN without padding
        predictions =  np.concatenate((pred_class.cpu().detach()[:,:self.ndim_x_class], pred_features.cpu().detach()[:, :self.ndim_x_features]), axis=1)
        return np.concatenate((predictions[:,:3],self.scaler.inverse_transform(predictions[:,3:])), axis=1)

    def create_table_from_outcomes(self, pred, data_subset):
        try:
            sampled_inputs = self.inputs_norm[data_subset]
        except AttributeError:
            sampled_inputs = self.inputs[data_subset]
        df = pd.DataFrame(columns = ['true_shape', 'pred_shape', 'radius','pred_radius'], index = [])
        df['true_shape'] = sampled_inputs[:,:3].argmax(axis=1)
        df['pred_shape'] = pred[:,:3].argmax(axis=1)
        df['radius'] = np.take_along_axis(sampled_inputs[:,3:6],df.true_shape.values.reshape(-1,1), axis=1)
        df['pred_radius'] = np.take_along_axis(pred[:,3:6],df.pred_shape.values.reshape(-1,1), axis=1)
        df['radius_pd'] = sampled_inputs[:,6]
        df['pred_radius_pd'] = pred[:,6]
        df['length'] = sampled_inputs[:,7]
        df['pred_length'] = pred[:,7] # only those identified as cylinder should have lengh and pd_length  df.pred_shape.values ==2, pred[:,7], 0) 
        df['length_pd'] = sampled_inputs[:,8]
        df['pred_length_pd'] =  pred[:,8]
        df['volfraction'] = sampled_inputs[:,9]
        df['pred_volfraction'] = pred[:,9]
        return df





class ScatteringProblem(ScatteringProblemSplit):
    def __init__(self, device, batch_size, ndim_x, ndim_y, ndim_z, ndim_pad_x, ndim_pad_zy, init_scale, final_decay, n_epochs, lr_init, adam_betas, l2_weight_reg, lambd_fit_forw, lambd_mmd_forw, lambd_mmd_back, lambd_reconstruct, mmd_back_weighted, y_uncertainty_sigma, add_pad_noise, add_z_noise, add_y_noise, n_its_per_epoch, pre_low_lr, filename_out, mmd_forw_kernels, mmd_back_kernels):
        self.device = device
        self.init_scale = init_scale
        self.final_decay = final_decay
        self.n_epochs = n_epochs
        self.lr_init = lr_init
        self.adam_betas = adam_betas
        self.l2_weight_reg = l2_weight_reg
        self.batch_size = batch_size
        self.ndim_x = ndim_x
        self.ndim_y = ndim_y
        self.ndim_z = ndim_z 
        self.ndim_pad_x = ndim_pad_x
        self.ndim_pad_zy = ndim_pad_zy
        self.lambd_fit_forw = lambd_fit_forw
        self.lambd_mmd_forw = lambd_mmd_forw
        self.lambd_mmd_back = lambd_mmd_back
        self.lambd_reconstruct = lambd_reconstruct
        self.mmd_back_weighted = mmd_back_weighted
        self.y_uncertainty_sigma = y_uncertainty_sigma
        self.add_pad_noise = add_pad_noise
        self.add_z_noise = add_z_noise
        self.add_y_noise = add_y_noise
        self.n_its_per_epoch = n_its_per_epoch
        self.pre_low_lr = pre_low_lr
        self.filename_out = filename_out
        self.mmd_forw_kernels = mmd_forw_kernels
        self.mmd_back_kernels = mmd_back_kernels

    def update_hyperparameters(self, device, batch_size, ndim_x, ndim_y, ndim_z, ndim_pad_x, ndim_pad_zy, init_scale, final_decay, n_epochs, lr_init, adam_betas, l2_weight_reg, lambd_fit_forw, lambd_mmd_forw, lambd_mmd_back, lambd_reconstruct, mmd_back_weighted, y_uncertainty_sigma, add_pad_noise, add_z_noise, add_y_noise, n_its_per_epoch, pre_low_lr, filename_out, mmd_forw_kernels, mmd_back_kernels):
        self.device = device
        self.init_scale = init_scale
        self.final_decay = final_decay
        self.n_epochs = n_epochs
        self.lr_init = lr_init
        self.adam_betas = adam_betas
        self.l2_weight_reg = l2_weight_reg
        self.batch_size = batch_size
        self.ndim_x = ndim_x
        self.ndim_y = ndim_y
        self.ndim_z = ndim_z 
        self.ndim_pad_x = ndim_pad_x
        self.ndim_pad_zy = ndim_pad_zy
        self.lambd_fit_forw = lambd_fit_forw
        self.lambd_mmd_forw = lambd_mmd_forw
        self.lambd_mmd_back = lambd_mmd_back
        self.lambd_reconstruct = lambd_reconstruct
        self.mmd_back_weighted = mmd_back_weighted
        self.y_uncertainty_sigma = y_uncertainty_sigma
        self.add_pad_noise = add_pad_noise
        self.add_z_noise = add_z_noise
        self.add_y_noise = add_y_noise
        self.n_its_per_epoch = n_its_per_epoch
        self.pre_low_lr = pre_low_lr
        self.filename_out = filename_out
        self.mmd_forw_kernels = mmd_forw_kernels
        self.mmd_back_kernels = mmd_back_kernels

    def loss_backward_mmd(self,x, y):
        x_samples, x_samples_jac = self.model(y, rev=True, jac = True) 
        MMD = losses.backward_mmd(x, x_samples, self.mmd_back_kernels, self.device) 
        if self.mmd_back_weighted:
            MMD *= torch.exp(- 0.5 / self.y_uncertainty_sigma**2 * losses.l2_dist_matrix(y, y))
        return self.lambd_mmd_back * torch.mean(MMD)


    def loss_reconstruction(self, out_y, x):
        cat_inputs = [out_y[:, :self.ndim_z] + self.add_z_noise * self.noise_batch(self.ndim_z)] # list with 1 tensor
        
        if self.ndim_pad_zy:
            cat_inputs.append(out_y[:, self.ndim_z:-self.ndim_y] + self.add_pad_noise * self.noise_batch(self.ndim_pad_zy)) # list with 2 tensor
        cat_inputs.append(out_y[:, -self.ndim_y:] + self.add_y_noise * self.noise_batch(self.ndim_y)) # list with 3 tensors
        x_reconstructed, x_reconstructed_jac = self.model(torch.cat(cat_inputs, 1), rev=True, jac = True) # concatenate list elements along axis 1
        return self.lambd_reconstruct * losses.l2_fit(x_reconstructed[:, :self.ndim_pad_x], x[:,:self.ndim_pad_x], self.batch_size)


    def train_epoch(self,i_epoch, test=False):
        if not test:
            self.model.train()
            loader = self.train_loader

        if test:
            self.model.eval()
            loader = self.test_loader
            nograd = torch.no_grad()
            nograd.__enter__()


        batch_idx = 0
        loss_history = []

        for x, y in loader:

            if batch_idx > self.n_its_per_epoch:
                break
            batch_losses = []

            batch_idx += 1

            x, y = x.to(self.device), y.to(self.device)

            if self.add_y_noise > 0:
                y += self.add_y_noise * self.noise_batch(self.ndim_y)
            if self.ndim_pad_x:
                x = torch.cat((x, self.add_pad_noise * self.noise_batch(self.ndim_pad_x)), dim=1)
        
            if self.ndim_pad_zy:
                y = torch.cat((self.add_pad_noise * self.noise_batch(self.ndim_pad_zy), y), dim=1)
            y = torch.cat((self.noise_batch(self.ndim_z), y), dim=1)

            out_y, out_y_jac = self.model(x, jac  = True)
            # tuple with output[0] and jacobian[1]

            batch_losses.extend(self.loss_forward_mmd(out_y, y))
            batch_losses.append(self.loss_backward_mmd(x, y))
            batch_losses.append(self.loss_reconstruction(out_y.data, x))

            l_total = sum(batch_losses)
            loss_history.append([l.item() for l in batch_losses]) # lisr of lists: list for each batch

            if not test:
                l_total.backward()
                self.optim_step()

        if test:
            monitoring.show_hist(out_y[:, :self.ndim_z])
            monitoring.show_cov(out_y[:, :self.ndim_z])
            nograd.__exit__(None, None, None)
        return np.mean(loss_history, axis=0)

    def make_prediction(self, data_subset, cond = None):
        y = torch.cat((self.add_pad_noise * torch.randn( len(self.labels[data_subset]), self.ndim_pad_zy).to(self.device), self.labels[data_subset].to(self.device)), dim=1)
        y = torch.cat((torch.randn( len(self.labels[data_subset]), self.ndim_z).to(self.device), y), dim=1)
        if cond is None:
            pred, _ = self.model(y, rev = True)  # output of NN without padding
        else:
            pred, _ = self.model(y, rev = True, c=cond.to(self.device))  # output of NN without padding
        predictions =  pred.cpu().detach()
        return np.concatenate((predictions[:,:3],self.scaler.inverse_transform(predictions[:,3:self.ndim_x])), axis=1)



class ScatteringProblemIResNet(ScatteringProblemSplit):
    def __init__(self, device, batch_size, ndim_x_class, ndim_x_features, ndim_y, ndim_z, ndim_pad_x_class, ndim_pad_x_features, ndim_pad_zy, init_scale, final_decay, n_epochs, lr_init, adam_betas, l2_weight_reg, lambd_fit_forw, lambd_mmd_forw, lambd_mmd_back_class, lambd_mmd_back_feature, lambd_reconstruct, mmd_back_weighted, y_uncertainty_sigma, add_pad_noise, add_z_noise, add_y_noise, n_its_per_epoch, pre_low_lr, filename_out, mmd_forw_kernels, mmd_back_kernels):
        super().__init__(device, batch_size, ndim_x_class, ndim_x_features, ndim_y, ndim_z, ndim_pad_x_class, ndim_pad_x_features, ndim_pad_zy, init_scale, final_decay, n_epochs, lr_init, adam_betas, l2_weight_reg, lambd_fit_forw, lambd_mmd_forw, lambd_mmd_back_class, lambd_mmd_back_feature, lambd_reconstruct, mmd_back_weighted, y_uncertainty_sigma, add_pad_noise, add_z_noise, add_y_noise, n_its_per_epoch, pre_low_lr, filename_out, mmd_forw_kernels, mmd_back_kernels)


    def train_epoch(self,i_epoch, test=False):
        if not test:
            self.model.train()
            loader = self.train_loader

        if test:
            loader = self.test_loader

        batch_idx = 0
        loss_history = []

        for x, y in loader:

            if batch_idx > self.n_its_per_epoch:
                break
            batch_losses = []

            batch_idx += 1

            x, y = x.to(self.device), y.to(self.device)

            if self.add_y_noise > 0:
                y += self.add_y_noise * self.noise_batch(self.ndim_y)
            if self.ndim_pad_x_class:
                x_class = torch.cat((x[:,:3], self.add_pad_noise * self.noise_batch(self.ndim_pad_x_class)), dim=1)
            
            if self.ndim_pad_x_features:
                x_features = torch.cat((x[:,3:], self.add_pad_noise * self.noise_batch(self.ndim_pad_x_features)), dim=1)
            if self.ndim_pad_zy:
                y = torch.cat((self.add_pad_noise * self.noise_batch(self.ndim_pad_zy), y), dim=1)
            y = torch.cat((self.noise_batch(self.ndim_z), y), dim=1)

            out_y, out_y_jac = self.model([x_class,x_features], jac  = True)
            # tuple with output[0] and jacobian[1]

            batch_losses.extend(self.loss_forward_mmd(out_y, y))
            batch_losses.append(self.loss_backward_mmd(x_class,x_features, y))
            batch_losses.append(self.loss_reconstruction(out_y.data, x_class, x_features))

            l_total = sum(batch_losses)
            loss_history.append([l.item() for l in batch_losses]) # lisr of lists: list for each batch

            if not test:
                l_total.backward()
                self.optim_step()

        if test:
            monitoring.show_hist(out_y[:, :self.ndim_z])
            monitoring.show_cov(out_y[:, :self.ndim_z])
        return np.mean(loss_history, axis=0)



class ScatteringProblemForward(ScatteringProblemSplit):
    def __init__(self, device,  batch_size, ndim_x_class, ndim_x_features, ndim_y, init_scale, final_decay, n_epochs, lr_init, adam_betas, l2_weight_reg, 
                        lambd_mmd_for_class, lambd_mmd_for_feature, mmd_back_weighted, y_uncertainty_sigma, n_its_per_epoch, pre_low_lr, filename_out, mmd_forw_kernels, mmd_back_kernels):
        self.device = device
        self.init_scale = init_scale
        self.final_decay = final_decay
        self.n_epochs = n_epochs
        self.lr_init = lr_init
        self.adam_betas = adam_betas
        self.l2_weight_reg = l2_weight_reg
        self.batch_size = batch_size
        self.ndim_x_class = ndim_x_class
        self.ndim_x_features = ndim_x_features
        self.ndim_y = ndim_y
        self.lambd_mmd_for_class = lambd_mmd_for_class
        self.lambd_mmd_for_feature = lambd_mmd_for_feature
        self.mmd_back_weighted = mmd_back_weighted
        self.y_uncertainty_sigma = y_uncertainty_sigma
        self.n_its_per_epoch = n_its_per_epoch
        self.pre_low_lr = pre_low_lr
        self.filename_out = filename_out
        self.mmd_forw_kernels = mmd_forw_kernels
        self.mmd_back_kernels = mmd_back_kernels

    def update_hyperparameters(self,  device,  batch_size, ndim_x_class, ndim_x_features, ndim_y, init_scale, final_decay, n_epochs, lr_init, adam_betas, l2_weight_reg, 
                        lambd_mmd_for_class, lambd_mmd_for_feature, mmd_back_weighted, y_uncertainty_sigma, n_its_per_epoch, pre_low_lr, filename_out, mmd_forw_kernels, mmd_back_kernels):
        self.device = device
        self.init_scale = init_scale
        self.final_decay = final_decay
        self.n_epochs = n_epochs
        self.lr_init = lr_init
        self.adam_betas = adam_betas
        self.l2_weight_reg = l2_weight_reg
        self.batch_size = batch_size
        self.ndim_x_class = ndim_x_class
        self.ndim_x_features = ndim_x_features
        self.ndim_y = ndim_y
        self.lambd_mmd_for_class = lambd_mmd_for_class
        self.lambd_mmd_for_feature = lambd_mmd_for_feature
        self.mmd_back_weighted = mmd_back_weighted
        self.y_uncertainty_sigma = y_uncertainty_sigma
        self.n_its_per_epoch = n_its_per_epoch
        self.pre_low_lr = pre_low_lr
        self.filename_out = filename_out
        self.mmd_forw_kernels = mmd_forw_kernels
        self.mmd_back_kernels = mmd_back_kernels

    def shape_condition(self,true_shape):
        a,b=1,1
        if true_shape=='hardsphere':
            while a+b>=1:
                b=np.random.normal(0.8, 0.2)
                if b<1:
                    a = np.random.uniform(0, (1-b)/2)
            c = 1-a-b    
            return torch.tensor([a,b,c])
        else:
            # if sphere or cylinder probs are calculated similarly
            while a+b>=1:
                a,b=np.random.normal(0.55, 0.1),np.random.normal(0.05, 0.05)
            c = 1-a-b
            if true_shape=='sphere':
                return torch.tensor([a,b,c])  
            else:
                return torch.tensor([c,b,a])    

    def read_data(self, path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
        self.inputs = torch.zeros(len(files), self.ndim_y, dtype = torch.float32)
        self.labels = torch.zeros(len(files), self.ndim_x_class+self.ndim_x_features, dtype = torch.float32)
        input_keys = ['shape', 'radius', 'radius_pd','length', 'length_pd',  'volfraction']
        for i,f in enumerate(files):
            with  h5py.File(os.path.join(path,f),'r') as file:
                print(file, end = '\r')
                self.inputs[i,:] = torch.from_numpy(file['entry/I'][()].flatten())#I_noisy
                for i_k, key in enumerate(input_keys):
                    try:
                        if key == 'shape':
                            shape = file['properties'][key][()].decode("utf-8")
                            if shape == 'sphere':
                                self.labels[i, i_k:i_k+3] = torch.tensor([1,0,0]) 
                                #conditions[i, i+k]
                            elif shape == 'hardsphere':
                                self.labels[i, i_k:i_k+3] = torch.tensor([0,1,0])
                            else:
                                self.labels[i, i_k:i_k+3] = torch.tensor([0,0,1])
                        elif key =='radius':
                            if shape == 'sphere':
                                self.labels[i, i_k+2:i_k+5] = torch.tensor([file['properties'][key][()],0,0]) 
                            elif shape == 'hardsphere':
                                self.labels[i, i_k+2:i_k+5] = torch.tensor([0,file['properties'][key][()],0])
                            else:
                                self.labels[i, i_k+2:i_k+5] = torch.tensor([0,0,file['properties'][key][()]])
                        else:
                            self.labels[i, i_k+4] = file['properties'][key][()]
                    except KeyError:
                        # spheres don't have all of the properties a cylinder does
                        pass
                        

    def normalize_inputs(self):
        self.scaler = StandardScaler()
        self.scaler.fit(self.inputs)
        self.inputs_norm = torch.from_numpy(self.scaler.transform(self.inputs)).type(torch.float32)


    def loss_forward_mmd(self, x_class, x_features, pred):
        pred_class, pred_features = pred[:,:3], pred[:, 3:]
        MMD_class = losses.backward_mmd(x_class, pred_class, self.mmd_forw_kernels, self.device) 
        MMD_features = losses.backward_mmd(x_features, pred_features, self.mmd_forw_kernels, self.device)
        if self.mmd_back_weighted:
            MMD_class *= torch.exp(- 0.5 / self.y_uncertainty_sigma**2 * losses.l2_dist_matrix(pred, pred))
            MMD_features *= torch.exp(- 0.5 / self.y_uncertainty_sigma**2 * losses.l2_dist_matrix(pred, pred))
        return self.lambd_mmd_for_class * torch.mean(MMD_class)+self.lambd_mmd_for_feature * torch.mean(MMD_features)


    def train_epoch(self, i_epoch, test=False):
        if not test:
            self.model.train()
            loader = self.train_loader

        if test:
            self.model.eval()
            loader = self.test_loader
            nograd = torch.no_grad()
            nograd.__enter__()


        batch_idx = 0
        loss_history = []

        for y,x in loader: # because y is  actualy the result of x

            if batch_idx > self.n_its_per_epoch:
                break

            batch_idx += 1

            x, y = x.to(self.device), y.to(self.device)
            x_class, x_features = x[:, :3],x[:, 3:]


            
            pred = self.model(y).reshape(-1,10)
            loss = self.loss_forward_mmd(x_class, x_features, pred)
            loss_history.append([loss.item()]) # lisr of lists: list for each batch
            
            if not test:
                loss.backward()
                self.optim_step()

        if test:
            nograd.__exit__(None, None, None)
        return np.mean(loss_history, axis=0)
        
    def train(self):

        try:
            t_start = time()
            for i_epoch in range(-self.pre_low_lr, self.n_epochs):

                if i_epoch < 0:
                    for param_group in self.optim.param_groups:
                        param_group['lr'] = self.lr_init * 1e-1

                train_losses = self.train_epoch(i_epoch) # mean over batches
                test_losses  = self.train_epoch(i_epoch, test=True)
                t = np.concatenate([train_losses, test_losses])
                print('Epoch {i_e}: training loss: {tl}, test loss: {testl}'.format(i_e = i_epoch, tl = t[0], testl = t[1]))


        except:
            self.save(self.filename_out + '_ABORT')
            raise

        finally:
            print("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.))
            self.save(self.filename_out)


    def make_prediction(self, data_subset):# labels, model):
        self.model.to('cpu')
        return self.model(self.inputs_norm[data_subset]).cpu().detach()



    def create_table_from_outcomes(self, pred, data_subset):
        try:
            sampled_labels = self.labels[data_subset]
        except AttributeError:
            sampled_labels = self.labels[data_subset]
        df = pd.DataFrame(columns = ['true_shape', 'pred_shape', 'radius','pred_radius'], index = [])
        df['true_shape'] = sampled_labels[:,:3].argmax(axis=1)
        df['pred_shape'] = pred[:,:3].argmax(axis=1)
        df['radius'] = np.take_along_axis(sampled_labels[:,3:6],df.true_shape.values.reshape(-1,1), axis=1)
        df['pred_radius'] = np.take_along_axis(pred[:,3:6],df.pred_shape.values.reshape(-1,1), axis=1)
        df['radius_pd'] = sampled_labels[:,6]
        df['pred_radius_pd'] = pred[:,6]
        df['length'] = sampled_labels[:,7]
        df['pred_length'] = pred[:,7] # only those identified as cylinder should have lengh and pd_length  df.pred_shape.values ==2, pred[:,7], 0) 
        df['length_pd'] = sampled_labels[:,8]
        df['pred_length_pd'] =  pred[:,8]
        df['volfraction'] = sampled_labels[:,9]
        df['pred_volfraction'] = pred[:,9]
        return df

