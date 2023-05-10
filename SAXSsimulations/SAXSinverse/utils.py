import numpy as np
import torch
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import os
import h5py
from time import time


from SAXSsimulations.SAXSinverse import losses
from SAXSsimulations.SAXSinverse  import monitoring
from sklearn.preprocessing import StandardScaler, Normalizer




class ScatteringProblem():
    """ a class to wrap up a scattering problem learning suite with user-defined hyperparameters"""
    def __init__(self, filename_out, 
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
                        flow):
    
        self.filename_out          = filename_out
        self.device                = device
        self.lr_init               = lr_init
        self.batch_size            = batch_size
        self.n_epochs              = n_epochs
        self.n_its_per_epoch = n_its_per_epoch
        self.pre_low_lr            = pre_low_lr
        self.final_decay           = final_decay
        self.l2_weight_reg         = l2_weight_reg
        self.adam_betas            = adam_betas
        self.ndim_pad_x            = ndim_pad_x
        self.ndim_y                = ndim_y
        self.ndim_z                = ndim_z
        self.ndim_pad_zy           = ndim_pad_zy 
        self.train_reconstruction  = train_reconstruction
        self.lambd_fit_forw        = lambd_fit_forw 
        self.lambd_mmd_forw        = lambd_mmd_forw
        self.lambd_mmd_back        = lambd_mmd_back
        self.lambd_reconstruct     = lambd_reconstruct
        self.add_y_noise           = add_y_noise 
        self.add_z_noise           = add_z_noise
        self.add_pad_noise         = add_pad_noise
        self.y_uncertainty_sigma   = y_uncertainty_sigma
        self.mmd_forw_kernels      = mmd_forw_kernels
        self.mmd_back_kernels      = mmd_back_kernels
        self.mmd_back_weighted     = mmd_back_weighted
        self.init_scale            = init_scale
        self.flow                  = flow

    def update_hyperparameters(self, filename_out, 
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
                        flow):
    
        self.filename_out          = filename_out
        self.device                = device
        self.lr_init               = lr_init
        self.batch_size            = batch_size
        self.n_epochs              = n_epochs
        self.n_its_per_epoch = n_its_per_epoch
        self.pre_low_lr            = pre_low_lr
        self.final_decay           = final_decay
        self.l2_weight_reg         = l2_weight_reg
        self.adam_betas            = adam_betas
        self.ndim_pad_x            = ndim_pad_x
        self.ndim_y                = ndim_y
        self.ndim_z                = ndim_z
        self.ndim_pad_zy           = ndim_pad_zy 
        self.train_reconstruction  = train_reconstruction
        self.lambd_fit_forw        = lambd_fit_forw 
        self.lambd_mmd_forw        = lambd_mmd_forw
        self.lambd_mmd_back        = lambd_mmd_back
        self.lambd_reconstruct     = lambd_reconstruct
        self.add_y_noise           = add_y_noise 
        self.add_z_noise           = add_z_noise
        self.add_pad_noise         = add_pad_noise
        self.y_uncertainty_sigma   = y_uncertainty_sigma
        self.mmd_forw_kernels      = mmd_forw_kernels
        self.mmd_back_kernels      = mmd_back_kernels
        self.mmd_back_weighted     = mmd_back_weighted
        self.init_scale            = init_scale
        self.flow                  = flow

    def read_data(self, path, shapes, input_keys):
        """
        Load the trainig data from HDF files:
        Arguments:
            path(str): a directory with all HDF available for training
            shapes(int): total number of shapes present in the training set
            input_keys(list): a list naming all the parameters a network is supposed to identify, named exatly as in the HDF trainung files and starting with 'shape' and 'radius'
        """
        if not(input_keys[0] =='shape' and input_keys[1]  =='radius'):
            raise ValueError('The first two parameters of input keys should be named "shape" and "radius"')
        self.input_features = input_keys.copy()
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
        self.ndim_x = shapes*2 +len(input_keys)-2
        if self.flow == 'inverse':
            assert (self.ndim_x + self.ndim_pad_x 
                    == self.ndim_y + self.ndim_z + self.ndim_pad_zy), "Dimensions don't match up"
        self.labels = torch.zeros(len(files), self.ndim_y, dtype = torch.float32)
        self.inputs = torch.zeros(len(files), self.ndim_x, dtype = torch.float32)
        self.shapes_dict = {}
        for i,f in enumerate(files):
            with  h5py.File(os.path.join(path,f),'r') as file:
                print(file, end = '\r')
                assert (self.ndim_y ==  torch.from_numpy(file['entry/I'][()].flatten()).shape[0]), "scattering curve has different size"
                self.labels[i,:] = torch.from_numpy(file['entry/I'][()].flatten())#I_noisy
                for i_k, key in enumerate(input_keys):
                    try:
                        if key == 'shape':
                            shape = file['properties'][key][()].decode("utf-8")
                            value = torch.zeros(shapes)
                            if shape  not in self.shapes_dict:
                                self.shapes_dict[shape] = max(list(self.shapes_dict.values()))+1 if len(self.shapes_dict.keys())>0 else 0
                            value[self.shapes_dict[shape]] = 1
                            self.inputs[i, i_k:i_k+shapes] = value
                        elif key =='radius':
                            value = torch.zeros(shapes)
                            value[self.shapes_dict[shape]] = file['properties'][key][()]
                            self.inputs[i, i_k+(shapes-1):i_k+(shapes*2-1)] = value
                        else:
                            self.inputs[i, i_k+(shapes*2-2)] = file['properties'][key][()]
                    except KeyError:
                        # e.g spheres don't have all of the properties a cylinder does
                        pass
        with open(self.filename_out.split('.pt')[0]+'_shapes.txt', 'w+', encoding="utf-8") as f:
            for k in self.shapes_dict.keys():
                f.write(''.join([k, ' : ', str(self.shapes_dict[k]), '\n']))
        # if the flow of the NN is forward swap the inputs and outputs
        if self.flow == 'forward':
            swap_labels = self.labels
            self.labels = self.inputs
            self.inputs = swap_labels 
        if self.flow == 'inverse':
            assert (self.ndim_x + self.ndim_pad_x 
                    == self.ndim_y + self.ndim_z + self.ndim_pad_zy), "Dimensions don't match up"


    def normalize_inputs(self):
        """
        Normalizes the data.  If the flow of the NN is inverse, properies such as radius, length, polydispersity etc( not shape) are standartized(mean = 0, sd = 1). If the flow is forward, the scattering curves are normalized.
        """

        if self.flow == "inverse":
            self.scaler = StandardScaler()
            self.scaler.fit(self.inputs[:,len(self.shapes_dict.keys()):])
            self.inputs_norm = self.scaler.transform(self.inputs[:,len(self.shapes_dict.keys()):])
            self.inputs_norm = torch.concatenate((self.inputs[:,:len(self.shapes_dict.keys())],torch.from_numpy(self.inputs_norm) ), axis=1).type(torch.float32)
            with open(self.filename_out.split('.pt')[0]+'_scaler.txt', 'w+', encoding="utf-8") as f:
                f.write('mean values: ')
                f.write(' '.join([str(i) for i in self.scaler.mean_.flatten()]))
                f.write('\nstandard deviation values:')
                f.write(' '.join([str(i) for i in self.scaler.scale_.flatten()]))


        else:
            self.scaler = Normalizer()
            self.inputs_norm = torch.from_numpy(self.scaler.fit_transform(self.inputs)).type(torch.float32)


    def create_loaders(self):
        """
        Prepare the dataloaders for NN weights learning: 
        1) split data into three parts (train - for weights learning, validation - optimizing, test - for the final EDA of trained model)
        2) shuffle data
        3) create loaders of batch size
        """
        # Creating data indices for training and validation splits:
        dataset_size = len(self.labels)
        indices = list(range(dataset_size))
        test_size = 0.1
        val_size = 0.2
        test_split = int(np.floor(test_size * dataset_size))
        val_split = int(np.floor(val_size * dataset_size))
        
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
        """
        Set the design of NN
        Arguments:
            model: torch model
        """
        self.model = model
        self.model.to(self.device)


    def set_optimizer(self):
        """
        set on optimizer with hyperparmaeters defined as initializing the class or after calling the `update_hyperparameters' function
        """
        self.params_trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        for p in self.params_trainable:
            p.data = self.init_scale * torch.randn(p.data.shape).to(self.device)

        gamma = (self.final_decay)**(1./self.n_epochs)
        self.optim = torch.optim.Adam(self.params_trainable, lr=self.lr_init, betas=self.adam_betas, eps=1e-6, weight_decay=self.l2_weight_reg)
        self.weight_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=gamma)

    def loss_backward_mmd(self,x, y):
        """
        Calculates the MMD loss in the backward direction
        """
        x_samples, x_samples_jac = self.model(y, rev=True, jac = True) 
        MMD = losses.backward_mmd(x, x_samples, self.mmd_back_kernels, self.device) 
        if self.mmd_back_weighted:
            MMD *= torch.exp(- 0.5 / self.y_uncertainty_sigma**2 * losses.l2_dist_matrix(y, y))
        return self.lambd_mmd_back * torch.mean(MMD)

    def loss_forward_mmd(self, out, y):
        """
        Calculates  MMD loss in the forward direction
        """
        output_block_grad = torch.cat((out[:, :self.ndim_z],
                                    out[:, -self.ndim_y:].data), dim=1) 
        y_short = torch.cat((y[:, :self.ndim_z], y[:, -self.ndim_y:]), dim=1)

        l_forw_fit = self.lambd_fit_forw * losses.l2_fit(out[:, self.ndim_z:], y[:, self.ndim_z:], self.batch_size)
        l_forw_mmd = self.lambd_mmd_forw  * torch.mean(losses.forward_mmd(output_block_grad, y_short, self.mmd_forw_kernels, self.device))

        return l_forw_fit, l_forw_mmd

    def loss_reconstruction(self, out_y, x):
        """
        calculates reconstruction loss
        """
        cat_inputs = [out_y[:, :self.ndim_z] + self.add_z_noise * self.noise_batch(self.ndim_z)] # list with 1 tensor
        
        if self.ndim_pad_zy:
            cat_inputs.append(out_y[:, self.ndim_z:-self.ndim_y] + self.add_pad_noise * self.noise_batch(self.ndim_pad_zy)) # list with 2 tensor
        cat_inputs.append(out_y[:, -self.ndim_y:] + self.add_y_noise * self.noise_batch(self.ndim_y)) # list with 3 tensors
        x_reconstructed, x_reconstructed_jac = self.model(torch.cat(cat_inputs, 1), rev=True, jac = True) # concatenate list elements along axis 1
        return self.lambd_reconstruct * losses.l2_fit(x_reconstructed[:, :self.ndim_pad_x], x[:,:self.ndim_pad_x], self.batch_size)

    def optim_step(self):
        """
        an optimizer step
        """
        self.optim.step()
        self.optim.zero_grad()


    def save(self,name):
        """
        saves the model
        """
        torch.save({'opt':self.optim.state_dict(),
                    'net':self.model.state_dict()}, name)

    def load(self,name):
        """
        load network and optimizer
        """
        state_dicts = torch.load(name)
        self.model.load_state_dict(state_dicts['net'])
        try:
            self.optim.load_state_dict(state_dicts['opt'])
        except ValueError:
            print('Cannot load optimizer for some reason')

    def noise_batch(self,ndim):
        """
        adds normlly distributed noize to the data
        """
        return torch.randn(self.batch_size, ndim).to(self.device)

    def train_epoch(self, test=False):
        """
        the training loop over one epoch
        """
        if not test:
            self.model.train()
            loader = self.train_loader

        else:
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

            out_y, out_y_jac = self.model(x, jac = True)

            batch_losses.extend(self.loss_forward_mmd(out_y, y))
            batch_losses.append(self.loss_backward_mmd(x, y))
            if self.train_reconstruction:
                batch_losses.append(self.loss_reconstruction(out_y.data, x))

            l_total = sum(batch_losses)
            loss_history.append([l.item() for l in batch_losses]) # lisr of lists: list for each batch

            if not test:
                l_total.backward()
                self.optim_step()

        if test:
            #monitoring.show_hist(out_y[:, :self.ndim_z])
            #monitoring.show_cov(out_y[:, :self.ndim_z])
            nograd.__exit__(None, None, None)
        return np.mean(loss_history, axis=0)

    def train(self):
        if self.flow == 'inverse':
            monitoring.restart(self.train_reconstruction)

        try:
            t_start = time()
            for i_epoch in range(-self.pre_low_lr, self.n_epochs):

                if i_epoch < 0:
                    for param_group in self.optim.param_groups:
                        param_group['lr'] = self.lr_init * 1e-1

                train_losses = self.train_epoch() # mean over batches
                test_losses        = self.train_epoch(test=True)
                t = np.concatenate([train_losses, test_losses])
                if self.flow == 'inverse':
                    monitoring.show_loss(t)
                else:
                    print('Epoch {i_e}: training loss: {tl}, test loss: {testl}'.format(i_e = i_epoch, tl = t[0], testl = t[1]))
        except:
            self.save(self.filename_out + '_ABORT')
            raise

        finally:
            print("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.))
            self.save(self.filename_out)

    def scale_back(self, predictions):
        """
        only applicable for the inverse flow. Scale back the predicted values with the respective mean and standard deviation of initial input data. 
        Arguments:
            predictions (torch.Tensor): predicted values of the scatterer
        Returns:
            torch.Tensor: rescaled predictions
        """
        scaled_predictions = np.copy(predictions)
        for i in range(7):
            x = predictions[:, len(self.shapes_dict.keys())+i]
            
            scaled_predictions[:,len(self.shapes_dict.keys())+i] = x * float(self.scaler[i, 1]) + float(self.scaler[i,0])
        return scaled_predictions

    def predict(self, data, shapes):
        """
        make prediction on some data
        """

        if self.flow == 'inverse':
            y = torch.cat((self.add_pad_noise * torch.randn( len(data), self.ndim_pad_zy).to(self.device), data.to(self.device)), dim=1)
            y = torch.cat((torch.randn( len(data), self.ndim_z).to(self.device), y), dim=1)
            pred, _ = self.model(y, rev = True) 
            predictions =  pred.cpu().detach()
            return np.concatenate((predictions[:,:shapes],self.scaler.inverse_transform(predictions[:,shapes:self.ndim_x])), axis=1)
        else:
            self.model.to('cpu')
            try:
                return self.model(data).cpu().detach()
            except AttributeError:
                return self.model(data).cpu().detach()

    def make_prediction_test(self, data_subset):
        """
        run trained Neural network to predict the scatterer parameters
        """
        if self.flow == 'inverse':
            y = torch.cat((self.add_pad_noise * torch.randn( len(self.labels[data_subset]), self.ndim_pad_zy).to(self.device), self.labels[data_subset].to(self.device)), dim=1)
            y = torch.cat((torch.randn( len(self.labels[data_subset]), self.ndim_z).to(self.device), y), dim=1)
            pred, _ = self.model(y, rev = True) 
            predictions =  pred.cpu().detach()
            return np.concatenate((predictions[:,:len(self.shapes_dict.keys())],self.scaler.inverse_transform(predictions[:,len(self.shapes_dict.keys()):self.ndim_x])), axis=1)
        else:
            self.model.to('cpu')
            try:
                return self.model(self.inputs_norm[data_subset]).cpu().detach()
            except AttributeError:
                return self.model(self.inputs[data_subset]).cpu().detach()
            

    def create_table_from_outcomes_test(self, pred, data_subset):
        """
        creates a table with all predicted vs true values
        """
        cols= [['true_'+i, 'pred_'+i] for i in self.input_features]
        if self.flow == 'inverse':
            sampled_inputs = self.inputs[data_subset]
        else:
            sampled_inputs = self.labels[data_subset]
        df = pd.DataFrame(columns =  [i for c in cols for i in c  ], index = [])
        shapes = len(self.shapes_dict.keys())
        df['true_shape'] = sampled_inputs[:,:shapes].argmax(axis=1)
        df['pred_shape'] = pred[:,:shapes].argmax(axis=1)
        df['true_radius'] = np.take_along_axis(sampled_inputs[:,shapes:2*shapes],df.true_shape.values.reshape(-1,1), axis=1)
        df['pred_radius'] = np.take_along_axis(pred[:,shapes:2*shapes],df.pred_shape.values.reshape(-1,1), axis=1)
        for i,c in enumerate(df.columns[4:]):
            if 'pred' in c:
                df[c] = pred[:,i//2+2*shapes]
            elif 'true' in c:
                df[c] = sampled_inputs[:,i//2+2*shapes]
        return df
    
    
    def create_table_from_outcomes(self, pred, data):
        """
        creates a table with all predicted vs true values
        """
        cols= [['true_'+i, 'pred_'+i] for i in self.input_features]
        sampled_inputs = data
        df = pd.DataFrame(columns =  [i for c in cols for i in c  ], index = [])
        shapes = len(self.shapes_dict.keys())
        df['true_shape'] = sampled_inputs[:,:shapes].argmax(axis=1)
        df['pred_shape'] = pred[:,:shapes].argmax(axis=1)
        df['true_radius'] = np.take_along_axis(sampled_inputs[:,shapes:2*shapes],df.true_shape.values.reshape(-1,1), axis=1)
        df['pred_radius'] = np.take_along_axis(pred[:,shapes:2*shapes],df.pred_shape.values.reshape(-1,1), axis=1)
        for i,c in enumerate(df.columns[4:]):
            if 'pred' in c:
                df[c] = pred[:,i//2+2*shapes]
            elif 'true' in c:
                df[c] = sampled_inputs[:,i//2+2*shapes]
        return df

    def predict_latent(self, data_subset):
        """
        Look at the latent space variables calculated for the data
        Arguments:
            data_subset(list): indices subsetting data
        """
        x = self.inputs_norm[data_subset].to(self.device)
        x = torch.cat((x, self.add_pad_noise * torch.randn(len(data_subset), self.ndim_pad_x).to(self.device)), dim=1)
        out_y, _ = self.model(x, jac  = True)
        return out_y[:,:2].cpu().detach()

class ScatteringProblemIResNet(ScatteringProblem):
    def __init__(self,  filename_out, 
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
                        flow):
        super().__init__(  filename_out, 
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


    def train_epoch(self, test=False):
        if not test:
            self.model.train()
            loader = self.train_loader

        else:
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
            if self.ndim_pad_x:
                x = torch.cat((x, self.add_pad_noise * self.noise_batch(self.ndim_pad_x)), dim=1)
            if self.ndim_pad_zy:
                y = torch.cat((self.add_pad_noise * self.noise_batch(self.ndim_pad_zy), y), dim=1)
            y = torch.cat((self.noise_batch(self.ndim_z), y), dim=1)
            
            out_y, out_y_jac = self.model([x], jac        = True)
            # tuple with output[0] and jacobian[1]

            batch_losses.extend(self.loss_forward_mmd(out_y, y))
            batch_losses.append(self.loss_backward_mmd(x, y))
            if self.train_reconstruction:
                batch_losses.append(self.loss_reconstruction(out_y.data, x))

            l_total = sum(batch_losses)
            loss_history.append([l.item() for l in batch_losses]) # lisr of lists: list for each batch

            if not test:
                l_total.backward()
                self.optim_step()

        #if test:
        #    monitoring.show_hist(out_y[:, :self.ndim_z])
        #    monitoring.show_cov(out_y[:, :self.ndim_z])
        return np.mean(loss_history, axis=0)



class ScatteringProblemForward(ScatteringProblem):
    def __init__(self,  filename_out, 
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
                        flow):
        self.filename_out             = filename_out 
        self.device                   = device
        self.lr_init                  = lr_init 
        self.batch_size               = batch_size 
        self.n_epochs                 = n_epochs
        self.n_its_per_epoch          = n_its_per_epoch
        self.pre_low_lr               = pre_low_lr
        self.final_decay              = final_decay 
        self.l2_weight_reg            = l2_weight_reg  
        self.adam_betas               = adam_betas
        self.ndim_y                   = ndim_y
        self.lambd_mmd_for_class      = lambd_mmd_for_class
        self.lambd_mmd_for_feature    = lambd_mmd_for_feature
        self.y_uncertainty_sigma      = y_uncertainty_sigma
        self.mmd_forw_kernels         = mmd_forw_kernels
        self.mmd_back_kernels         = mmd_back_kernels 
        self.mmd_back_weighted        = mmd_back_weighted 
        self.init_scale               = init_scale 
        self.flow                     = flow

    def update_hyperparameters(self,   filename_out, 
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
                                        flow):
        self.filename_out             = filename_out 
        self.device                   = device 
        self.lr_init                  = lr_init 
        self.batch_size               = batch_size 
        self.n_epochs                 = n_epochs
        self.n_its_per_epoch          = n_its_per_epoch
        self.pre_low_lr               = pre_low_lr
        self.final_decay              = final_decay
        self.l2_weight_reg            = l2_weight_reg  
        self.adam_betas               = adam_betas
        self.ndim_y                   = ndim_y
        self.lambd_mmd_for_class      = lambd_mmd_for_class
        self.lambd_mmd_for_feature    = lambd_mmd_for_feature 
        self.y_uncertainty_sigma      = y_uncertainty_sigma
        self.mmd_forw_kernels         = mmd_forw_kernels
        self.mmd_back_kernels         = mmd_back_kernels 
        self.mmd_back_weighted        = mmd_back_weighted 
        self.init_scale               = init_scale
        self.flow                     = flow

    def loss_forward_mmd(self, x_class, x_features, pred):
        """
        Calculates the MMD loss in the forward direction; penalize shape and other parameters by different constants
        """
        pred_class, pred_features = pred[:,:len(self.shapes_dict.keys())], pred[:, len(self.shapes_dict.keys()):]
        MMD_class = losses.backward_mmd(x_class, pred_class, self.mmd_forw_kernels, self.device) 
        MMD_features = losses.backward_mmd(x_features, pred_features, self.mmd_forw_kernels, self.device)
        if self.mmd_back_weighted:
            MMD_class *= torch.exp(- 0.5 / self.y_uncertainty_sigma**2 * losses.l2_dist_matrix(pred, pred))
            MMD_features *= torch.exp(- 0.5 / self.y_uncertainty_sigma**2 * losses.l2_dist_matrix(pred, pred))
        return self.lambd_mmd_for_class * torch.mean(MMD_class)+self.lambd_mmd_for_feature * torch.mean(MMD_features)


    def train_epoch(self, test=False):
        """
        training loop over one epoch, loaders are passed in different order as inputs and outputs are mixed. Differentiation between shape and features is mased to penalize those
        """
        if not test:
            self.model.train()
            loader = self.train_loader

        else:
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
            x_class, x_features = x[:, :len(self.shapes_dict.keys())],x[:, len(self.shapes_dict.keys()):]


            
            pred = self.model(y)
            pred = pred.reshape(-1,len(self.shapes_dict.keys())*2 + len(self.input_features)-2)
            loss = self.loss_forward_mmd(x_class, x_features, pred)
            loss_history.append([loss.item()]) # lisr of lists: list for each batch
            
            if not test:
                loss.backward()
                self.optim_step()

        if test:
            nograd.__exit__(None, None, None)
        return np.mean(loss_history, axis=0)
        

