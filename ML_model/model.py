import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import FrEIA.framework as Ff
import FrEIA.modules as Fm


import config as c

input1 = Ff.InputNode(c.ndim_x_class + c.ndim_pad_x_class, name='input_class')
input2 = Ff.InputNode(c.ndim_x_features + c.ndim_pad_x_features, name='input_features')

def subnet1(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, c.hidden_layer_sizes), nn.ReLU(),
                        nn.Linear(c.hidden_layer_sizes,  c.hidden_layer_sizes//2), nn.ReLU(),
                        nn.Linear(c.hidden_layer_sizes//2,  dims_out))


def subnet2(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, c.hidden_layer_sizes*2), nn.ReLU(),
                        nn.Linear(c.hidden_layer_sizes*2,  c.hidden_layer_sizes), nn.ReLU(),
                        nn.Linear(c.hidden_layer_sizes,  dims_out))

RNVP1 = Ff.Node([input1.out0],Fm.RNVPCouplingBlock, {'subnet_constructor':subnet1, 'clamp':c.exponent_clamping}, name='class_coupling_1')
permute_class1 = Ff.Node([RNVP1.out0], Fm.PermuteRandom, {'seed':1}, name='class_permute_1')
RNVP2 = Ff.Node([permute_class1.out0],Fm.RNVPCouplingBlock, {'subnet_constructor':subnet1, 'clamp':c.exponent_clamping}, name='class_coupling_2')
permute_class2 = Ff.Node([RNVP2.out0], Fm.PermuteRandom, {'seed':2}, name='class_permute_2')

concat = Ff.Node([permute_class2.out0, input2.out0], Fm.Concat, {}, name='Concat')

RNVP5 = Ff.Node([concat.out0],Fm.RNVPCouplingBlock, {'subnet_constructor':subnet2, 'clamp':c.exponent_clamping}, name='feature_coupling_1')
permute_class5 = Ff.Node([RNVP5.out0], Fm.PermuteRandom, {'seed':1}, name='feature_permute_1')
RNVP6 = Ff.Node([permute_class5.out0],Fm.RNVPCouplingBlock, {'subnet_constructor':subnet2, 'clamp':c.exponent_clamping}, name='feature_coupling_2')
permute_class6 = Ff.Node([RNVP6.out0], Fm.PermuteRandom, {'seed':2}, name='feature_permute_2')
output = Ff.OutputNode([permute_class6.out0], name='output')


model = Ff.GraphINN([input1, input2, RNVP1, permute_class1, RNVP2, permute_class2, concat,
                         RNVP5, permute_class5, RNVP6, permute_class6,  output], verbose=c.verbose_construction)
model.to(c.device)

params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
for p in params_trainable:
    p.data = c.init_scale * torch.randn(p.data.shape).to(c.device)

gamma = (c.final_decay)**(1./c.n_epochs)
optim = torch.optim.Adam(params_trainable, lr=c.lr_init, betas=c.adam_betas, eps=1e-6, weight_decay=c.l2_weight_reg)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=gamma)

def optim_step():
    #for p in params_trainable:
        #print(torch.mean(torch.abs(p.grad.data)).item())
    optim.step()
    optim.zero_grad()

def scheduler_step():
    #weight_scheduler.step()
    pass

def save(name):
    torch.save({'opt':optim.state_dict(),
                'net':model.state_dict()}, name)

def load(name):
    state_dicts = torch.load(name)
    model.load_state_dict(state_dicts['net'])
    try:
        optim.load_state_dict(state_dicts['opt'])
    except ValueError:
        print('Cannot load optimizer for some reason or other')
