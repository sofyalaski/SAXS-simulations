import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import FrEIA.framework as Ff
import FrEIA.modules as Fm


import config as c

def subnet2(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, c.hidden_layer_sizes), nn.ReLU(),
                        nn.Linear(c.hidden_layer_sizes,  c.hidden_layer_sizes//2), nn.ReLU(),
                        nn.Linear(c.hidden_layer_sizes//2,  dims_out))


def subnet1(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, c.hidden_layer_sizes*2), nn.ReLU(),
                        nn.Linear(c.hidden_layer_sizes*2,  c.hidden_layer_sizes), nn.ReLU(),
                        nn.Linear(c.hidden_layer_sizes,  dims_out))

input1 = Ff.InputNode(c.ndim_x_class + c.ndim_pad_x_class, name='input_class')
input2 = Ff.InputNode(c.ndim_x_features + c.ndim_pad_x_features, name='input_features')
nodes = [input1]
for i in range(c.N_blocks):
    nodes.append(Ff.Node([nodes[-1].out0],Fm.IResNetLayer, {'internal_size':c.hidden_layer_sizes, 
                                                'n_internal_layers':4, 
                                                'jacobian_iterations':12,
                                                'hutchinson_samples':1, 
                                                'fixed_point_iterations':50,
                                                'lipschitz_iterations':10,
                                                'lipschitz_batchsize':10,
                                                'spectral_norm_max':0.8,
                                                }, name='ires'+str(i)))

    nodes.append(Ff.Node([nodes[-1].out0], Fm.PermuteRandom, {'seed':i}, name='class_permute_'+str(i)))
nodes.append(input2)
nodes.append(Ff.Node([nodes[-2].out0, nodes[-1].out0], Fm.Concat, {}, name='Concat'))

for i in range(c.N_blocks):
    nodes.append(Ff.Node([nodes[-1].out0],Fm.IResNetLayer, {'internal_size':c.hidden_layer_sizes//2, 
                                                'n_internal_layers':3, 
                                                'jacobian_iterations':12,
                                                'hutchinson_samples':1, 
                                                'fixed_point_iterations':50,
                                                'lipschitz_iterations':10,
                                                'lipschitz_batchsize':10,
                                                'spectral_norm_max':0.8,
                                                }, name='ires_concat_'+str(i)))
    nodes.append(Ff.Node([nodes[-1].out0], Fm.PermuteRandom, {'seed':c.N_blocks+i}, name='class_permute_'+str(c.N_blocks+i)))
nodes.append(Ff.OutputNode([nodes[-1].out0], name='output'))
model = Ff.GraphINN(nodes, verbose=c.verbose_construction)
      
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



'''
def subnet2(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, c.hidden_layer_sizes), nn.ReLU(),
                        nn.Linear(c.hidden_layer_sizes,  c.hidden_layer_sizes//2), nn.ReLU(),
                        nn.Linear(c.hidden_layer_sizes//2,  dims_out))


def subnet1(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, c.hidden_layer_sizes*2), nn.ReLU(),
                        nn.Linear(c.hidden_layer_sizes*2,  c.hidden_layer_sizes), nn.ReLU(),
                        nn.Linear(c.hidden_layer_sizes,  dims_out))
RNVP1 = Ff.Node([input1.out0],Fm.AllInOneBlock, {'subnet_constructor':subnet1, 'affine_clamping':c.exponent_clamping}, name='class_coupling_1')
permute_class1 = Ff.Node([RNVP1.out0], Fm.PermuteRandom, {'seed':1}, name='class_permute_1')
RNVP2 = Ff.Node([permute_class1.out0],Fm.AllInOneBlock, {'subnet_constructor':subnet1, 'affine_clamping':c.exponent_clamping}, name='class_coupling_2')
permute_class2 = Ff.Node([RNVP2.out0], Fm.PermuteRandom, {'seed':2}, name='class_permute_2')
RNVP3 = Ff.Node([permute_class2.out0],Fm.AllInOneBlock, {'subnet_constructor':subnet1, 'affine_clamping':c.exponent_clamping}, name='class_coupling_3')
permute_class3 = Ff.Node([RNVP3.out0], Fm.PermuteRandom, {'seed':1}, name='class_permute_3')
RNVP4 = Ff.Node([permute_class3.out0],Fm.AllInOneBlock, {'subnet_constructor':subnet1, 'affine_clamping':c.exponent_clamping}, name='class_coupling_4')
permute_class4 = Ff.Node([RNVP4.out0], Fm.PermuteRandom, {'seed':4}, name='class_permute_4')

concat = Ff.Node([permute_class4.out0, input2.out0], Fm.Concat, {}, name='Concat')

RNVP5 = Ff.Node([concat.out0],Fm.AllInOneBlock, {'subnet_constructor':subnet2, 'affine_clamping':c.exponent_clamping}, name='feature_coupling_1')
permute_class5 = Ff.Node([RNVP5.out0], Fm.PermuteRandom, {'seed':1}, name='feature_permute_1')
RNVP6 = Ff.Node([permute_class5.out0],Fm.AllInOneBlock, {'subnet_constructor':subnet2, 'affine_clamping':c.exponent_clamping}, name='feature_coupling_2')
permute_class6 = Ff.Node([RNVP6.out0], Fm.PermuteRandom, {'seed':2}, name='feature_permute_2')
output = Ff.OutputNode([permute_class6.out0], name='output')


model = Ff.GraphINN([input1, input2,  RNVP1, permute_class1, RNVP2, permute_class2, RNVP3, permute_class3, RNVP4, permute_class4, concat,
                         RNVP5, permute_class5, RNVP6, permute_class6,  output], verbose=c.verbose_construction)
                         
                         
                         

IRes1 = Ff.Node([input1.out0],Fm.IResNetLayer, {'internal_size':c.hidden_layer_sizes, 
                                                'n_internal_layers':1, 
                                                'jacobian_iterations':12,
                                                'hutchinson_samples':1, 
                                                'fixed_point_iterations':50,
                                                'lipschitz_iterations':10,
                                                'lipschitz_batchsize':10,
                                                'spectral_norm_max':0.8,
                                                }, name='ires1')
permute_class1 = Ff.Node([IRes1.out0], Fm.PermuteRandom, {'seed':1}, name='class_permute_1')
concat = Ff.Node([permute_class1.out0, input2.out0], Fm.Concat, {}, name='Concat')

IRes3 = Ff.Node([concat.out0],Fm.IResNetLayer, {'internal_size':c.hidden_layer_sizes//2, 
                                                'n_internal_layers':1, 
                                                'jacobian_iterations':12,
                                                'hutchinson_samples':1, 
                                                'fixed_point_iterations':50,
                                                'lipschitz_iterations':10,
                                                'lipschitz_batchsize':10,
                                                'spectral_norm_max':0.8,
                                                }, name='ires3')
permute_class2 = Ff.Node([IRes3.out0], Fm.PermuteRandom, {'seed':1}, name='class_permute_2')
output = Ff.OutputNode([permute_class2.out0], name='output')
model = Ff.GraphINN([input1, input2,IRes1, permute_class1, concat, IRes3,  permute_class2, output], verbose=c.verbose_construction)

                        '''