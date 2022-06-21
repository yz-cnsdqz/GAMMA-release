import os, sys, glob
import time
from typing import NamedTuple
import random
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim

import pickle
import json
import pdb
from tensorboardX import SummaryWriter

from models.baseops import MLP


class MLPBlock(nn.Module):
    def __init__(self, h_dim, out_dim, n_blocks, actfun='relu', residual=True):
        super(MLPBlock, self).__init__()
        self.residual = residual
        self.layers = nn.ModuleList([MLP(h_dim, h_dims=(h_dim, h_dim),
                                        activation=actfun)
                                        for _ in range(n_blocks)]) # two fc layers in each MLP
        self.out_fc = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        h = x
        for layer in self.layers:
            r = h if self.residual else 0
            h = layer(h) + r
        y = self.out_fc(h)
        return y


class GAMMAPolicy(nn.Module):
    '''
    the network input is the states:
        [vec_to_target_marker, vec_to_walking_path]
    the network output is the distribution of z, i.e. N(mu, logvar)
    '''
    def __init__(self, config):
        super(GAMMAPolicy, self).__init__()
        self.h_dim = config['h_dim']
        self.z_dim = config['z_dim']
        self.n_blocks = config['n_blocks']
        self.n_recur = config['n_recur'] # n_recur=0 means no recursive scheme
        self.actfun = config['actfun']
        self.is_stochastic = config.get('is_stochastic', True)
        self.min_logvar = config.get('min_logvar', -1)
        self.max_logvar = config.get('max_logvar', 3)

        if config['body_repr'] in {'ssm2_67_condi_marker', 'ssm2_67_condi_marker_l2norm', 'ssm2_67_condi_marker_height'}:
            self.in_dim = 67*3*2
        elif config['body_repr'] == 'ssm2_67_condi_wpath':
            self.in_dim = 67*3+2
        elif config['body_repr'] == 'ssm2_67_condi_wpath_height':
            self.in_dim = 67*3+3
        else:
            raise NotImplementedError('other body_repr is not implemented yet.')

        ## first a gru to encode X
        self.x_enc = nn.GRU(self.in_dim, self.h_dim)

        ## about the policy network
        self.pnet = MLPBlock(self.h_dim,
                            self.z_dim*2 if self.is_stochastic else self.z_dim,
                            self.n_blocks,
                            actfun=self.actfun)
        ## about the value network
        self.vnet = MLPBlock(self.h_dim,
                            1,
                            self.n_blocks,
                            actfun=self.actfun)

    def forward(self, x_in):
        '''
        x_in has
        - vec_to_ori:    [t, batch, dim=201]
        - vec_to_target: [t, batch, dim=201]
        - vec_to_wpath:  [t, batch, dim=2]
        '''
        _, hx = self.x_enc(x_in)
        hx = hx[0] #[b, d]
        z_prob = self.pnet(hx)
        # z_prob[:,:self.z_dim] = torch.tanh(z_prob[:,:self.z_dim])*4
        val = self.vnet(hx)
        if self.is_stochastic:
            mu = z_prob[:,:self.z_dim]
            logvar = z_prob[:, self.z_dim:]
            return mu, logvar, val
        else:
            return z_prob, val











