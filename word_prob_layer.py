# -*- coding: utf-8 -*-
#pylint: skip-file
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils_pg import *

class WordProbLayer(nn.Module):
    def __init__(self, hidden_size, ctx_size, dim_y, dict_size, device, copy, coverage):
        super(WordProbLayer, self).__init__()
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size 
        self.dim_y = dim_y
        self.dict_size = dict_size
        self.device = device
        self.copy = copy
        self.coverage = coverage

        self.w_ds = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size + self.ctx_size + self.dim_y))
        self.b_ds = nn.Parameter(torch.Tensor(self.hidden_size)) 
        self.w_logit = nn.Parameter(torch.Tensor(self.dict_size, self.hidden_size))
        self.b_logit = nn.Parameter(torch.Tensor(self.dict_size)) 

        if self.copy:
            self.v = nn.Parameter(torch.Tensor(1, self.hidden_size + self.ctx_size + self.dim_y))
            self.bv = nn.Parameter(torch.Tensor(1))

        self.init_weights()

    def init_weights(self):
        init_xavier_weight(self.w_ds)
        init_bias(self.b_ds)
        init_xavier_weight(self.w_logit)
        init_bias(self.b_logit)
        if self.copy:
            init_xavier_weight(self.v)
            init_bias(self.bv)


    def forward(self, ds, ac, y_emb, att_dist=None, xids=None, max_ext_len=None):
        h = T.cat((ds, ac, y_emb), 2)
        logit = T.tanh(F.linear(h, self.w_ds, self.b_ds))
        logit = F.linear(logit, self.w_logit, self.b_logit)
        y_dec = T.softmax(logit, dim = 2)
        
        if self.copy:
            ext_zeros = Variable(torch.zeros(y_dec.size(0), y_dec.size(1), max_ext_len)).to(self.device)
            y_dec = T.cat((y_dec, ext_zeros), 2)
            g = T.sigmoid(F.linear(h, self.v, self.bv))
            y_dec = (g * y_dec).scatter_add(2, xids, (1 - g) * att_dist)
        
        return y_dec
