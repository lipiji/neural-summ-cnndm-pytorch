# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import numpy as np
import torch
import torch as T
import torch.nn as nn
from torch.autograd import Variable

from utils_pg import *
from gru_dec import *
from lstm_dec_v2 import *
from word_prob_layer import *

class Model(nn.Module):
    def __init__(self, modules, consts, options):
        super(Model, self).__init__()  
        
        self.has_learnable_w2v = options["has_learnable_w2v"]
        self.is_predicting = options["is_predicting"]
        self.is_bidirectional = options["is_bidirectional"]
        self.beam_decoding = options["beam_decoding"]
        self.cell = options["cell"]
        self.device = options["device"]
        self.copy = options["copy"]
        self.coverage = options["coverage"]
        self.avg_nll = options["avg_nll"]

        self.dim_x = consts["dim_x"]
        self.dim_y = consts["dim_y"]
        self.len_x = consts["len_x"]
        self.len_y = consts["len_y"]
        self.hidden_size = consts["hidden_size"]
        self.dict_size = consts["dict_size"] 
        self.pad_token_idx = consts["pad_token_idx"] 
        self.ctx_size = self.hidden_size * 2 if self.is_bidirectional else self.hidden_size

        self.w_rawdata_emb = nn.Embedding(self.dict_size, self.dim_x, self.pad_token_idx)
        if self.cell == "gru":
            self.encoder = nn.GRU(self.dim_x, self.hidden_size, bidirectional=self.is_bidirectional)
            self.decoder = GRUAttentionDecoder(self.dim_y, self.hidden_size, self.ctx_size, self.device, self.copy, self.coverage, self.is_predicting)
        else:
            self.encoder = nn.LSTM(self.dim_x, self.hidden_size, bidirectional=self.is_bidirectional)
            self.decoder = LSTMAttentionDecoder(self.dim_y, self.hidden_size, self.ctx_size, self.device, self.copy, self.coverage, self.is_predicting)
            
        self.get_dec_init_state = nn.Linear(self.ctx_size, self.hidden_size)
        self.word_prob = WordProbLayer(self.hidden_size, self.ctx_size, self.dim_y, self.dict_size, self.device, self.copy, self.coverage)

        self.init_weights()

    def init_weights(self):
        init_uniform_weight(self.w_rawdata_emb.weight)
        if self.cell == "gru": 
            init_gru_weight(self.encoder)
        else:
            init_lstm_weight(self.encoder)
        init_linear_weight(self.get_dec_init_state)
    
    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -T.log(T.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = T.sum(cost * y_mask, 0) / T.sum(y_mask, 0)
        else:
            cost = T.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        return T.mean(cost) 

    def encode(self, x, len_x, mask_x):
        self.encoder.flatten_parameters()
        emb_x = self.w_rawdata_emb(x)
        
        emb_x = torch.nn.utils.rnn.pack_padded_sequence(emb_x, len_x)
        hs, hn = self.encoder(emb_x, None)
        hs, _ = torch.nn.utils.rnn.pad_packed_sequence(hs)
         
        dec_init_state = T.sum(hs * mask_x, 0) / T.sum(mask_x, 0)
        dec_init_state = T.tanh(self.get_dec_init_state(dec_init_state))
        return hs, dec_init_state

    def decode_once(self, y, hs, dec_init_state, mask_x, x=None, max_ext_len=None, acc_att=None):
        batch_size = hs.size(1)
        if T.sum(y) < 0:
            y_emb = Variable(T.zeros((1, batch_size, self.dim_y))).to(self.device)
        else:
            y_emb = self.w_rawdata_emb(y)
        mask_y = Variable(T.ones((1, batch_size, 1))).to(self.device)

        if self.copy and self.coverage:
            hcs, dec_status, atted_context, att_dist, xids, C = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y, x, acc_att)
        elif self.copy:
            hcs, dec_status, atted_context, att_dist, xids = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y, xid=x)
        elif self.coverage:
            hcs, dec_status, atted_context, att_dist, C = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y, init_coverage=acc_att)
        else:
            hcs, dec_status, atted_context = self.decoder(y_emb, hs, dec_init_state, mask_x, mask_y)
        
        if self.copy:
            y_pred = self.word_prob(dec_status, atted_context, y_emb, att_dist, xids, max_ext_len)
        else:
            y_pred = self.word_prob(dec_status, atted_context, y_emb)

        if self.coverage:
            return y_pred, hcs, C
        else:
            return y_pred, hcs

    def forward(self, x, len_x, y, mask_x, mask_y, x_ext, y_ext, max_ext_len):
        
        hs, dec_init_state = self.encode(x, len_x, mask_x)

        y_emb = self.w_rawdata_emb(y)
        y_shifted = y_emb[:-1, :, :]
        y_shifted = T.cat((Variable(torch.zeros(1, *y_shifted[0].size())).to(self.device), y_shifted), 0)
        h0 = dec_init_state
        if self.cell == "lstm":
            h0 = (dec_init_state, dec_init_state)
        if self.coverage:
            acc_att = Variable(torch.zeros(T.transpose(x, 0, 1).size())).to(self.device) # B * len(x)

        if self.copy and self.coverage:
            hcs, dec_status, atted_context, att_dist, xids, C = self.decoder(y_shifted, hs, h0, mask_x, mask_y, x_ext, acc_att)
        elif self.copy:
            hcs, dec_status, atted_context, att_dist, xids = self.decoder(y_shifted, hs, h0, mask_x, mask_y, xid=x_ext)
        elif self.coverage:
            hcs, dec_status, atted_context, att_dist, C = self.decoder(y_shifted, hs, h0, mask_x, mask_y, init_coverage=acc_att)
        else:
            hcs, dec_status, atted_context = self.decoder(y_shifted, hs, h0, mask_x, mask_y)
        
        if self.copy:
            y_pred = self.word_prob(dec_status, atted_context, y_shifted, att_dist, xids, max_ext_len)
            cost = self.nll_loss(y_pred, y_ext, mask_y, self.avg_nll)
        else:
            y_pred = self.word_prob(dec_status, atted_context, y_shifted)
            cost = self.nll_loss(y_pred, y, mask_y, self.avg_nll)
        
        if self.coverage:
            cost_c = T.mean(T.sum(T.min(att_dist, C), 2))
            return y_pred, cost, cost_c
        else:
            return y_pred, cost, None
    

