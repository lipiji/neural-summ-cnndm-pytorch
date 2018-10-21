import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
from utils_pg import *

class LSTMAttentionDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, ctx_size, device):
        super(LSTMAttentionDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.device = device

        self.lstm_1 = nn.LSTMCell(self.input_size, self.hidden_size)

        self.Wx = nn.Parameter(torch.Tensor(self.hidden_size, self.ctx_size))
        self.Ux = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bx = nn.Parameter(torch.Tensor(self.hidden_size))

        self.Wc_att = nn.Parameter(torch.Tensor(self.ctx_size, self.ctx_size))
        self.b_att = nn.Parameter(torch.Tensor(self.ctx_size))

        self.W_comb_att = nn.Parameter(torch.Tensor(self.ctx_size, 2 * self.hidden_size))
        self.U_att = nn.Parameter(torch.Tensor(1, self.ctx_size))

        self.init_weights()

    def init_weights(self):
        init_ortho_weight(self.Wx)
        init_ortho_weight(self.Ux)
        init_bias(self.bx)
        init_ortho_weight(self.Wc_att)
        init_bias(self.b_att)
        init_ortho_weight(self.W_comb_att)
        init_ortho_weight(self.U_att)

    def forward(self, y_emb, context, init_state, x_mask, y_mask):
        
        def _get_word_atten(pctx, s, x_mask):
            h = F.linear(s, self.W_comb_att)
            unreg_att = T.tanh(pctx + h) * x_mask
            unreg_att = F.linear(unreg_att, self.U_att)

            word_atten = T.exp(unreg_att - T.max(unreg_att, 0, keepdim = True)[0]) * x_mask
            sum_word_atten = T.sum(word_atten, 0, keepdim = True)
            word_atten =  word_atten / sum_word_atten
            return word_atten

        def recurrence(x, y_mask, hidden, pctx, context, x_mask):
            pre_h, pre_c = hidden
            
            h1, c1 = self.lstm_1(x, hidden)  
            h1 = y_mask * h1 + (1.0 - y_mask) * pre_h
            c1 = y_mask * c1 + (1.0 - y_mask) * pre_c
            
            # len(x) * batch_size * 1
            s = T.cat((h1.view(-1, self.hidden_size), c1.view(-1, self.hidden_size)), 1)
            word_atten = _get_word_atten(pctx, s, x_mask)
            atted_ctx = T.sum(word_atten * context, 0)

            h2 = T.tanh(F.linear(h1, self.Ux) + F.linear(atted_ctx, self.Wx, self.bx))
            h2 = y_mask * h2 + (1.0 - y_mask) * h1

            return (h1, c1), h2, atted_ctx

        hs = []
        cs = []
        ss = []
        atts = []
        steps = range(y_emb.size(0))
        hidden = init_state #Variable(torch.zeros(y_emb.size(1), self.hidden_size)).to(self.device)
        
        pctx = F.linear(context, self.Wc_att, self.b_att)
        x = y_emb
        for i in steps:
            hidden, s, att = recurrence(x[i], y_mask[i], hidden, pctx, context, x_mask)
            hs.append(hidden[0])
            cs.append(hidden[1])
            ss.append(s)
            atts.append(att)

        hs = T.cat(hs, 0).view(y_emb.size(0), *hs[0].size())
        cs = T.cat(cs, 0).view(y_emb.size(0), *cs[0].size())
        ss = T.cat(ss, 0).view(y_emb.size(0), *ss[0].size())
        atts = T.cat(atts, 0).view(y_emb.size(0), *atts[0].size())
        return (hs, cs), ss, atts


