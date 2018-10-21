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

        self.W = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.input_size))
        self.U = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(4 * self.hidden_size))
        
        self.Wx = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.ctx_size))
        self.Ux = nn.Parameter(torch.Tensor(4 * self.hidden_size, self.hidden_size))
        self.bx = nn.Parameter(torch.Tensor(4 * self.hidden_size))

        self.Wc_att = nn.Parameter(torch.Tensor(self.ctx_size, self.ctx_size))
        self.b_att = nn.Parameter(torch.Tensor(self.ctx_size))

        self.W_comb_att = nn.Parameter(torch.Tensor(self.ctx_size, self.hidden_size))
        self.U_att = nn.Parameter(torch.Tensor(1, self.ctx_size))

        self.init_weights()

    def init_weights(self):
        init_ortho_weight(self.W)
        init_ortho_weight(self.U)
        init_bias(self.b)
        init_ortho_weight(self.Wx)
        init_ortho_weight(self.Ux)
        init_bias(self.bx)
        init_ortho_weight(self.Wc_att)
        init_bias(self.b_att)
        init_ortho_weight(self.W_comb_att)
        init_ortho_weight(self.U_att)

    def forward(self, y_emb, context, init_state, x_mask, y_mask):
        
        def _get_word_atten(pctx, h1, x_mask):
            h = F.linear(h1, self.W_comb_att)
            unreg_att = T.tanh(pctx + h) * x_mask
            unreg_att = F.linear(unreg_att, self.U_att)

            word_atten = T.exp(unreg_att - T.max(unreg_att, 0, keepdim = True)[0]) * x_mask
            sum_word_atten = T.sum(word_atten, 0, keepdim = True)
            word_atten =  word_atten / sum_word_atten
            return word_atten

        def recurrence(x, y_mask, hidden, pctx, context, x_mask):
            pre_h, pre_c = hidden

            ifoc_preact1 = x + F.linear(pre_h, self.U)
            x4i1, x4f1, x4o1, x4c1 = ifoc_preact1.chunk(4, 1)
            i1 = torch.sigmoid(x4i1)
            f1 = torch.sigmoid(x4f1)
            o1 = torch.sigmoid(x4o1)
            c1 = f1 * pre_c + i1 * torch.tanh(x4c1)
            h1 = o1 * torch.tanh(c1)
            h1 = y_mask * h1 + (1.0 - y_mask) * pre_h

            # len(x) * batch_size * 1
            word_atten = _get_word_atten(pctx, h1, x_mask)
            atted_ctx = T.sum(word_atten * context, 0)

            ifoc_preact1 = F.linear(h1, self.Ux) + F.linear(atted_ctx, self.Wx, self.bx)
            x4i1, x4f1, x4o1, x4c1 = ifoc_preact1.chunk(4, 1)
            i1 = torch.sigmoid(x4i1)
            f1 = torch.sigmoid(x4f1)
            o1 = torch.sigmoid(x4o1)
            c2 = f1 * c1 + i1 * torch.tanh(x4c1)
            h2 = o1 * torch.tanh(c2)
            h2 = y_mask * h2 + (1.0 - y_mask) * h1


            return (h2, c2), atted_ctx

        hs = []
        cs = []
        atts = []
        steps = range(y_emb.size(0))
        hidden = init_state #Variable(torch.zeros(y_emb.size(1), self.hidden_size)).to(self.device)
        
        pctx = F.linear(context, self.Wc_att, self.b_att)
        x = F.linear(y_emb, self.W, self.b)
        for i in steps:
            hidden, att = recurrence(x[i], y_mask[i], hidden, pctx, context, x_mask)
            hs.append(hidden[0])
            cs.append(hidden[1])
            atts.append(att)

        hs = T.cat(hs, 0).view(y_emb.size(0), *hs[0].size())
        cs = T.cat(cs, 0).view(y_emb.size(0), *cs[0].size())
        atts = T.cat(atts, 0).view(y_emb.size(0), *atts[0].size())
        return (hs, cs), atts


