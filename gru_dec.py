import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
from utils_pg import *

class GRUAttentionDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, ctx_size, device, copy, coverage, is_predicting):
        super(GRUAttentionDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size = ctx_size
        self.is_predicting = is_predicting
        self.device = device
        self.copy = copy
        self.coverage = coverage

        self.W = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.input_size))
        self.U = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(2 * self.hidden_size))
        
        self.Wx = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.Ux = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bx = nn.Parameter(torch.Tensor(self.hidden_size))

        self.Wc_att = nn.Parameter(torch.Tensor(self.ctx_size, self.ctx_size))
        self.b_att = nn.Parameter(torch.Tensor(self.ctx_size))

        self.W_comb_att = nn.Parameter(torch.Tensor(self.ctx_size, self.hidden_size))
        self.U_att = nn.Parameter(torch.Tensor(1, self.ctx_size))
        self.U_nl = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size)) 
        self.b_nl = nn.Parameter(torch.Tensor(2 * self.hidden_size))

        self.Ux_nl = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bx_nl = nn.Parameter(torch.Tensor(self.hidden_size))

        self.Wc = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.ctx_size))
        self.Wcx = nn.Parameter(torch.Tensor(self.hidden_size, self.ctx_size))

        if self.coverage:
            self.W_coverage= nn.Parameter(torch.Tensor(self.ctx_size, 1))

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
        init_ortho_weight(self.U_nl)
        init_bias(self.b_nl)
        init_ortho_weight(self.Ux_nl)
        init_bias(self.bx_nl)
        init_ortho_weight(self.Wc)
        init_ortho_weight(self.Wcx)
        if self.coverage:
            init_ortho_weight(self.W_coverage)


    def forward(self, y_emb, context, init_state, x_mask, y_mask, xid=None, init_coverage=None):

        def _get_word_atten(pctx, h1, x_mask, acc_att=None): #acc_att: B * len(x)
            if acc_att is not None:
                h = F.linear(h1, self.W_comb_att) + F.linear(T.transpose(acc_att, 0, 1).unsqueeze(2), self.W_coverage) # len(x) * B * ?
            else:
                h = F.linear(h1, self.W_comb_att)
            unreg_att = T.tanh(pctx + h) * x_mask
            unreg_att = F.linear(unreg_att, self.U_att)

            word_atten = T.exp(unreg_att - T.max(unreg_att, 0, keepdim = True)[0]) * x_mask
            sum_word_atten = T.sum(word_atten, 0, keepdim = True)
            word_atten =  word_atten / sum_word_atten
            return word_atten

        def recurrence(x, xx, y_mask, pre_h, pctx, context, x_mask, acc_att=None):
            tmp1 = T.sigmoid(F.linear(pre_h, self.U) + x) 
            r1, u1 = tmp1.chunk(2, 1) 
            h1 = T.tanh(F.linear(pre_h * r1, self.Ux) + xx)
            h1 = u1 * pre_h + (1.0 - u1) * h1
            h1 = y_mask * h1 + (1.0 - y_mask) * pre_h

            # len(x) * batch_size * 1
            if self.coverage:
                word_atten = _get_word_atten(pctx, h1, x_mask, acc_att)
            else:
                word_atten = _get_word_atten(pctx, h1, x_mask)
            atted_ctx = T.sum(word_atten * context, 0)

            tmp2 = T.sigmoid(F.linear(atted_ctx, self.Wc) + F.linear(h1, self.U_nl) + self.b_nl)
            r2, u2 = tmp2.chunk(2, 1)  
            h2 = T.tanh(F.linear(atted_ctx, self.Wcx) + F.linear(h1 * r2, self.Ux_nl) + self.bx_nl)
            h2 = u2 * h1 + (1.0 - u2) * h2
            h2 = y_mask * h2 + (1.0 - y_mask) * h1

            word_atten_ = T.transpose(word_atten.view(x_mask.size(0), -1), 0, 1) # B * len(x)
            if self.coverage:
                acc_att += word_atten_
                return h2, h2, atted_ctx, word_atten_, acc_att
            else:
                return h2, h2, atted_ctx, word_atten_


        hs, ss, atts, dists, xids, cs = [], [], [], [], [], []
        hidden = init_state
        acc_att = init_coverage
        
        if self.copy:
            xid = T.transpose(xid, 0, 1) # B * len(x)
        
        pctx = F.linear(context, self.Wc_att, self.b_att)
        x = F.linear(y_emb, self.W, self.b)
        xx = F.linear(y_emb, self.Wx, self.bx)
        
        steps = range(y_emb.size(0))
        for i in steps:
            if self.coverage:
                cs += [acc_att]
                hidden, s, att, att_dist, acc_att = recurrence(x[i], xx[i], y_mask[i], hidden, pctx, context, x_mask, acc_att)
            else:
                hidden, s, att, att_dist = recurrence(x[i], xx[i], y_mask[i], hidden, pctx, context, x_mask)
            hs += [hidden]
            ss += [s]
            atts += [att]
            dists += [att_dist]
            xids += [xid]
        
        if self.coverage:
            if self.is_predicting :
                cs += [acc_att]
                cs = cs[1:]
            cs = T.stack(cs).view(y_emb.size(0), *cs[0].size())
        
        hs = T.stack(hs).view(y_emb.size(0), *hs[0].size())
        ss = T.stack(ss).view(y_emb.size(0), *ss[0].size())
        atts = T.stack(atts).view(y_emb.size(0), *atts[0].size())
        dists = T.stack(dists).view(y_emb.size(0), *dists[0].size())
        if self.copy:
            xids = T.stack(xids).view(y_emb.size(0), *xids[0].size())
        
        if self.copy and self.coverage:
            return hs, ss, atts, dists, xids, cs
        elif self.copy:
            return hs, ss, atts, dists, xids
        elif self.coverage:
            return hs, ss, atts, dists, cs
        else:
            return hs, ss, atts

