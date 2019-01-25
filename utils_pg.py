# -*- coding: utf-8 -*-
#pylint: skip-file
import numpy as np
from numpy.random import random as rand
import pickle
import sys
import os
import shutil
from copy import deepcopy
import random

import torch
from torch import nn


def init_seeds():
    random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

def init_lstm_weight(lstm):
    for param in lstm.parameters():
        if len(param.shape) >= 2: # weights
            init_ortho_weight(param.data)
        else: # bias
            init_bias(param.data)

def init_gru_weight(gru):
    for param in gru.parameters():
        if len(param.shape) >= 2: # weights
            init_ortho_weight(param.data)
        else: # bias
            init_bias(param.data)

def init_linear_weight(linear):
    init_xavier_weight(linear.weight)
    if linear.bias is not None:
        init_bias(linear.bias)

def init_normal_weight(w):
    nn.init.normal_(w, mean=0, std=0.01)

def init_uniform_weight(w):
    nn.init.uniform_(w, -0.1, 0.1)

def init_ortho_weight(w):
    nn.init.orthogonal_(w)

def init_xavier_weight(w):
    nn.init.xavier_normal_(w)

def init_bias(b):
    nn.init.constant_(b, 0.)

def rebuild_dir(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except OSError:
            pass
    os.mkdir(path)

def save_model(f, model, optimizer):
    torch.save({"model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict()},
            f)
 
def load_model(f, model, optimizer):
    checkpoint = torch.load(f)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer

def sort_samples(x, len_x, mask_x, y, len_y, \
                 mask_y, oys, x_ext, y_ext, oovs):
    sorted_x_idx = np.argsort(len_x)[::-1]
    
    sorted_x_len = np.array(len_x)[sorted_x_idx]
    sorted_x = x[:, sorted_x_idx]
    sorted_x_mask = mask_x[:, sorted_x_idx, :]
    sorted_oovs = [oovs[i] for i in sorted_x_idx]

    sorted_y_len = np.array(len_y)[sorted_x_idx]
    sorted_y = y[:, sorted_x_idx]
    sorted_y_mask = mask_y[:, sorted_x_idx, :]
    sorted_oys = [oys[i] for i in sorted_x_idx]
    sorted_x_ext = x_ext[:, sorted_x_idx]
    sorted_y_ext = y_ext[:, sorted_x_idx]
    
    return sorted_x, sorted_x_len, sorted_x_mask, sorted_y, \
           sorted_y_len, sorted_y_mask, sorted_oys, \
           sorted_x_ext, sorted_y_ext, sorted_oovs

def print_sent_dec(y_pred, y, y_mask, oovs, modules, consts, options, batch_size):
    print("golden truth and prediction samples:")
    max_y_words = np.sum(y_mask, axis = 0)
    max_y_words = max_y_words.reshape((batch_size))
    max_num_docs = 16 if batch_size > 16 else batch_size
    is_unicode = options["is_unicode"]
    dict_size = len(modules["i2w"])
    for idx_doc in range(max_num_docs):
        print(idx_doc + 1, "----------------------------------------------------------------------------------------------------")
        sent_true= ""
        for idx_word in range(max_y_words[idx_doc]):
            i = y[idx_word, idx_doc] if options["has_learnable_w2v"] else np.argmax(y[idx_word, idx_doc]) 
            if i in modules["i2w"]:
                sent_true += modules["i2w"][i]
            else:
                sent_true += oovs[idx_doc][i - dict_size]
            if not is_unicode:
                sent_true += " "

        if is_unicode:
            print(sent_true.encode("utf-8"))
        else:
            print(sent_true)

        print()

        sent_pred = ""
        for idx_word in range(max_y_words[idx_doc]):
            i = torch.argmax(y_pred[idx_word, idx_doc, :]).item()
            if i in modules["i2w"]:
                sent_pred += modules["i2w"][i]
            else:
                sent_pred += oovs[idx_doc][i - dict_size]
            if not is_unicode:
                sent_pred += " "
        if is_unicode:
            print(sent_pred.encode("utf-8"))
        else:
            print(sent_pred)
    print("----------------------------------------------------------------------------------------------------")
    print()


def write_for_rouge(fname, ref_sents, dec_words, cfg):
    dec_sents = []
    while len(dec_words) > 0:
        try:
            fst_period_idx = dec_words.index(".")
        except ValueError:
            fst_period_idx = len(dec_words)
        sent = dec_words[:fst_period_idx + 1]
        dec_words = dec_words[fst_period_idx + 1:]
        dec_sents.append(' '.join(sent))

    ref_file = "".join((cfg.cc.GROUND_TRUTH_PATH, fname))
    decoded_file = "".join((cfg.cc.SUMM_PATH, fname))

    with open(ref_file, "w") as f:
        for idx, sent in enumerate(ref_sents):
            sent = sent.strip()
            f.write(sent) if idx == len(ref_sents) - 1 else f.write(sent + "\n")
    with open(decoded_file, "w") as f:
        for idx, sent in enumerate(dec_sents):
            sent = sent.strip()
            f.write(sent) if idx == len(dec_sents) - 1 else f.write(sent + "\n")

def write_summ(dst_path, summ_list, num_summ, options, i2w = None, oovs=None, score_list = None):
    assert num_summ > 0
    with open(dst_path, "w") as f_summ:
        if num_summ == 1:
            if score_list != None:
                f_summ.write(str(score_list[0]))
                f_summ.write("\t")
            if i2w != None:
                '''
                for e in summ_list:
                    e = int(e)
                    if e in i2w:
                        print i2w[e],
                    else:
                        print oovs[e - len(i2w)],
                print "\n"
                '''
                s = []
                for e in summ_list:
                    e = int(e)
                    if e in i2w:
                        s.append(i2w[e])
                    else:
                        s.append(oovs[e - len(i2w)])
                s = " ".join(s)
            else:
                s = " ".join(summ_list)
            f_summ.write(s)
            f_summ.write("\n")
        else:
            assert num_summ == len(summ_list)
            if score_list != None:
                assert num_summ == len(score_list)

            for i in range(num_summ):
                if score_list != None:
                    f_summ.write(str(score_list[i]))
                    f_summ.write("\t")
                if i2w != None:
                    '''
                    for e in summ_list[i]:
                        e = int(e)
                        if e in i2w:
                            print i2w[e],
                        else:
                            print oovs[e - len(i2w)],
                    print "\n"
                    '''
                    s = []
                    for e in summ_list[i]:
                        e = int(e)
                        if e in i2w:
                            s.append(i2w[e])
                        else:
                            s.append(oovs[e - len(i2w)])
                    s = " ".join(s)
                else:
                    s = " ".join(summ_list[i])

                f_summ.write(s)
                f_summ.write("\n")


