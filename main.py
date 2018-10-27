# -*- coding: utf-8 -*-
import os
cudaid = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaid)

import sys
import time
import numpy as np
import cPickle as pickle
import copy
import random
from random import shuffle
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import data as datar
from model import *
from utils_pg import *
from configs import *

cfg = DeepmindConfigs()
TRAINING_DATASET_CLS = DeepmindTraining
TESTING_DATASET_CLS = DeepmindTesting

def print_basic_info(modules, consts, options):
    if options["is_debugging"]:
        print "\nWARNING: IN DEBUGGING MODE\n"
    if options["copy"]:
        print "USE COPY MECHANISM"
    if options["coverage"]:
        print "USE COVERAGE MECHANISM"
    if  options["avg_nll"]:
        print "USE AVG NLL as LOSS"
    else:
        print "USE NLL as LOSS"
    if options["has_learnable_w2v"]:
        print "USE LEARNABLE W2V EMBEDDING"
    if options["is_bidirectional"]:
        print "USE BI-DIRECTIONAL RNN"
    if options["omit_eos"]:
        print "<eos> IS OMITTED IN TESTING DATA"
    if options["prediction_bytes_limitation"]:
        print "MAXIMUM BYTES IN PREDICTION IS LIMITED"
    print "RNN TYPE: " + options["cell"]
    for k in consts:
        print k + ":", consts[k]

def init_modules():
    
    init_seeds()

    options = {}

    options["is_debugging"] = False
    options["is_predicting"] = False
    options["model_selection"] = False # When options["is_predicting"] = True, true means use validation set for tuning, false is real testing.

    options["cuda"] = cfg.CUDA and torch.cuda.is_available()
    options["device"] = torch.device("cuda" if  options["cuda"] else "cpu")
    
    #in config.py
    options["cell"] = cfg.CELL
    options["copy"] = cfg.COPY
    options["coverage"] = cfg.COVERAGE
    options["is_bidirectional"] = cfg.BI_RNN
    options["avg_nll"] = cfg.AVG_NLL

    options["beam_decoding"] = cfg.BEAM_SEARCH # False for greedy decoding
    
    assert TRAINING_DATASET_CLS.IS_UNICODE == TESTING_DATASET_CLS.IS_UNICODE
    options["is_unicode"] = TRAINING_DATASET_CLS.IS_UNICODE # True Chinese dataet
    options["has_y"] = TRAINING_DATASET_CLS.HAS_Y
    
    options["has_learnable_w2v"] = True
    options["omit_eos"] = False # omit <eos> and continuously decode until length of sentence reaches MAX_LEN_PREDICT (for DUC testing data)
    options["prediction_bytes_limitation"] = False if TESTING_DATASET_CLS.MAX_BYTE_PREDICT == None else True

    assert options["is_unicode"] == False

    consts = {}
    
    consts["idx_gpu"] = cudaid

    consts["norm_clip"] = cfg.NORM_CLIP
    consts["dim_x"] = cfg.DIM_X
    consts["dim_y"] = cfg.DIM_Y
    consts["len_x"] = cfg.MAX_LEN_X + 1 # plus 1 for eos
    consts["len_y"] = cfg.MAX_LEN_Y + 1
    consts["num_x"] = cfg.MAX_NUM_X
    consts["num_y"] = cfg.NUM_Y
    consts["hidden_size"] = cfg.HIDDEN_SIZE

    consts["batch_size"] = 5 if options["is_debugging"] else TRAINING_DATASET_CLS.BATCH_SIZE
    if options["is_debugging"]:
        consts["testing_batch_size"] = 1 if options["beam_decoding"] else 2
    else:
        #consts["testing_batch_size"] = 1 if options["beam_decoding"] else TESTING_DATASET_CLS.BATCH_SIZE 
        consts["testing_batch_size"] = TESTING_DATASET_CLS.BATCH_SIZE

    consts["min_len_predict"] = TESTING_DATASET_CLS.MIN_LEN_PREDICT
    consts["max_len_predict"] = TESTING_DATASET_CLS.MAX_LEN_PREDICT
    consts["max_byte_predict"] = TESTING_DATASET_CLS.MAX_BYTE_PREDICT
    consts["testing_print_size"] = TESTING_DATASET_CLS.PRINT_SIZE

    consts["lr"] = cfg.LR
    consts["beam_size"] = cfg.BEAM_SIZE

    consts["max_epoch"] = 150 if options["is_debugging"] else 30 
    consts["print_time"] = 5
    consts["save_epoch"] = 1

    assert consts["dim_x"] == consts["dim_y"]
    assert consts["beam_size"] >= 1

    modules = {}
    
    [_, dic, hfw, w2i, i2w, w2w] = pickle.load(open(cfg.cc.TRAINING_DATA_PATH + "dic.pkl", "r")) 
    consts["dict_size"] = len(dic)
    modules["dic"] = dic
    modules["w2i"] = w2i
    modules["i2w"] = i2w
    modules["lfw_emb"] = modules["w2i"][cfg.W_UNK]
    modules["eos_emb"] = modules["w2i"][cfg.W_EOS]
    consts["pad_token_idx"] = modules["w2i"][cfg.W_PAD]

    return modules, consts, options

def greedy_decode(flist, batch, model, modules, consts, options):
    testing_batch_size = len(flist)

    dec_result = [[] for i in xrange(testing_batch_size)]
    existence = [True] * testing_batch_size
    num_left = testing_batch_size

    if options["copy"]:
        x, word_emb, dec_state, x_mask, y, len_y, ref_sents, max_ext_len, oovs = batch
    else:
        x, word_emb, dec_state, x_mask, y, len_y, ref_sents = batch

    next_y = torch.LongTensor(-np.ones((1, testing_batch_size), dtype="int64")).to(options["device"])

    if options["cell"] == "lstm":
        dec_state = (dec_state, dec_state)
    if options["coverage"]:
        acc_att = Variable(torch.zeros(T.transpose(x, 0, 1).size())).to(options["device"]) # B *len(x)
     
    for step in xrange(consts["max_len_predict"]):
        if num_left == 0:
            break
        if options["copy"] and options["coverage"]:
            y_pred, dec_state, acc_att = model.decode_once(next_y, word_emb, dec_state, x_mask, x, max_ext_len, acc_att)
        elif options["copy"]:
            y_pred, dec_state = model.decode_once(next_y, word_emb, dec_state, x_mask, x, max_ext_len)
        elif options["coverage"]:
            y_pred, dec_state, acc_att = model.decode_once(next_y, word_emb, dec_state, x_mask, acc_att=acc_att)
        else:
            y_pred, dec_state = model.decode_once(next_y, word_emb, dec_state, x_mask)

        dict_size = y_pred.shape[-1]
        y_pred = y_pred.view(testing_batch_size, dict_size)
        next_y_ = torch.argmax(y_pred, 1)
        next_y = []
        for e in range(testing_batch_size):
            eid = next_y_[e].item()
            if eid in modules["i2w"]:
                next_y.append(eid)
            else:
                next_y.append(modules["lfw_emb"]) # unk for copy mechanism
        next_y = np.array(next_y).reshape((1, testing_batch_size))
        next_y = torch.LongTensor(next_y).to(options["device"])

        if options["coverage"]:
            acc_att = acc_att.view(testing_batch_size, acc_att.shape[-1])

        if options["cell"] == "lstm":
            dec_state = (dec_state[0].view(testing_batch_size, dec_state[0].shape[-1]), dec_state[1].view(testing_batch_size, dec_state[1].shape[-1]))
        else:
            dec_state = dec_state.view(testing_batch_size, dec_state.shape[-1])

        for idx_doc in xrange(testing_batch_size):
            if existence[idx_doc] == False:
                continue

            idx_max = next_y[0, idx_doc].item()
            if idx_max == modules["eos_emb"] and len(dec_result[idx_doc]) >= consts["min_len_predict"]:
                existence[idx_doc] = False
                num_left -= 1
            else:
                dec_result[idx_doc].append(str(idx_max))
    
    # for task with bytes-length limitation 
    if options["prediction_bytes_limitation"]:
        for i in xrange(len(dec_result)):
            sample = dec_result[i]
            b = 0
            for j in xrange(len(sample)):
                e = int(sample[j]) 
                if e in modules["i2w"]:
                    word = modules["i2w"][e]
                else:
                    word = oovs[e - len(modules["i2w"])]
                if j == 0:
                    b += len(word)
                else:
                    b += len(word) + 1 
                if b > consts["max_byte_predict"]:
                    sorted_samples[i] = sorted_samples[i][0 : j]
                    break

    for idx_doc in xrange(testing_batch_size):
        fname = str(flist[idx_doc])
        if len(dec_result[idx_doc]) >= consts["min_len_predict"]:
            dec_words = []
            for e in dec_result[idx_doc]:
                e = int(e)
                if e in modules["i2w"]: # if not copy, the word are all in dict
                    dec_words.append(modules["i2w"][e])
                else:
                    dec_words.append(oovs[e - len(modules["i2w"])])
            write_for_rouge(fname, ref_sents[idx_doc], dec_words, cfg)
        else:
            print "ERROR: " + fname


def beam_decode(fname, batch, model, modules, consts, options):
    fname = str(fname)

    beam_size = consts["beam_size"]
    num_live = 1
    num_dead = 0
    samples = []
    sample_scores = np.zeros(beam_size)

    last_traces = [[]]
    last_scores = torch.FloatTensor(np.zeros(1)).to(options["device"])
    last_states = []

    if options["copy"]:
        x, word_emb, dec_state, x_mask, y, len_y, ref_sents, max_ext_len, oovs = batch
    else:
        x, word_emb, dec_state, x_mask, y, len_y, ref_sents = batch
    
    next_y = torch.LongTensor(-np.ones((1, num_live), dtype="int64")).to(options["device"])
    x = x.unsqueeze(1)
    word_emb = word_emb.unsqueeze(1)
    x_mask = x_mask.unsqueeze(1)
    dec_state = dec_state.unsqueeze(0)
    if options["cell"] == "lstm":
        dec_state = (dec_state, dec_state)
    
    if options["coverage"]:
        acc_att = Variable(torch.zeros(T.transpose(x, 0, 1).size())).to(options["device"]) # B *len(x)
        last_acc_att = [] 

    for step in xrange(consts["max_len_predict"]):
        tile_word_emb = word_emb.repeat(1, num_live, 1)
        tile_x_mask = x_mask.repeat(1, num_live, 1)
        if options["copy"]:
            tile_x = x.repeat(1, num_live)

        if options["copy"] and options["coverage"]:
            y_pred, dec_state, acc_att = model.decode_once(next_y, tile_word_emb, dec_state, tile_x_mask, tile_x, max_ext_len, acc_att)
        elif options["copy"]:
            y_pred, dec_state = model.decode_once(next_y, tile_word_emb, dec_state, tile_x_mask, tile_x, max_ext_len)
        elif options["coverage"]:
            y_pred, dec_state, acc_att = model.decode_once(next_y, tile_word_emb, dec_state, tile_x_mask, acc_att=acc_att)
        else:
            y_pred, dec_state = model.decode_once(next_y, tile_word_emb, dec_state, tile_x_mask)
        dict_size = y_pred.shape[-1]
        y_pred = y_pred.view(num_live, dict_size)
        if options["coverage"]:
            acc_att = acc_att.view(num_live, acc_att.shape[-1])

        if options["cell"] == "lstm":
            dec_state = (dec_state[0].view(num_live, dec_state[0].shape[-1]), dec_state[1].view(num_live, dec_state[1].shape[-1]))
        else:
            dec_state = dec_state.view(num_live, dec_state.shape[-1])
  
        cand_scores = last_scores + torch.log(y_pred) # larger is better
        cand_scores = cand_scores.flatten()
        idx_top_joint_scores = torch.topk(cand_scores, beam_size - num_dead)[1]


        idx_last_traces = idx_top_joint_scores / dict_size
        idx_word_now = idx_top_joint_scores % dict_size
        top_joint_scores = cand_scores[idx_top_joint_scores]

        traces_now = []
        scores_now = np.zeros((beam_size - num_dead))
        states_now = []
        if options["coverage"]: 
            acc_att_now = []
            last_acc_att = []
        
        for i, [j, k] in enumerate(zip(idx_last_traces, idx_word_now)):
            traces_now.append(last_traces[j] + [k])
            scores_now[i] = copy.copy(top_joint_scores[i])
            if options["cell"] == "lstm":
                states_now.append((copy.copy(dec_state[0][j, :]), copy.copy(dec_state[1][j, :])))
            else:
                states_now.append(copy.copy(dec_state[j, :]))
            if options["coverage"]:
                acc_att_now.append(copy.copy(acc_att[j, :]))

        num_live = 0
        last_traces = []
        last_scores = []
        last_states = []
        for i in xrange(len(traces_now)):
            if traces_now[i][-1] == modules["eos_emb"] and len(traces_now[i]) >= consts["min_len_predict"]:
                samples.append([str(e.item()) for e in traces_now[i][:-1]])
                sample_scores[num_dead] = scores_now[i]
                num_dead += 1
            else:
                last_traces.append(traces_now[i])
                last_scores.append(scores_now[i])
                last_states.append(states_now[i])
                if options["coverage"]:
                    last_acc_att.append(acc_att_now[i])
                num_live += 1
        if num_live == 0 or num_dead >= beam_size:
            break

        last_scores = torch.FloatTensor(np.array(last_scores).reshape((num_live, 1))).to(options["device"])
        next_y = []
        for e in last_traces:
            eid = e[-1].item()
            if eid in modules["i2w"]:
                next_y.append(eid)
            else:
                next_y.append(modules["lfw_emb"]) # unk for copy mechanism

        next_y = np.array(next_y).reshape((1, num_live))
        next_y = torch.LongTensor(next_y).to(options["device"])
        if options["cell"] == "lstm":
            h_states = []
            c_states = []
            for state in last_states:
                h_states.append(state[0])
                c_states.append(state[1])
            dec_state = (torch.stack(h_states).view((num_live, h_states[0].shape[-1])),\
                         torch.stack(c_states).view((num_live, c_states[0].shape[-1])))
        else:
            dec_state = torch.stack(last_states).view((num_live, dec_state.shape[-1]))
        if options["coverage"]:
            acc_att = torch.stack(last_acc_att).view((num_live, acc_att.shape[-1])) 
            
        assert num_live + num_dead == beam_size

    if num_live > 0:
        for i in xrange(num_live):
            samples.append([str(e.item()) for e in last_traces[i]])
            sample_scores[num_dead] = last_scores[i]
            num_dead += 1
    
    #weight by length
    for i in xrange(len(sample_scores)):
        sent_len = float(len(samples[i]))
        sample_scores[i] = sample_scores[i] / sent_len #avg is better than sum.   #*  math.exp(-sent_len / 10)

    idx_sorted_scores = np.argsort(sample_scores) # ascending order
    if options["has_y"]:
        ly = len_y[0]
        y_true = y[0 : ly].tolist()
        y_true = [str(i) for i in y_true[:-1]] # delete <eos>

    sorted_samples = []
    sorted_scores = []
    filter_idx = []
    for e in idx_sorted_scores:
        if len(samples[e]) >= consts["min_len_predict"]:
            filter_idx.append(e)
    if len(filter_idx) == 0:
        filter_idx = idx_sorted_scores
    for e in filter_idx:
        sorted_samples.append(samples[e])
        sorted_scores.append(sample_scores[e])

    num_samples = len(sorted_samples)
    if len(sorted_samples) == 1:
        sorted_samples = sorted_samples[0]
        num_samples = 1

    # for task with bytes-length limitation 
    if options["prediction_bytes_limitation"]:
        for i in xrange(len(sorted_samples)):
            sample = sorted_samples[i]
            b = 0
            for j in xrange(len(sample)):
                e = int(sample[j]) 
                if e in modules["i2w"]:
                    word = modules["i2w"][e]
                else:
                    word = oovs[e - len(modules["i2w"])]
                if j == 0:
                    b += len(word)
                else:
                    b += len(word) + 1 
                if b > consts["max_byte_predict"]:
                    sorted_samples[i] = sorted_samples[i][0 : j]
                    break

    dec_words = []

    for e in sorted_samples[-1]:
        e = int(e)
        if e in modules["i2w"]: # if not copy, the word are all in dict
            dec_words.append(modules["i2w"][e])
        else:
            dec_words.append(oovs[e - len(modules["i2w"])])
    
    write_for_rouge(fname, ref_sents, dec_words, cfg)

    # beam search history for checking
    if not options["copy"]:
        oovs = None
    write_summ("".join((cfg.cc.BEAM_SUMM_PATH, fname)), sorted_samples, num_samples, options, modules["i2w"], oovs, sorted_scores)
    write_summ("".join((cfg.cc.BEAM_GT_PATH, fname)), y_true, 1, options, modules["i2w"], oovs) 


def predict(model, modules, consts, options):
    print "start predicting,"
    options["has_y"] = TESTING_DATASET_CLS.HAS_Y
    if options["beam_decoding"]:
        print "using beam search"
    else:
        print "using greedy search"
    rebuild_dir(cfg.cc.BEAM_SUMM_PATH)
    rebuild_dir(cfg.cc.BEAM_GT_PATH)
    rebuild_dir(cfg.cc.GROUND_TRUTH_PATH)
    rebuild_dir(cfg.cc.SUMM_PATH)

    print "loading test set..."
    if options["model_selection"]:
        xy_list = pickle.load(open(cfg.cc.VALIDATE_DATA_PATH + "pj1000.pkl", "r")) 
    else:
        xy_list = pickle.load(open(cfg.cc.TESTING_DATA_PATH + "test.pkl", "r")) 
    batch_list, num_files, num_batches = datar.batched(len(xy_list), options, consts)

    print "num_files = ", num_files, ", num_batches = ", num_batches
    
    running_start = time.time()
    partial_num = 0
    total_num = 0
    si = 0
    for idx_batch in xrange(num_batches):
        test_idx = batch_list[idx_batch]
        batch_raw = [xy_list[xy_idx] for xy_idx in test_idx]
        batch = datar.get_data(batch_raw, modules, consts, options)
        
        assert len(test_idx) == batch.x.shape[1] # local_batch_size

        x, len_x, x_mask, y, len_y, y_mask, oy, x_ext, y_ext, oovs = sort_samples(batch.x, batch.len_x, \
                                                             batch.x_mask, batch.y, batch.len_y, batch.y_mask, \
                                                             batch.original_summarys, batch.x_ext, batch.y_ext, batch.x_ext_words)
                    
        word_emb, dec_state = model.encode(torch.LongTensor(x).to(options["device"]),\
                                           torch.LongTensor(len_x).to(options["device"]),\
                                           torch.FloatTensor(x_mask).to(options["device"]))

        if options["beam_decoding"]:
            for idx_s in xrange(len(test_idx)):
                if options["copy"]:
                    inputx = (torch.LongTensor(x_ext[:, idx_s]).to(options["device"]), word_emb[:, idx_s, :], dec_state[idx_s, :],\
                          torch.FloatTensor(x_mask[:, idx_s, :]).to(options["device"]), y[:, idx_s], [len_y[idx_s]], oy[idx_s],\
                          batch.max_ext_len, oovs[idx_s])
                else:
                    inputx = (torch.LongTensor(x[:, idx_s]).to(options["device"]), word_emb[:, idx_s, :], dec_state[idx_s, :],\
                          torch.FloatTensor(x_mask[:, idx_s, :]).to(options["device"]), y[:, idx_s], [len_y[idx_s]], oy[idx_s])

                beam_decode(si, inputx, model, modules, consts, options)
                si += 1
        else:
            if options["copy"]:
                inputx = (torch.LongTensor(x_ext).to(options["device"]), word_emb, dec_state, \
                          torch.FloatTensor(x_mask).to(options["device"]), y, len_y, oy, batch.max_ext_len, oovs)
            else:
                inputx = (torch.LongTensor(x).to(options["device"]), word_emb, dec_state, torch.FloatTensor(x_mask).to(options["device"]), y, len_y, oy)
            greedy_decode(test_idx, inputx, model, modules, consts, options)
            si += len(test_idx)

        testing_batch_size = len(test_idx)
        partial_num += testing_batch_size
        total_num += testing_batch_size
        if partial_num >= consts["testing_print_size"]:
            print total_num, "summs are generated"
            partial_num = 0
    print si, total_num

def run(existing_model_name = None):
    modules, consts, options = init_modules()

    #use_gpu(consts["idx_gpu"])
    if options["is_predicting"]:
        need_load_model = True
        training_model = False
        predict_model = True
    else:
        need_load_model = False
        training_model = True
        predict_model = False

    print_basic_info(modules, consts, options)

    if training_model:
        print "loading train set..."
        if options["is_debugging"]:
            xy_list = pickle.load(open(cfg.cc.VALIDATE_DATA_PATH + "pj1000.pkl", "r")) 
        else:
            xy_list = pickle.load(open(cfg.cc.TRAINING_DATA_PATH + "train.pkl", "r")) 
        batch_list, num_files, num_batches = datar.batched(len(xy_list), options, consts)
        print "num_files = ", num_files, ", num_batches = ", num_batches

    running_start = time.time()
    if True: #TODO: refactor
        print "compiling model ..." 
        model = Model(modules, consts, options)
        if options["cuda"]:
            model.cuda()
        optimizer = torch.optim.Adagrad(model.parameters(), lr=consts["lr"], initial_accumulator_value=0.1)
        
        model_name = "".join(["cnndm.s2s.", options["cell"]])
        existing_epoch = 0
        if need_load_model:
            if existing_model_name == None:
                existing_model_name = "cnndm.s2s.gpu4.epoch7.1"
            print "loading existed model:", existing_model_name
            model, optimizer = load_model(cfg.cc.MODEL_PATH + existing_model_name, model, optimizer)

        if training_model:
            print "start training model "
            print_size = num_files / consts["print_time"] if num_files >= consts["print_time"] else num_files

            last_total_error = float("inf")
            print "max epoch:", consts["max_epoch"]
            for epoch in xrange(0, consts["max_epoch"]):
                print "epoch: ", epoch + existing_epoch
                num_partial = 1
                total_error = 0.0
                error_c = 0.0
                partial_num_files = 0
                epoch_start = time.time()
                partial_start = time.time()
                # shuffle the trainset
                batch_list, num_files, num_batches = datar.batched(len(xy_list), options, consts)
                used_batch = 0.
                for idx_batch in xrange(num_batches):
                    train_idx = batch_list[idx_batch]
                    batch_raw = [xy_list[xy_idx] for xy_idx in train_idx]
                    if len(batch_raw) != consts["batch_size"]:
                        continue
                    local_batch_size = len(batch_raw)
                    batch = datar.get_data(batch_raw, modules, consts, options)
                  
                    x, len_x, x_mask, y, len_y, y_mask, oy, x_ext, y_ext, oovs = sort_samples(batch.x, batch.len_x, \
                                                             batch.x_mask, batch.y, batch.len_y, batch.y_mask, \
                                                             batch.original_summarys, batch.x_ext, batch.y_ext, batch.x_ext_words)
                    
                    model.zero_grad()
                    y_pred, cost, cost_c = model(torch.LongTensor(x).to(options["device"]), torch.LongTensor(len_x).to(options["device"]),\
                                   torch.LongTensor(y).to(options["device"]),  torch.FloatTensor(x_mask).to(options["device"]), \
                                   torch.FloatTensor(y_mask).to(options["device"]), torch.LongTensor(x_ext).to(options["device"]),\
                                   torch.LongTensor(y_ext).to(options["device"]), \
                                   batch.max_ext_len)
                    if cost_c is None:
                        loss = cost
                    else:
                        loss = cost + cost_c
                        cost_c = cost_c.item()
                        error_c += cost_c
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), consts["norm_clip"])
                    optimizer.step()
                    
                    cost = cost.item()
                    total_error += cost
                    used_batch += 1
                    partial_num_files += consts["batch_size"]
                    if partial_num_files / print_size == 1 and idx_batch < num_batches:
                        print idx_batch + 1, "/" , num_batches, "batches have been processed,", 
                        print "average cost until now:", "cost =", total_error / used_batch, ",", 
                        print "cost_c =", error_c / used_batch, ",",
                        print "time:", time.time() - partial_start
                        partial_num_files = 0
                        if not options["is_debugging"]:
                            print "save model... ",
                            save_model(cfg.cc.MODEL_PATH + model_name + ".gpu" + str(consts["idx_gpu"]) + ".epoch" + str(epoch / consts["save_epoch"] + existing_epoch) + "." + str(num_partial), model, optimizer)
                            print "finished"
                        num_partial += 1
                print "in this epoch, total average cost =", total_error / used_batch, ",", 
                print "cost_c =", error_c / used_batch, ",",
                print "time:", time.time() - epoch_start

                print_sent_dec(y_pred, y_ext, y_mask, oovs, modules, consts, options, local_batch_size)
                
                if last_total_error > total_error or options["is_debugging"]:
                    last_total_error = total_error
                    if not options["is_debugging"]:
                        print "save model... ",
                        save_model(cfg.cc.MODEL_PATH + model_name + ".gpu" + str(consts["idx_gpu"]) + ".epoch" + str(epoch / consts["save_epoch"] + existing_epoch) + "." + str(num_partial), model, optimizer)
                        print "finished"
                else:
                    print "optimization finished"
                    break

            print "save final model... ",
            save_model(cfg.cc.MODEL_PATH + model_name + ".final.gpu" + str(consts["idx_gpu"]) + ".epoch" + str(epoch / consts["save_epoch"] + existing_epoch) + "." + str(num_partial), model, optimizer)
            print "finished"
        else:
            print "skip training model"

        if predict_model:
            predict(model, modules, consts, options)
    print "Finished, time:", time.time() - running_start

if __name__ == "__main__":
    np.set_printoptions(threshold = np.inf)
    existing_model_name = sys.argv[1] if len(sys.argv) > 1 else None
    run(existing_model_name)
