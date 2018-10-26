# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import os
import os.path
import time
from operator import itemgetter
import numpy as np
import cPickle as pickle
from random import shuffle

class BatchData:
    def __init__(self, flist, modules, consts, options):
        self.batch_size = len(flist) 
        self.x = np.zeros((consts["len_x"], self.batch_size), dtype = np.int64)
        self.x_ext = np.zeros((consts["len_x"], self.batch_size), dtype = np.int64)
        self.y = np.zeros((consts["len_y"], self.batch_size), dtype = np.int64)
        self.y_ext = np.zeros((consts["len_y"], self.batch_size), dtype = np.int64)
        self.x_mask = np.zeros((consts["len_x"], self.batch_size, 1), dtype = np.int64)
        self.y_mask = np.zeros((consts["len_y"], self.batch_size, 1), dtype = np.int64)
        self.len_x = []
        self.len_y = []
        self.original_contents = []
        self.original_summarys = []
        self.x_ext_words = []
        self.max_ext_len = 0

        w2i = modules["w2i"]
        i2w = modules["i2w"]
        dict_size = len(w2i)

        for idx_doc in xrange(len(flist)):
            if len(flist[idx_doc]) == 2:
                contents, summarys = flist[idx_doc]
            else:
                print "ERROR!"
                return
            
            content, original_content = contents
            summary, original_summary = summarys
            self.original_contents.append(original_content)
            self.original_summarys.append(original_summary)
            xi_oovs = []
            for idx_word in xrange(len(content)):
                    # some sentences in duc is longer than len_x
                    if idx_word == consts["len_x"]:
                        break
                    w = content[idx_word]
                    
                    if w not in w2i: # OOV
                        if w not in xi_oovs:
                            xi_oovs.append(w)
                        self.x_ext[idx_word, idx_doc] = dict_size + xi_oovs.index(w) # 500005, 51000
                        w = i2w[modules["lfw_emb"]]
                    else:
                        self.x_ext[idx_word, idx_doc] = w2i[w]
                    
                    self.x[idx_word, idx_doc] = w2i[w]
                    self.x_mask[idx_word, idx_doc, 0] = 1
            self.len_x.append(np.sum(self.x_mask[:, idx_doc, :]))
            self.x_ext_words.append(xi_oovs)
            if self.max_ext_len < len(xi_oovs):
                self.max_ext_len = len(xi_oovs)

            if options["has_y"]:
                for idx_word in xrange(len(summary)):
                    w = summary[idx_word]
                    
                    if w not in w2i:
                        if w in xi_oovs:
                            self.y_ext[idx_word, idx_doc] = dict_size + xi_oovs.index(w)
                        else:
                            self.y_ext[idx_word, idx_doc] = w2i[i2w[modules["lfw_emb"]]] # unk
                        w = i2w[modules["lfw_emb"]] 
                    else:
                        self.y_ext[idx_word, idx_doc] =  w2i[w]
                    self.y[idx_word, idx_doc] = w2i[w]
                    if not options["is_predicting"]:
                        self.y_mask[idx_word, idx_doc, 0] = 1
                self.len_y.append(len(summary))
            else:
                self.y = self.y_mask = None

        max_len_x = int(np.max(self.len_x))
        max_len_y = int(np.max(self.len_y))
        
        self.x = self.x[0:max_len_x, :]
        self.x_ext = self.x_ext[0:max_len_x, :]
        self.x_mask = self.x_mask[0:max_len_x, :, :]
        self.y = self.y[0:max_len_y, :]
        self.y_ext = self.y_ext[0:max_len_y, :]
        self.y_mask = self.y_mask[0:max_len_y, :, :]

def get_data(xy_list, modules, consts, options):
    return BatchData(xy_list,  modules, consts, options)

def batched(x_size, options, consts):
    batch_size = consts["testing_batch_size"] if options["is_predicting"] else consts["batch_size"]
    if options["is_debugging"]:
        x_size = 13
    ids = [i for i in xrange(x_size)]
    if not options["is_predicting"]:
        shuffle(ids)
    batch_list = []
    batch_ids = []
    for i in xrange(x_size):
        idx = ids[i]
        batch_ids.append(idx)
        if len(batch_ids) == batch_size or i == (x_size - 1):
            batch_list.append(batch_ids)
            batch_ids = []
    return batch_list, len(ids), len(batch_list)

