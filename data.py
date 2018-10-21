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
        is_lvt = options["has_lvt_trick"]
        self.batch_size = len(flist) 
        self.x = np.zeros((consts["len_x"], self.batch_size), dtype = np.int64)
        self.y = np.zeros((consts["len_y"], self.batch_size), dtype = np.int64)
        self.x_lvt = np.zeros((consts["len_x"], self.batch_size), dtype = np.int64) if is_lvt else None
        self.y_lvt = np.zeros((consts["len_y"], self.batch_size), dtype = np.int64) if is_lvt else None
        self.x_mask = np.zeros((consts["len_x"], self.batch_size, 1), dtype = np.int64)
        self.y_mask = np.zeros((consts["len_y"], self.batch_size, 1), dtype = np.int64)
        self.len_x = []
        self.len_y = []
        self.original_contents = []
        self.original_summarys = []

        w2i = modules["w2i"]
        i2w = modules["i2w"]
        if is_lvt:
            hfw = modules["freq_words"]
            lvt_dict_size = consts["lvt_dict_size"]
            lvt_w2i = {}
            lvt_i2i = {}
            lvt_dict = []

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

            for idx_word in xrange(len(content)):
                    # some sentences in duc is longer than len_x
                    if idx_word == consts["len_x"]:
                        break
                    w = content[idx_word]
                    
                    # remove eos for duc dataset
                    if options["omit_eos"] and w == i2w[modules["eos_emb"]]:
                        break
                    # words in duc dataset may not in dict
                    if w not in w2i: # duc
                        w = i2w[modules["lfw_emb"]]

                    self.x[idx_word, idx_doc] = w2i[w]
                    if is_lvt:
                        if w not in lvt_w2i:
                            lvt_w2i[w] = len(lvt_dict)
                            lvt_dict.append(w2i[w])
                        self.x_lvt[idx_word, idx_doc] = lvt_w2i[w]
                    self.x_mask[idx_word, idx_doc, 0] = 1
            self.len_x.append(np.sum(self.x_mask[:, idx_doc, :]))

            if options["has_y"]:
                for idx_word in xrange(len(summary)):
                    w = summary[idx_word]
                    
                    if w not in w2i:
                        w = i2w[modules["lfw_emb"]] 

                    self.y[idx_word, idx_doc] = w2i[w]
                    if is_lvt:
                        if not options["is_predicting"]:
                            if w not in lvt_w2i:
                                lvt_w2i[w] = len(lvt_dict)
                                lvt_dict.append(w2i[w])
                            self.y_lvt[idx_word, idx_doc] = lvt_w2i[w]
                    if not options["is_predicting"]:
                        self.y_mask[idx_word, idx_doc, 0] = 1
                self.len_y.append(len(summary))
            else:
                self.y = self.y_mask = None

        max_len_x = int(np.max(self.len_x))
        max_len_y = int(np.max(self.len_y))
        
        self.x = self.x[0:max_len_x, :]
        self.x_mask = self.x_mask[0:max_len_x, :, :]
        self.y = self.y[0:max_len_y, :]
        self.y_mask = self.y_mask[0:max_len_y, :, :]


        if not is_lvt:
            return
       
        # use lvt for duc dataset to remove eos!!!
        if options["omit_eos"]:
            if len(lvt_dict) > lvt_dict_size:
                print "len(lvt_dict) > lvt_dict_size", len(lvt_dict), lvt_dict_size
            if len(lvt_dict) < lvt_dict_size:
                for w in hfw:
                    if w not in lvt_w2i and w != i2w[modules["eos_emb"]]:
                        lvt_w2i[w] = len(lvt_dict)
                        lvt_dict.append(w2i[w])
                    if len(lvt_dict) == lvt_dict_size:
                        break

            assert len(lvt_w2i) == lvt_dict_size
            assert len(lvt_dict) == lvt_dict_size
            for i in xrange(lvt_dict_size):
                lvt_i2i[i] = lvt_dict[i]

        else:
            # process lvt dict
            if len(lvt_dict) > lvt_dict_size:
                print "len(lvt_dict) > lvt_dict_size", len(lvt_dict), lvt_dict_size
            if len(lvt_dict) < lvt_dict_size:
                for w in hfw:
                    if w not in lvt_w2i:
                        lvt_w2i[w] = len(lvt_dict)
                        lvt_dict.append(w2i[w])
                    if len(lvt_dict) == lvt_dict_size:
                        break

            assert len(lvt_w2i) == lvt_dict_size
            assert len(lvt_dict) == lvt_dict_size
            for i in xrange(lvt_dict_size):
                lvt_i2i[i] = lvt_dict[i]

        self.lvt_dict = lvt_dict
        self.lvt_w2i = lvt_w2i
        self.lvt_i2i = lvt_i2i

        self.x_lvt = self.x_lvt[0:max_len_x, :]
        self.y_lvt = self.y_lvt[0:max_len_y, :]

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

