# -*- coding: utf-8 -*-
import operator
from os import makedirs
from os.path import exists
import argparse
from configs import *
import cPickle as pickle
import numpy as np
import re
from random import shuffle
import string
import struct

from tensorflow.core.example import example_pb2


def run(d_type, d_path):
    prepare_deepmind(d_path)

stop_words = {"-lrb-", "-rrb-", "-"}
unk_words = {"unk", "<unk>"}

def get_xy_tuple(cont, head, cfg):
    x = read_cont(cont, cfg)
    y = read_head(head, cfg)

    if x != None and y != None:
        return (x, y)
    else:
        return None

def load_lines(d_path, f_name,  configs):
    lines = []
    f_path = d_path + f_name
    reader = open(f_path, 'rb')
    while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        e = example_pb2.Example.FromString(example_str)
        try:
            article_text = e.features.feature['article'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
            abstract_text = e.features.feature['abstract'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
        except ValueError:
            print "ValueError"
            continue
        if len(article_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
            print 'Found an example with empty article text. Skipping it.'
        else:
            xy_tuple = get_xy_tuple(article_text, abstract_text, configs)
            if xy_tuple != None:
                lines.append(xy_tuple)
    reader.close()
    return lines

def load_dict(d_path, f_name, dic, dic_list):
    f_path = d_path + f_name
    f = open(f_path, "r")
    for line in f:
        line = line.strip('\n').strip('\r').lower()
        if line:
            tf = line.split()
            dic[tf[0]] = int(tf[1])
            dic_list.append(tf[0])
    return dic, dic_list

def to_dict(xys, dic):
    # dict should not consider test set!!!!!
    for xy in xys:
        sents, summs = xy
        y = summs[0]
        for w in y:
            if w in dic:
                dic[w] += 1
            else:
                dic[w] = 1
                
        x = sents[0]
        for w in x:
            if w in dic:
                dic[w] += 1
            else:
                dic[w] = 1
    return dic


def del_num(s):
    return re.sub(r"(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b","#", s)

def read_cont(f_cont, cfg):
    lines = []
    line = f_cont #del_num(f_cont)
    words = line.split()
    num_words = len(words)
    if num_words >= cfg.MIN_LEN_X and num_words < cfg.MAX_LEN_X:
        lines += words
    elif num_words >= cfg.MAX_LEN_X:
        lines += words[0:cfg.MAX_LEN_X]
    lines += [cfg.W_EOS]
    return (lines, f_cont) if len(lines) >= cfg.MIN_LEN_X and len(lines) <= cfg.MAX_LEN_X+1 else None

def abstract2sents(abstract, cfg):
  cur = 0
  sents = []
  while True:
    try:
      start_p = abstract.index(cfg.W_LS, cur)
      end_p = abstract.index(cfg.W_RS, start_p + 1)
      cur = end_p + len(cfg.W_RS)
      sents.append(abstract[start_p+len(cfg.W_LS):end_p])
    except ValueError as e: # no more sentences
      return sents

def read_head(f_head, cfg):
    lines = []

    sents = abstract2sents(f_head, cfg)
    line = ' '.join(sents)
    words = line.split()
    num_words = len(words)
    if num_words >= cfg.MIN_LEN_Y and num_words <= cfg.MAX_LEN_Y:
        lines += words
        lines += [cfg.W_EOS]
    elif num_words > cfg.MAX_LEN_Y: # do not know if should be stoped
        lines = words[0 : cfg.MAX_LEN_Y + 1] # one more word.
    
    return (lines, sents) if len(lines) >= cfg.MIN_LEN_Y and len(lines) <= cfg.MAX_LEN_Y+1  else None

def prepare_deepmind(d_path):
    configs = DeepmindConfigs()
    TRAINING_PATH = configs.cc.TRAINING_DATA_PATH
    VALIDATE_PATH = configs.cc.VALIDATE_DATA_PATH
    TESTING_PATH = configs.cc.TESTING_DATA_PATH
    RESULT_PATH = configs.cc.RESULT_PATH
    MODEL_PATH = configs.cc.MODEL_PATH
    BEAM_SUMM_PATH = configs.cc.BEAM_SUMM_PATH
    BEAM_GT_PATH = configs.cc.BEAM_GT_PATH
    GROUND_TRUTH_PATH = configs.cc.GROUND_TRUTH_PATH
    SUMM_PATH = configs.cc.SUMM_PATH
    TMP_PATH = configs.cc.TMP_PATH

    print "train: " + TRAINING_PATH
    print "test: " + TESTING_PATH
    print "validate: " + VALIDATE_PATH 
    print "result: " + RESULT_PATH
    print "model: " + MODEL_PATH
    print "tmp: " + TMP_PATH

    if not exists(TRAINING_PATH):
        makedirs(TRAINING_PATH)
    if not exists(VALIDATE_PATH):
        makedirs(VALIDATE_PATH)
    if not exists(TESTING_PATH):
        makedirs(TESTING_PATH)
    if not exists(RESULT_PATH):
        makedirs(RESULT_PATH)
    if not exists(MODEL_PATH):
        makedirs(MODEL_PATH)
    if not exists(BEAM_SUMM_PATH):
        makedirs(BEAM_SUMM_PATH)
    if not exists(BEAM_GT_PATH):
        makedirs(BEAM_GT_PATH)
    if not exists(GROUND_TRUTH_PATH):
        makedirs(GROUND_TRUTH_PATH)
    if not exists(SUMM_PATH):
        makedirs(SUMM_PATH)
    if not exists(TMP_PATH):
        makedirs(TMP_PATH)
    
        
    print "trainset..."
    train_xy_list = load_lines(d_path, "train.bin", configs)
    
    print "dump train..."
    pickle.dump(train_xy_list, open(TRAINING_PATH + "train.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    

    print "fitering and building dict..."    
    use_abisee = True
    all_dic1 = {}
    all_dic2 = {}
    dic_list = []
    all_dic1, dic_list = load_dict(d_path, "vocab", all_dic1, dic_list)
    all_dic2 = to_dict(train_xy_list, all_dic2)
    for w, tf in all_dic2.items():
        if w not in all_dic1:
            all_dic1[w] = tf

    candiate_list = dic_list[0:configs.PG_DICT_SIZE] # 50000
    candiate_set = set(candiate_list)

    dic = {}
    w2i = {}
    i2w = {}
    w2w = {}

    for w in [configs.W_PAD, configs.W_UNK, configs.W_EOS]:
    #for w in [configs.W_PAD, configs.W_UNK, configs.W_BOS, configs.W_EOS, configs.W_LS, configs.W_RS]:
        w2i[w] = len(dic)
        i2w[w2i[w]] = w
        dic[w] = 10000
        w2w[w] = w

    for w, tf in all_dic1.items():
        if w in candiate_set:
            w2i[w] = len(dic)
            i2w[w2i[w]] = w
            dic[w] = tf
            w2w[w] = w
        else:
            w2w[w] = configs.W_UNK 
    hfw = []
    sorted_x = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
    for w in sorted_x:
        hfw.append(w[0])

    assert len(hfw) == len(dic)
    assert len(w2i) == len(dic)
    print "dump dict..."
    pickle.dump([all_dic1, dic, hfw, w2i, i2w, w2w], open(TRAINING_PATH + "dic.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    
    print "testset..."
    test_xy_list = load_lines(d_path, "test.bin", configs)

    print "validset..."
    valid_xy_list = load_lines(d_path, "val.bin", configs)


    print "#train = ", len(train_xy_list)
    print "#test = ", len(test_xy_list)
    print "#validate = ", len(valid_xy_list)
        
    print "#all_dic = ", len(all_dic1), ", #dic = ", len(dic), ", #hfw = ", len(hfw)

    print "dump test..."
    pickle.dump(test_xy_list, open(TESTING_PATH + "test.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    shuffle(test_xy_list)
    pickle.dump(test_xy_list[0:2000], open(TESTING_PATH + "pj2000.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)

    print "dump validate..."
    pickle.dump(valid_xy_list, open(VALIDATE_PATH + "valid.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    pickle.dump(valid_xy_list[0:1000], open(VALIDATE_PATH + "pj1000.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    
    print "done."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="deepmind", help="dataset path", )
    args = parser.parse_args()

    data_type = "deepmind"
    # download from finished_files: https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail
    raw_path = "./data/CNN-Dailymail/finished_files/"

    print data_type, raw_path
    run(data_type, raw_path)
