# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================



"""Python implementation of BLEU and smooth-BLEU.

copy from: https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math
import os
import argparse

def load_lines(f_path):
    lines = []
    with open(f_path, "r") as f:
        for line in f:
            line = line.strip('\n').strip('\r')
            fs = line.split()
            lines.append(fs)
    return lines
 
def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)



def bleu(ref_path, pred_path, smooth=True, n = 1):
    id2f_ref = {}
    id2f_pred = {}
    
    flist = os.listdir(ref_path)
    for fname in flist:
        id_ = fname
        id2f_ref[id_] = ref_path + fname
    
    flist = os.listdir(pred_path)
    for fname in flist:
        id_ = fname
        id2f_pred[id_] = pred_path + fname

    assert len(id2f_ref) == len(id2f_pred)
    
    ref_lists = []
    pred_lists = []
    for fid, fpath in id2f_ref.items():
        ref_list = load_lines(fpath)
        assert len(ref_list) == n
        ref_lists.append(ref_list)

        pred_list = load_lines(id2f_pred[fid])
        assert len(pred_list) == n
        pred_lists.append(pred_list[0])


    return compute_bleu(ref_lists, pred_lists, smooth=smooth) 
 
bleu("./weibo/result/ground_truth/", "./weibo/result/summary/", smooth=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ref", help="reference path")
    parser.add_argument("-p", "--pred", help="prediction path")
    args = parser.parse_args()

    bleu, precisions, bp, ratio, translation_length, reference_length = bleu(args.ref, args.pred)
    print "BLEU = ",bleu
    print "BLEU1 = ",precisions[0]
    print "BLEU2 = ",precisions[1]
    print "BLEU3 = ",precisions[2]
    print "BLEU4 = ",precisions[3]
    print "ratio = ",ratio
