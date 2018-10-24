#pylint: skip-file
import sys
import os
from configs import * 

cfg = DeepmindConfigs()

# config file for ROUGE
ROUGE_PATH = cfg.cc.RESULT_PATH 
SUMM_PATH = cfg.cc.SUMM_PATH
MODEL_PATH = cfg.cc.GROUND_TRUTH_PATH
i2summ = {}
summ2i = {}
i2model = {}

# for result
flist = os.listdir(SUMM_PATH)
i = 0
for fname in flist:
    i2summ[str(i)] = fname
    summ2i[fname] = str(i)
    i += 1

# for models
flist = os.listdir(MODEL_PATH)
i2model = {}
for fname in flist:
    if fname not in summ2i:
        raise IOError

    i = summ2i[fname]
    i2model[i] = fname

assert len(i2model) == len(i2summ)

# write to config file
rouge_s = "<ROUGE-EVAL version=\"1.0\">"
file_id = 0
for file_id, fsumm in i2summ.items():
    rouge_s +=  "\n<EVAL ID=\"" + file_id + "\">" \
            + "\n<PEER-ROOT>" \
            + SUMM_PATH \
            + "\n</PEER-ROOT>" \
            + "\n<MODEL-ROOT>" \
            + "\n" + MODEL_PATH \
            + "\n</MODEL-ROOT>" \
            + "\n<INPUT-FORMAT TYPE=\"SPL\">" \
            + "\n</INPUT-FORMAT>" \
            + "\n<PEERS>" \
            + "\n<P ID=\"C\">" + fsumm + "</P>" \
            + "\n</PEERS>" \
            + "\n<MODELS>"

    rouge_s += "\n<M ID=\"" + file_id + "\">" + i2model[file_id] + "</M>"
    rouge_s += "\n</MODELS>\n</EVAL>"
                    
rouge_s += "\n</ROUGE-EVAL>"

with open(ROUGE_PATH + "myROUGE_Config.xml", "w") as f_rouge:
    f_rouge.write(rouge_s) 
