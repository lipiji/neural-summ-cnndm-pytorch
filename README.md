## Abstractive Summarization on CNN-DailyMail

### Results
#### Model-1: attention-seq2seq

#### Model-2: attention-seq2seq + copy

#### Model-3: attention-seq2seq + coverage

#### Model-4: attention-seq2seq + copy + coverage

### How to run:
- Download FINISHED_FILES from: https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail , and put it under ./data/
- Run python prepare_data.py
- Training: python main.py | tee train.log
- Tuning: modify main.py: is_predicting=true and model_selection=true, then run "bash tuning_deepmind.sh | tee tune.log"
- Testing: modify main.py: is_predicting=true and model_selection=false, then run "python main.py you-best-model (say cnndm.s2s.gpu4.epoch7.1)", go to "./deepmind/result/" and run  $ROUGE$ myROUGE_Config.xml C, you will get the results.
- The Perl Rouge package is enough, I did not use pyrouge.
