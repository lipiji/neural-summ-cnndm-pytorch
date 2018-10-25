## Abstractive Summarization on CNN-DailyMail

### Results
#### Model-1: attention-seq2seq

#### Model-2: attention-seq2seq + copy

#### Model-3: attention-seq2seq + coverage

#### Model-4: attention-seq2seq + copy + coverage
##### GRU:
C ROUGE-1 Average_R: 0.44517 (95%-conf.int. 0.44225 - 0.44785)</br>
C ROUGE-1 Average_P: 0.37019 (95%-conf.int. 0.36757 - 0.37286)</br>
C ROUGE-1 Average_F: 0.39081 (95%-conf.int. 0.38862 - 0.39309)</br>

C ROUGE-2 Average_R: 0.19478 (95%-conf.int. 0.19212 - 0.19740)</br>
C ROUGE-2 Average_P: 0.16320 (95%-conf.int. 0.16092 - 0.16562)</br>
C ROUGE-2 Average_F: 0.17147 (95%-conf.int. 0.16920 - 0.17378)</br>

C ROUGE-L Average_R: 0.40901 (95%-conf.int. 0.40618 - 0.41163)</br>
C ROUGE-L Average_P: 0.34055 (95%-conf.int. 0.33796 - 0.34318)</br>
C ROUGE-L Average_F: 0.35930 (95%-conf.int. 0.35708 - 0.36152)</br>

C ROUGE-SU4 Average_R: 0.20330 (95%-conf.int. 0.20083 - 0.20567)</br>
C ROUGE-SU4 Average_P: 0.16916 (95%-conf.int. 0.16704 - 0.17141)</br>
C ROUGE-SU4 Average_F: 0.17787 (95%-conf.int. 0.17579 - 0.18000)</br>


### How to run:
- Download FINISHED_FILES from: https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail , and put it under ./data/
- Run python prepare_data.py
- Training: python main.py | tee train.log
- Tuning: modify main.py: is_predicting=true and model_selection=true, then run "bash tuning_deepmind.sh | tee tune.log"
- Testing: modify main.py: is_predicting=true and model_selection=false, then run "python main.py you-best-model (say cnndm.s2s.gpu4.epoch7.1)", go to "./deepmind/result/" and run  $ROUGE$ myROUGE_Config.xml C, you will get the results.
- The Perl Rouge package is enough, I did not use pyrouge.
