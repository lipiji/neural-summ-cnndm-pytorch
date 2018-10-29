## Abstractive Summarization on CNN-DailyMail

### Results
#### Model-1: attention-seq2seq

#### Model-2: attention-seq2seq + copy
##### GRU （NLL loss + norm-clip=5）:
```
C ROUGE-1 Average_R: 0.43722 (95%-conf.int. 0.43441 - 0.43994)
C ROUGE-1 Average_P: 0.33340 (95%-conf.int. 0.33090 - 0.33587)
C ROUGE-1 Average_F: 0.36604 (95%-conf.int. 0.36376 - 0.36829)

C ROUGE-2 Average_R: 0.18389 (95%-conf.int. 0.18144 - 0.18637)
C ROUGE-2 Average_P: 0.14111 (95%-conf.int. 0.13902 - 0.14328)
C ROUGE-2 Average_F: 0.15435 (95%-conf.int. 0.15225 - 0.15645)

C ROUGE-L Average_R: 0.39519 (95%-conf.int. 0.39245 - 0.39785)
C ROUGE-L Average_P: 0.30170 (95%-conf.int. 0.29935 - 0.30418)
C ROUGE-L Average_F: 0.33105 (95%-conf.int. 0.32886 - 0.33341)

C ROUGE-SU4 Average_R: 0.19460 (95%-conf.int. 0.19234 - 0.19681)
C ROUGE-SU4 Average_P: 0.14813 (95%-conf.int. 0.14624 - 0.15019)
C ROUGE-SU4 Average_F: 0.16220 (95%-conf.int. 0.16028 - 0.16405)
```
#### Model-3: attention-seq2seq + coverage

##### GRU （NLL loss + norm-clip=5）:
```
C ROUGE-1 Average_R: 0.38197 (95%-conf.int. 0.37955 - 0.38433)
C ROUGE-1 Average_P: 0.36479 (95%-conf.int. 0.36235 - 0.36742)
C ROUGE-1 Average_F: 0.36002 (95%-conf.int. 0.35802 - 0.36230)

C ROUGE-2 Average_R: 0.15487 (95%-conf.int. 0.15277 - 0.15708)
C ROUGE-2 Average_P: 0.14912 (95%-conf.int. 0.14701 - 0.15130)
C ROUGE-2 Average_F: 0.14638 (95%-conf.int. 0.14440 - 0.14846)

C ROUGE-L Average_R: 0.35101 (95%-conf.int. 0.34873 - 0.35333)
C ROUGE-L Average_P: 0.33577 (95%-conf.int. 0.33346 - 0.33824)
C ROUGE-L Average_F: 0.33113 (95%-conf.int. 0.32923 - 0.33335)

C ROUGE-SU4 Average_R: 0.16692 (95%-conf.int. 0.16490 - 0.16894)
C ROUGE-SU4 Average_P: 0.16021 (95%-conf.int. 0.15818 - 0.16222)
C ROUGE-SU4 Average_F: 0.15709 (95%-conf.int. 0.15528 - 0.15892)
```
#### Model-4: attention-seq2seq + copy + coverage

##### GRU （NLL loss + norm-clip=5）:
```
C ROUGE-1 Average_R: 0.44517 (95%-conf.int. 0.44225 - 0.44785)
C ROUGE-1 Average_P: 0.37019 (95%-conf.int. 0.36757 - 0.37286)
C ROUGE-1 Average_F: 0.39081 (95%-conf.int. 0.38862 - 0.39309)

C ROUGE-2 Average_R: 0.19478 (95%-conf.int. 0.19212 - 0.19740)
C ROUGE-2 Average_P: 0.16320 (95%-conf.int. 0.16092 - 0.16562)
C ROUGE-2 Average_F: 0.17147 (95%-conf.int. 0.16920 - 0.17378)

C ROUGE-L Average_R: 0.40901 (95%-conf.int. 0.40618 - 0.41163)
C ROUGE-L Average_P: 0.34055 (95%-conf.int. 0.33796 - 0.34318)
C ROUGE-L Average_F: 0.35930 (95%-conf.int. 0.35708 - 0.36152)

C ROUGE-SU4 Average_R: 0.20330 (95%-conf.int. 0.20083 - 0.20567)
C ROUGE-SU4 Average_P: 0.16916 (95%-conf.int. 0.16704 - 0.17141)
C ROUGE-SU4 Average_F: 0.17787 (95%-conf.int. 0.17579 - 0.18000)
```
##### GRU （avg NLL loss + norm-clip=2）：
```
C ROUGE-1 Average_R: 0.46082 (95%-conf.int. 0.45804 - 0.46365)
C ROUGE-1 Average_P: 0.37176 (95%-conf.int. 0.36919 - 0.37447)
C ROUGE-1 Average_F: 0.39686 (95%-conf.int. 0.39461 - 0.39909)

C ROUGE-2 Average_R: 0.20237 (95%-conf.int. 0.19977 - 0.20520)
C ROUGE-2 Average_P: 0.16415 (95%-conf.int. 0.16175 - 0.16654)
C ROUGE-2 Average_F: 0.17448 (95%-conf.int. 0.17225 - 0.17683)

C ROUGE-L Average_R: 0.42083 (95%-conf.int. 0.41817 - 0.42347)
C ROUGE-L Average_P: 0.33970 (95%-conf.int. 0.33722 - 0.34230)
C ROUGE-L Average_F: 0.36250 (95%-conf.int. 0.36024 - 0.36468)

C ROUGE-SU4 Average_R: 0.21030 (95%-conf.int. 0.20801 - 0.21277)
C ROUGE-SU4 Average_P: 0.16956 (95%-conf.int. 0.16745 - 0.17182)
C ROUGE-SU4 Average_F: 0.18028 (95%-conf.int. 0.17835 - 0.18236)
```
##### LSTM-v1:
```
C ROUGE-1 Average_R: 0.39289 (95%-conf.int. 0.39037 - 0.39537)
C ROUGE-1 Average_P: 0.40210 (95%-conf.int. 0.39939 - 0.40500)
C ROUGE-1 Average_F: 0.38322 (95%-conf.int. 0.38101 - 0.38563)

C ROUGE-2 Average_R: 0.17302 (95%-conf.int. 0.17077 - 0.17537)
C ROUGE-2 Average_P: 0.17902 (95%-conf.int. 0.17642 - 0.18162)
C ROUGE-2 Average_F: 0.16941 (95%-conf.int. 0.16720 - 0.17173)

C ROUGE-L Average_R: 0.36119 (95%-conf.int. 0.35873 - 0.36362)
C ROUGE-L Average_P: 0.37002 (95%-conf.int. 0.36737 - 0.37298)
C ROUGE-L Average_F: 0.35247 (95%-conf.int. 0.35025 - 0.35482)

C ROUGE-SU4 Average_R: 0.17830 (95%-conf.int. 0.17624 - 0.18045)
C ROUGE-SU4 Average_P: 0.18458 (95%-conf.int. 0.18225 - 0.18696)
C ROUGE-SU4 Average_F: 0.17415 (95%-conf.int. 0.17214 - 0.17623)
```
##### LSTM-v2:
```
---------------------------------------------
C ROUGE-1 Average_R: 0.43865 (95%-conf.int. 0.43618 - 0.44132)
C ROUGE-1 Average_P: 0.38804 (95%-conf.int. 0.38547 - 0.39081)
C ROUGE-1 Average_F: 0.39701 (95%-conf.int. 0.39489 - 0.39922)

C ROUGE-2 Average_R: 0.19277 (95%-conf.int. 0.19049 - 0.19538)
C ROUGE-2 Average_P: 0.17168 (95%-conf.int. 0.16935 - 0.17413)
C ROUGE-2 Average_F: 0.17480 (95%-conf.int. 0.17261 - 0.17711)

C ROUGE-L Average_R: 0.40145 (95%-conf.int. 0.39890 - 0.40411)
C ROUGE-L Average_P: 0.35558 (95%-conf.int. 0.35308 - 0.35830)
C ROUGE-L Average_F: 0.36359 (95%-conf.int. 0.36142 - 0.36591)

C ROUGE-SU4 Average_R: 0.19977 (95%-conf.int. 0.19763 - 0.20212)
C ROUGE-SU4 Average_P: 0.17737 (95%-conf.int. 0.17521 - 0.17965)
C ROUGE-SU4 Average_F: 0.18040 (95%-conf.int. 0.17845 - 0.18253)
```

### How to run:
- Python 2.7, Pytorch 0.4
- Download FINISHED_FILES from: https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail , and put it under ./data/
- Run python prepare_data.py
- Training: python main.py | tee train.log
- Tuning: modify main.py: is_predicting=true and model_selection=true, then run "bash tuning_deepmind.sh | tee tune.log"
- Testing: modify main.py: is_predicting=true and model_selection=false, then run "python main.py you-best-model (say cnndm.s2s.gpu4.epoch7.1)", go to "./deepmind/result/" and run  $ROUGE$ myROUGE_Config.xml C, you will get the results.
- The Perl Rouge package is enough, I did not use pyrouge.
