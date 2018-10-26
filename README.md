## Abstractive Summarization on CNN-DailyMail

### Results
#### Model-1: attention-seq2seq

#### Model-2: attention-seq2seq + copy
##### GRU （NLL loss + norm-clip=5）:
```
C ROUGE-1 Average_R: 0.43722 (95%-conf.int. 0.43441 - 0.43994)</br>
C ROUGE-1 Average_P: 0.33340 (95%-conf.int. 0.33090 - 0.33587)</br>
C ROUGE-1 Average_F: 0.36604 (95%-conf.int. 0.36376 - 0.36829)</br>

C ROUGE-2 Average_R: 0.18389 (95%-conf.int. 0.18144 - 0.18637)</br>
C ROUGE-2 Average_P: 0.14111 (95%-conf.int. 0.13902 - 0.14328)</br>
C ROUGE-2 Average_F: 0.15435 (95%-conf.int. 0.15225 - 0.15645)</br>

C ROUGE-L Average_R: 0.39519 (95%-conf.int. 0.39245 - 0.39785)</br>
C ROUGE-L Average_P: 0.30170 (95%-conf.int. 0.29935 - 0.30418)</br>
C ROUGE-L Average_F: 0.33105 (95%-conf.int. 0.32886 - 0.33341)</br>

C ROUGE-SU4 Average_R: 0.19460 (95%-conf.int. 0.19234 - 0.19681)</br>
C ROUGE-SU4 Average_P: 0.14813 (95%-conf.int. 0.14624 - 0.15019)</br>
C ROUGE-SU4 Average_F: 0.16220 (95%-conf.int. 0.16028 - 0.16405)</br>
```
#### Model-3: attention-seq2seq + coverage

##### GRU （NLL loss + norm-clip=5）:
```
C ROUGE-1 Average_R: 0.38197 (95%-conf.int. 0.37955 - 0.38433)</br>
C ROUGE-1 Average_P: 0.36479 (95%-conf.int. 0.36235 - 0.36742)</br>
C ROUGE-1 Average_F: 0.36002 (95%-conf.int. 0.35802 - 0.36230)</br>

C ROUGE-2 Average_R: 0.15487 (95%-conf.int. 0.15277 - 0.15708)</br>
C ROUGE-2 Average_P: 0.14912 (95%-conf.int. 0.14701 - 0.15130)</br>
C ROUGE-2 Average_F: 0.14638 (95%-conf.int. 0.14440 - 0.14846)</br>

C ROUGE-L Average_R: 0.35101 (95%-conf.int. 0.34873 - 0.35333)</br>
C ROUGE-L Average_P: 0.33577 (95%-conf.int. 0.33346 - 0.33824)</br>
C ROUGE-L Average_F: 0.33113 (95%-conf.int. 0.32923 - 0.33335)</br>

C ROUGE-SU4 Average_R: 0.16692 (95%-conf.int. 0.16490 - 0.16894)</br>
C ROUGE-SU4 Average_P: 0.16021 (95%-conf.int. 0.15818 - 0.16222)</br>
C ROUGE-SU4 Average_F: 0.15709 (95%-conf.int. 0.15528 - 0.15892)</br>
```
#### Model-4: attention-seq2seq + copy + coverage

##### GRU （NLL loss + norm-clip=5）:
```
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
```
##### GRU （avg NLL loss + norm-clip=2）：
```
C ROUGE-1 Average_R: 0.46080 (95%-conf.int. 0.45828 - 0.46352)</br>
C ROUGE-1 Average_P: 0.37468 (95%-conf.int. 0.37218 - 0.37748)</br>
C ROUGE-1 Average_F: 0.39739 (95%-conf.int. 0.39519 - 0.39968)</br>

C ROUGE-2 Average_R: 0.20124 (95%-conf.int. 0.19877 - 0.20371)</br>
C ROUGE-2 Average_P: 0.16484 (95%-conf.int. 0.16236 - 0.16718)</br>
C ROUGE-2 Average_F: 0.17391 (95%-conf.int. 0.17158 - 0.17612)</br>

C ROUGE-L Average_R: 0.41827 (95%-conf.int. 0.41577 - 0.42091)</br>
C ROUGE-L Average_P: 0.34040 (95%-conf.int. 0.33784 - 0.34313)</br>
C ROUGE-L Average_F: 0.36089 (95%-conf.int. 0.35869 - 0.36331)</br>

C ROUGE-SU4 Average_R: 0.20868 (95%-conf.int. 0.20651 - 0.21080)</br>
C ROUGE-SU4 Average_P: 0.16992 (95%-conf.int. 0.16774 - 0.17214)</br>
C ROUGE-SU4 Average_F: 0.17928 (95%-conf.int. 0.17719 - 0.18128)</br>
```

### How to run:
- Download FINISHED_FILES from: https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail , and put it under ./data/
- Run python prepare_data.py
- Training: python main.py | tee train.log
- Tuning: modify main.py: is_predicting=true and model_selection=true, then run "bash tuning_deepmind.sh | tee tune.log"
- Testing: modify main.py: is_predicting=true and model_selection=false, then run "python main.py you-best-model (say cnndm.s2s.gpu4.epoch7.1)", go to "./deepmind/result/" and run  $ROUGE$ myROUGE_Config.xml C, you will get the results.
- The Perl Rouge package is enough, I did not use pyrouge.
