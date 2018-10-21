#!/bin/bash
FILES=./deepmind/model/*
for f in $FILES; do
    echo "==========================" ${f##*/}
    python -u main.py ${f##*/}
    python prepare_rouge.py
    cd ./deepmind/result/ 
    perl /home/pijili/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl -n 4 -w 1.2 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0  myROUGE_Config.xml C
    cd ../../
done
