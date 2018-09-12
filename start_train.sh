export PYTHONPATH=`pwd`
CUDA_VISIBLE_DEVICES=4 python -u training_ptr_gen/train.py  | tee ./log/training_log 

