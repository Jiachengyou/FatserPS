EXP_NAME='yolox_s_PS_reid_woMosaic_maxepoch300_woreidhead'
FILE='./exps/example/custom/yolox_s_PS_update'
CUDA_VISIBLE_DEVICES=3 python tools/train.py -f $FILE -d 1 -b 16 -expn $EXP_NAME
    # -o n
# -c $CHECKPOINT