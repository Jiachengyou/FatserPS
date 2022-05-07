EXP_NAME='yolox_s_PS_reid_woMosaic_maxepoch300_woreidhead'
EXP_NAME='yolox_s_PS_reid_woMosaic_maxepoch120'
EXP_NAME='yolox_s_PS_reid_wMosaic_maxepoch300'
EXP_NAME='yolox_l_PS_reid256_wMosaic_maxepoch300_tt'
EXP_NAME='yolox_s_PS_reid128_wMosaic_maxepoch300_simple'
# FILE
FILE='./exps/example/custom/yolox_s_PS_update'
FILE='./exps/example/custom/yolox_s_PS_update_simple'
# FILE='./exps/example/custom/yolox_l_PS_update'
#CHECKPOINT
CHECKPOINT='yolox_s_PS_reid_woMosaic_maxepoch300'
CHECKPOINT='YOLOX_outputs/yolox_s_PS_reid_woMosaic_maxepoch300/latest_ckpt.pth'
CUDA_VISIBLE_DEVICES=1 python tools/train.py -f $FILE -d 1 -b 16 -expn $EXP_NAME --cache
# --resume -c $CHECKPOINT
    # -o n
# -c $CHECKPOINT