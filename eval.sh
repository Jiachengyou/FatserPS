CHECKPOINT='models/yolox_s.pth'
# CHECKPOINT='YOLOX_outputs/yolox_s_PS_reid_woMosaic/best_ckpt.pth'
CHECKPOINT='YOLOX_outputs/yolox_s_PS_reid_woMosaic_maxepoch100/epoch_15_ckpt.pth'
CHECKPOINT='YOLOX_outputs/yolox_s_PS_reid_woMosaic_total/latest_ckpt.pth'
CHECKPOINT='YOLOX_outputs/yolox_s_PS_reid_woMosaic_maxepoch300_woreidhead/epoch_105_ckpt.pth'
CHECKPOINT='YOLOX_outputs/yolox_s_PS__reid_embedding256/latest_ckpt.pth'
CHECKPOINT='YOLOX_outputs/yolox_s_PS_test/last_mosaic_epoch_ckpt.pth'
CHECKPOINT='YOLOX_outputs/yolox_l_PS_reid256_wMosaic_maxepoch300_tt/latest_ckpt.pth'
EXP_NAME='yolox_s_PS_reid_woMosaic_eval'
# EXP_NAME='yolox_s_PS__reid_embedding256_eval'
EXP_NAME='yolox_s_PS_reid_woMosaic_maxepoch300_woreidhead_eval'
EXP_NAME='yolox_s_PS_test_eval'
# EXP_NAME='yolox_s_PS_reid_woMosaic_maxepoch100_eval'
FILE='./exps/example/custom/yolox_s_PS_update'
FILE='./exps/example/custom/yolox_s_PS_update_simple'
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -f $FILE -d 1 -b 16 -expn $EXP_NAME --conf 0.001 -c $CHECKPOINT --fuse
# -o 
# -c $CHECKPOINT
# python test_results.py
# --fp16