EXP_NAME='yolox_s_PS_reid_woMosaic_maxepoch300_woreidhead'
EXP_NAME='yolox_s_PS_reid_woMosaic_maxepoch120'
EXP_NAME='yolox_s_PS_reid_wMosaic_maxepoch300'
EXP_NAME='yolox_l_PS_reid256_wMosaic_maxepoch300_tt'
EXP_NAME='yolox_s_PS_reid128_wMosaic_maxepoch300_simple_0.5'
EXP_NAME='yolox_darket_PS_reid128_wMosaic_maxepoch300'
EXP_NAME='yolox_darket_PS_reid256_wMosaic_maxepoch300_woreidhead'
EXP_NAME='yolox_darket_PS_reid256_woMosaic_maxepoch300_wofp16_update'
EXP_NAME='yolox_l_PS_reid256_baseline'
EXP_NAME='yolox_m_PS_reid128_baseline'
EXP_NAME='yolox_s_PS_reid128_baseline_fpn_update_v3_woreid_v3'
EXP_NAME='yolox_s_PS_reid128_baseline_woreid_nae_v0'
EXP_NAME='yolox_s_PS_reid128_baseline_kd_256_hierarchy_decouple_seed2'
EXP_NAME='yolox_s_PS_reid128_baseline_kd_256_hierarchy_fpn256/8'
EXP_NAME='yolox_s_PS_reid128_baseline_kd_256_hierarchy_transformer_v2_2048'
EXP_NAME='yolox_s_PS_reid128_baseline_kd_256_hierarchy_fpn512_16_v2'
# EXP_NAME='yolox_s_PS_test'
FILE='./exps/example/custom/yolox_s_PS'
FILE='./exps/example/custom/yolox_s_PS_update'
# FILE='./exps/example/custom/yolox_s_PS_update_simple'
# FILE='./exps/example/custom/yolox_darket_PS_update'
# FILE='./exps/example/custom/yolox_s_PS_update'
# FILE='./exps/example/custom/yolox_l_PS_update'
# FILE='./exps/example/custom/yolox_l_PS_update'
# FILE='./exps/example/custom/yolox_s_PS_update_fpn'
# FILE='./exps/example/custom/yolox_l_PS_update'
#CHECKPOINT
CHECKPOINT='yolox_s_PS_reid_woMosaic_maxepoch300'
CHECKPOINT='YOLOX_outputs/yolox_s_PS_reid_woMosaic_maxepoch300/latest_ckpt.pth'
CHECKPOINT='YOLOX_outputs/yolox_darket_PS_reid128_wMosaic_maxepoch300/latest_ckpt.pth'
# CHECKPOINT='YOLOX_outputs/yolox_s_PS_reid128_wMosaic_maxepoch300_simple_detection/latest_ckpt.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py -f $FILE -d 4 -b 64 -expn $EXP_NAME --cache -o
# -o
# --fp16
# --resume -c $CHECKPOINT
    # -o n
# -c $CHECKPOINT