CHECKPOINT='models/yolox_s.pth'
CHECKPOINT='./YOLOX_outputs/yolox_s_PS_prw_class_w2/latest_ckpt.pth'
FILE='./exps/example/custom/yolox_s_PS_update_prw'
IMAGE='../PSDT-main/data/PRW-v16.04.20/frames/c4s4_059866.jpg'
CUDA_VISIBLE_DEVICES=0 python tools/demo.py image -f $FILE -c $CHECKPOINT --path $IMAGE --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu