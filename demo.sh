CHECKPOINT='models/yolox_s.pth'
IMAGE='../TransPS-main/data/CUHK-SYSU/Image/SSM/s15174.jpg'
CUDA_VISIBLE_DEVICES=0 python tools/demo.py image -n yolox-s -c $CHECKPOINT --path $IMAGE --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu