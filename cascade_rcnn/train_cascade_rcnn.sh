WORK_DIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/cascade_rcnn
CONFIG_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/cascade_rcnn/configs

python3 ./mmdetection/tools/train.py \
    $CONFIG_PATH/cascade_rcnn/cascade-rcnn_x101_64x4d_fpn_20e_coco_fold_1.py \
    --work-dir $WORK_DIR/cascade-rcnn_x101_64x4d_fpn_20e_coco_fold_1