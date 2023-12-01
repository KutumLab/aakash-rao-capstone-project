WORK_DIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/cascade_rcnn
CONFIG_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/cascade_rcnn/configs

python3 mmcv/tools/train.py \
    $CONFIG_PATH/cascade_rcnn/cascade_rcnn_r50_fpn_1x.py \
    --work-dir $WORK_DIR/cascade_rcnn_r50_fpn_1x