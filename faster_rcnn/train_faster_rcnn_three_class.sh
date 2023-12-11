
# SEMI-CONSTANT INFO
DATA_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/detectron_three_class
SAVE_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs
MAX_ITERS=15 # 15000
VERSION=three_class

# VARIABLE INFO faster_rcnn_R_50_DC5_3x [DONE]
CONFIG=COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml
NAME=faster_rcnn_R_50_DC5_3x_three_class

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 1 \
    --save_path $SAVE_PATH \
    --version $VERSION

# python3 train_faster_rcnn.py \
#     --data_path $DATA_PATH \
#     --config_info $CONFIG \
#     --max_iters $MAX_ITERS \
#     --name $NAME \
#     --fold 2 \
#     --save_path $SAVE_PATH \
#     --version $VERSION

# python3 train_faster_rcnn.py \
#     --data_path $DATA_PATH \
#     --config_info $CONFIG \
#     --max_iters $MAX_ITERS \
#     --name $NAME \
#     --fold 3 \
#     --save_path $SAVE_PATH \
#     --version $VERSION