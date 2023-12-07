
# SEMI-CONSTANT INFO
DATA_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/detectron_single
SAVE_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs
MAX_ITERS=10


# # VARIABLE INFO [DONE]
CONFIG=./configs/faster_rcnn_R_50_DC5_3x.yaml
NAME=faster_rcnn_R_50_DC5_3x

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 1 \
    --save_path $SAVE_PATH \

# python3 train_faster_rcnn.py \
#     --data_path $DATA_PATH \
#     --config_info $CONFIG \
#     --max_iters $MAX_ITERS \
#     --name $NAME \
#     --fold 2 \
#     --save_path $SAVE_PATH \

# python3 train_faster_rcnn.py \
#     --data_path $DATA_PATH \
#     --config_info $CONFIG \
#     --max_iters $MAX_ITERS \
#     --name $NAME \
#     --fold 3 \
#     --save_path $SAVE_PATH \

