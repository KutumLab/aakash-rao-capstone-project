
# SEMI-CONSTANT INFO
DATA_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/detectron
SAVE_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs
MAX_ITERS=15


# VARIABLE INFO
CONFIG=COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml
NAME=faster_rcnn_R_50_DC5_1x

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 1 \
    --save_path $SAVE_PATH \

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 2 \
    --save_path $SAVE_PATH \

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 3 \
    --save_path $SAVE_PATH \


# VARIABLE INFO faster_rcnn_R_101_C4_3x
CONFIG=COCO-Detection/faster_rcnn_R_101_C4_3x.yaml
NAME=faster_rcnn_R_101_C4_3x

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 1 \
    --save_path $SAVE_PATH \

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 2 \
    --save_path $SAVE_PATH \

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 3 \
    --save_path $SAVE_PATH \


# VARIABLE INFO faster_rcnn_R_101_DC5_3x
CONFIG=COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml
NAME=faster_rcnn_R_101_DC5_3x

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 1 \
    --save_path $SAVE_PATH \

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 2 \
    --save_path $SAVE_PATH \

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 3 \
    --save_path $SAVE_PATH \



# VARIABLE INFO faster_rcnn_R_50_C4_1x
CONFIG=COCO-Detection/faster_rcnn_R_50_C4_1x.yaml
NAME=faster_rcnn_R_50_C4_1x

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 1 \
    --save_path $SAVE_PATH \

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 2 \
    --save_path $SAVE_PATH \

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 3 \
    --save_path $SAVE_PATH \




# VARIABLE INFO faster_rcnn_R_50_C4_3x
CONFIG=COCO-Detection/faster_rcnn_R_50_C4_3x.yaml
NAME=faster_rcnn_R_50_C4_3x

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 1 \
    --save_path $SAVE_PATH \

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 2 \
    --save_path $SAVE_PATH \

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 3 \
    --save_path $SAVE_PATH \



# VARIABLE INFO faster_rcnn_R_50_DC5_3x
CONFIG=COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml
NAME=faster_rcnn_R_50_DC5_3x

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 1 \
    --save_path $SAVE_PATH \

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 2 \
    --save_path $SAVE_PATH \

python3 train_faster_rcnn.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 3 \
    --save_path $SAVE_PATH \


# # VARIABLE INFO rpn_R_50_C4_1x
# CONFIG=COCO-Detection/rpn_R_50_C4_1x.yaml
# NAME=rpn_R_50_C4_1x

# python3 train_faster_rcnn.py \
#     --data_path $DATA_PATH \
#     --config_info $CONFIG \
#     --max_iters $MAX_ITERS \
#     --name $NAME \
#     --fold 1 \
#     --save_path $SAVE_PATH \

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



# # VARIABLE INFO rpn_R_50_FPN_1x
# CONFIG=COCO-Detection/rpn_R_50_FPN_1x.yaml
# NAME=rpn_R_50_FPN_1x

# python3 train_faster_rcnn.py \
#     --data_path $DATA_PATH \
#     --config_info $CONFIG \
#     --max_iters $MAX_ITERS \
#     --name $NAME \
#     --fold 1 \
#     --save_path $SAVE_PATH \

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