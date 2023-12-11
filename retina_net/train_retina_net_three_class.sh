
# SEMI-CONSTANT INFO
DATA_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/detectron_three_class
SAVE_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs
MAX_ITERS=15000
VERSION=three_class


# VARIABLE INFO
CONFIG=COCO-Detection/retinanet_R_101_FPN_3x.yaml
NAME=retinanet_R_101_FPN_3x

python3 train_retina_net.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 1 \
    --save_path $SAVE_PATH \
    --version $VERSION

python3 train_retina_net.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 2 \
    --save_path $SAVE_PATH \
    --version $VERSION

python3 train_retina_net.py \
    --data_path $DATA_PATH \
    --config_info $CONFIG \
    --max_iters $MAX_ITERS \
    --name $NAME \
    --fold 3 \
    --save_path $SAVE_PATH \
    --version $VERSION

