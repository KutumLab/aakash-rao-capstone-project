#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate detectron

BASE_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project

# For Four-Class training 
# PID: 1298623
# [1] 1298623
DATA_PATH=$BASE_PATH/datasets/detectron #Specify a path to your dataset folder
SAVE_PATH=$BASE_PATH/outputs #Specify a path to save the model and other outputs
MAX_ITERS=1 #18000 #Since we want to train for about 200 epochs, we set the number of iterations to num_train_samples * 200 / batch_size
VERSION=four_class #Specify a version name for the model
BATCHSIZE=8
#!/bin/bash

# Create an array
model_list=("faster_rcnn_R_50_C4_1x" "faster_rcnn_R_50_DC5_1x" "faster_rcnn_R_50_FPN_1x" "faster_rcnn_R_50_C4_3x" "faster_rcnn_R_50_DC5_3x" "faster_rcnn_R_50_FPN_3x" "faster_rcnn_R_101_C4_3x" "faster_rcnn_R_101_DC5_3x" "faster_rcnn_R_101_FPN_3x" "faster_rcnn_X_101_32x8d_FPN_3x")

# Loop through the array elements
for model_name in "${model_list[@]}"
do
    CONFIG=COCO-Detection/"$model_name".yaml #Specify a config file which is used to source the model from detectron2's model zoo
    NAME="$model_name"_"$VERSION" #Specify a name for the model used for saving and logistics
    echo $NAME
    echo $CONFIG
    python3 initialize.py \
        --data_path $DATA_PATH \
        --config_info $CONFIG \
        --max_iters $MAX_ITERS \
        --name $NAME \
        --fold 1 \
        --save_path $SAVE_PATH \
        --version $VERSION \
        --batch_size $BATCHSIZE
    # VARIABLE INFO "$model_name"_[DONE]
    # CONFIG=COCO-Detection/"$model_name".yaml #Specify a config file which is used to source the model from detectron2's model zoo
    # NAME="$model_name"_"$VERSION" #Specify a name for the model used for saving and logistics
    # echo $NAME
    # echo $CONFIG
    # python3 train_faster_rcnn.py \
    #     --data_path $DATA_PATH \
    #     --config_info $CONFIG \
    #     --max_iters $MAX_ITERS \
    #     --name $NAME \
    #     --fold 1 \
    #     --save_path $SAVE_PATH \
    #     --version $VERSION \
    #     --batch_size $BATCHSIZE

    # python3 train_faster_rcnn.py \
    #     --data_path $DATA_PATH \
    #     --config_info $CONFIG \
    #     --max_iters $MAX_ITERS \
    #     --name $NAME \
    #     --fold 2 \
    #     --save_path $SAVE_PATH \
    #     --version $VERSION \
    #     --batch_size $BATCHSIZE

    # python3 train_faster_rcnn.py \
    #     --data_path $DATA_PATH \
    #     --config_info $CONFIG \
    #     --max_iters $MAX_ITERS \
    #     --name $NAME \
    #     --fold 3 \
    #     --save_path $SAVE_PATH \
    #     --version $VERSION \
    #     --batch_size $BATCHSIZE
done




# # For Three-Class training 

# DATA_PATH=$BASE_PATH/datasets/detectron_three_class #Specify a path to your dataset folder
# SAVE_PATH=$BASE_PATH/outputs #Specify a path to save the model and other outputs
# MAX_ITERS=18000 #Since we want to train for about 200 epochs, we set the number of iterations to num_train_samples * 200 / batch_size
# VERSION=three_class #Specify a version name for the model
# BATCHSIZE=8
# #!/bin/bash

# # Create an array
# model_list=("faster_rcnn_R_50_C4_1x" "faster_rcnn_R_50_DC5_1x" "faster_rcnn_R_50_FPN_1x" "faster_rcnn_R_50_C4_3x" "faster_rcnn_R_50_DC5_3x" "faster_rcnn_R_50_FPN_3x" "faster_rcnn_R_101_C4_3x" "faster_rcnn_R_101_DC5_3x" "faster_rcnn_R_101_FPN_3x" "faster_rcnn_X_101_32x8d_FPN_3x")

# # Loop through the array elements
# for model_name in "${model_list[@]}"
# do
#     # VARIABLE INFO "$model_name"_[DONE]
#     CONFIG=COCO-Detection/"$model_name".yaml #Specify a config file which is used to source the model from detectron2's model zoo
#     NAME="$model_name"_"$VERSION" #Specify a name for the model used for saving and logistics
#     echo $NAME
#     echo $CONFIG
#     python3 train_faster_rcnn.py \
#         --data_path $DATA_PATH \
#         --config_info $CONFIG \
#         --max_iters $MAX_ITERS \
#         --name $NAME \
#         --fold 1 \
#         --save_path $SAVE_PATH \
#         --version $VERSION \
#         --batch_size $BATCHSIZE

#     python3 train_faster_rcnn.py \
#         --data_path $DATA_PATH \
#         --config_info $CONFIG \
#         --max_iters $MAX_ITERS \
#         --name $NAME \
#         --fold 2 \
#         --save_path $SAVE_PATH \
#         --version $VERSION \
#         --batch_size $BATCHSIZE

#     python3 train_faster_rcnn.py \
#         --data_path $DATA_PATH \
#         --config_info $CONFIG \
#         --max_iters $MAX_ITERS \
#         --name $NAME \
#         --fold 3 \
#         --save_path $SAVE_PATH \
#         --version $VERSION \
#         --batch_size $BATCHSIZE
# done






# # For Single-Class training 
# DATA_PATH=$BASE_PATH/datasets/detectron_single #Specify a path to your dataset folder
# SAVE_PATH=$BASE_PATH/outputs #Specify a path to save the model and other outputs
# MAX_ITERS=18000 #Since we want to train for about 200 epochs, we set the number of iterations to num_train_samples * 200 / batch_size
# VERSION=single_class #Specify a version name for the model
# BATCHSIZE=8
# #!/bin/bash

# # Create an array
# model_list=("faster_rcnn_R_50_C4_1x" "faster_rcnn_R_50_DC5_1x" "faster_rcnn_R_50_FPN_1x" "faster_rcnn_R_50_C4_3x" "faster_rcnn_R_50_DC5_3x" "faster_rcnn_R_50_FPN_3x" "faster_rcnn_R_101_C4_3x" "faster_rcnn_R_101_DC5_3x" "faster_rcnn_R_101_FPN_3x" "faster_rcnn_X_101_32x8d_FPN_3x")

# # Loop through the array elements
# for model_name in "${model_list[@]}"
# do
#     # VARIABLE INFO "$model_name"_[DONE]
#     CONFIG=COCO-Detection/"$model_name".yaml #Specify a config file which is used to source the model from detectron2's model zoo
#     NAME="$model_name"_"$VERSION" #Specify a name for the model used for saving and logistics
#     echo $NAME
#     echo $CONFIG
#     python3 train_faster_rcnn.py \
#         --data_path $DATA_PATH \
#         --config_info $CONFIG \
#         --max_iters $MAX_ITERS \
#         --name $NAME \
#         --fold 1 \
#         --save_path $SAVE_PATH \
#         --version $VERSION \
#         --batch_size $BATCHSIZE

#     python3 train_faster_rcnn.py \
#         --data_path $DATA_PATH \
#         --config_info $CONFIG \
#         --max_iters $MAX_ITERS \
#         --name $NAME \
#         --fold 2 \
#         --save_path $SAVE_PATH \
#         --version $VERSION \
#         --batch_size $BATCHSIZE

#     python3 train_faster_rcnn.py \
#         --data_path $DATA_PATH \
#         --config_info $CONFIG \
#         --max_iters $MAX_ITERS \
#         --name $NAME \
#         --fold 3 \
#         --save_path $SAVE_PATH \
#         --version $VERSION \
#         --batch_size $BATCHSIZE
# done