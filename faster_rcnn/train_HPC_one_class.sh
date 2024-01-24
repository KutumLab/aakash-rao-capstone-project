#! /bin/bash
#PBS -N One_Class_Faster_RCNN
#PBS -o out_one_class.log
#PBS -e err_one_class.log
#PBS -l ncpus=30
#PBS -q gpu


eval "$(conda shell.bash hook)"
conda activate detectron


declare -A test_dict
test_dict["faster_rcnn_R_50_C4_1x"]="R50-C4-1x.pkl"
test_dict["faster_rcnn_R_50_C4_3x"]="R50-C4-3x.pkl"
test_dict["faster_rcnn_R_50_DC5_1x"]="R50-DC5-1x.pkl"
test_dict["faster_rcnn_R_50_DC5_3x"]="R50-DC5-3x.pkl"
test_dict["faster_rcnn_R_50_FPN_1x"]="R50-FPN-1x.pkl"
test_dict["faster_rcnn_R_50_FPN_3x"]="R50-FPN-3x.pkl"
test_dict["faster_rcnn_R_101_C4_3x"]="R101-C4-3x.pkl"
test_dict["faster_rcnn_R_101_DC5_3x"]="R101-DC5-3x.pkl"
test_dict["faster_rcnn_R_101_FPN_3x"]="R101-FPN-3x.pkl"
test_dict["faster_rcnn_X_101_32x8d_FPN_3x"]="X101-FPN-3x.pkl"



# module load compiler/anaconda3
# source  /home/rintu.kutum/.conda/envs/detectron/bin/activate

BASE_PATH=/storage/bic/Aakash/aakash-rao-capstone-project


DATA_PATH=$BASE_PATH/datasets/detectron_single #Specify a path to your dataset folder
SAVE_PATH=$BASE_PATH/outputs #Specify a path to save the model and other outputs
MAX_ITERS=18000 #Since we want to train for about 200 epochs, we set the number of iterations to num_train_samples * 200 / batch_size
VERSION=single_class #Specify a version name for the model
BATCHSIZE=8
BASE_WEIGHT_PATH=$BASE_PATH/faster_rcnn/bases
#!/bin/bash

# Create an array
model_list=("faster_rcnn_R_50_C4_1x" "faster_rcnn_R_50_DC5_1x" "faster_rcnn_R_50_FPN_1x" "faster_rcnn_R_50_C4_3x" "faster_rcnn_R_50_DC5_3x" "faster_rcnn_R_50_FPN_3x" "faster_rcnn_R_101_C4_3x" "faster_rcnn_R_101_DC5_3x" "faster_rcnn_R_101_FPN_3x" "faster_rcnn_X_101_32x8d_FPN_3x")

# Loop through the array elements
for model_name in "${model_list[@]}"
do
    # VARIABLE INFO "$model_name"_[DONE]
    CONFIG=COCO-Detection/"$model_name".yaml #Specify a config file which is used to source the model from detectron2's model zoo
    NAME="$model_name"_"$VERSION" #Specify a name for the model used for saving and logistics
    WEIGHT=$BASE_WEIGHT_PATH/${test_dict[$model_name]}
    echo $NAME
    echo $CONFIG
    echo $WEIGHT
    python3 /storage/bic/Aakash/aakash-rao-capstone-project/faster_rcnn/train_faster_rcnn.py \
        --data_path $DATA_PATH \
        --config_info $CONFIG \
        --max_iters $MAX_ITERS \
        --name $NAME \
        --fold 1 \
        --save_path $SAVE_PATH \
        --version $VERSION \
        --batch_size $BATCHSIZE \
        --weight_path $WEIGHT

    python3 /storage/bic/Aakash/aakash-rao-capstone-project/faster_rcnn/train_faster_rcnn.py \
        --data_path $DATA_PATH \
        --config_info $CONFIG \
        --max_iters $MAX_ITERS \
        --name $NAME \
        --fold 2 \
        --save_path $SAVE_PATH \
        --version $VERSION \
        --batch_size $BATCHSIZE \
        --weight_path $WEIGHT

    python3 /storage/bic/Aakash/aakash-rao-capstone-project/faster_rcnn/train_faster_rcnn.py \
        --data_path $DATA_PATH \
        --config_info $CONFIG \
        --max_iters $MAX_ITERS \
        --name $NAME \
        --fold 3 \
        --save_path $SAVE_PATH \
        --version $VERSION \
        --batch_size $BATCHSIZE \
        --weight_path $WEIGHT
done