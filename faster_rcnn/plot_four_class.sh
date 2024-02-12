#! /bin/bash
#PBS -N PLOT_Faster_RCNN
#PBS -o out_plot.log
#PBS -e err_plot.log
#PBS -l ncpus=50
#PBS -q cpu

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
test_dict["faster_rcnn_X_101_32x8d_FPN_3x"]="X101-FPN.pkl"




BASE_PATH=/storage/bic/Aakash/aakash-rao-capstone-project


DATA_PATH=$BASE_PATH/datasets/detectron #Specify a path to your dataset folder
SAVE_PATH=$BASE_PATH/outputs #Specify a path to save the model and other outputs
MAX_ITERS=18000 #Since we want to train for about 200 epochs, we set the number of iterations to num_train_samples * 200 / batch_size
VERSION=four_class #Specify a version name for the model
BATCHSIZE=8
BASE_WEIGHT_PATH=$BASE_PATH/faster_rcnn/bases
#!/bin/bash

# models=("faster_rcnn_R_50_C4_1x" "faster_rcnn_R_50_DC5_1x" "faster_rcnn_R_50_FPN_1x" "faster_rcnn_R_50_C4_3x" "faster_rcnn_R_50_DC5_3x" "faster_rcnn_R_50_FPN_3x" "faster_rcnn_R_101_C4_3x" "faster_rcnn_R_101_DC5_3x" "faster_rcnn_R_101_FPN_3x" "faster_rcnn_X_101_32x8d_FPN_3x")
# for model_name in "${models[@]}"
# do
#     save_path=$SAVE_PATH/plots
#     python /storage/bic/Aakash/aakash-rao-capstone-project/faster_rcnn/plot.py \
#     --inpath $SAVE_PATH/detectron \
#     --model_name $model_name \
#     --output_path $SAVE_PATH \

# done


python /storage/bic/Aakash/aakash-rao-capstone-project/faster_rcnn/plot_collective.py \
    --inpath $SAVE_PATH/detectron \
    --output_path $SAVE_PATH \