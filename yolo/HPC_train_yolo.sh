#! /bin/bash
#PBS -N Four_Class_Faster_RCNN
#PBS -o out_four_class.log
#PBS -e err_four_class.log
#PBS -l ncpus=50
#PBS -q gpu

eval "$(conda shell.bash hook)"
conda activate yolov5


declare -A test_dict
test_dict["yolov5n"]="yolov5n.pt"
test_dict["yolov5s"]="yolov5s.pt"
test_dict["yolov5m"]="yolov5m.pt"
test_dict["yolov5l"]="yolov5l.pt"
test_dict["yolov5x"]="yolov5x.pt"

IMG_SIZE=640
EPOCHS=200
DEVICE=0
BATCH_SIZE=16
SAVE_PERIOD=10
SAVE_PATH=/storage/bic/Aakash/aakash-rao-capstone-project/outputs/yolov5
YAML_PATH=/storage/bic/Aakash/aakash-rao-capstone-project/yolo/configs

models=("yolov5s") # "yolov5m" "yolov5l" "yolov5x")
for model_name in "${models[@]}"
do
    for FOLD in 0 #1 2 3 4
    do
        NAME="$model_name"_"$VERSION" #Specify a name for the model used for saving and logistics
        WEIGHT=$BASE_WEIGHT_PATH/${test_dict[$model_name]}
        PROJECT=$SAVE_PATH/$NAME/fold_$FOLD
        YAML_FOLD=$YAML_PATH/fold_$FOLD.yaml

        echo $NAME
        echo $WEIGHT
        python ./yolov5/train.py \
            --img $IMG_SIZE \
            --epochs $EPOCHS \
            --data $YAML_FOLD \
            --weights yolov5x.pt \
            --device $DEVICE \
            --batch-size $BATCH_SIZE \
            --project $PROJECT \
            --save-period $SAVE_PERIOD \
            --name yolov5x-fold_3 \
            --mode setup
    done
done