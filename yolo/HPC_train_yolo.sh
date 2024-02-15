#! /bin/bash
#PBS -N Four_Class_YOLO
#PBS -o out_four_class.log
#PBS -e err_four_class.log
#PBS -l ncpus=50
#PBS -q cpu

eval "$(conda shell.bash hook)"
conda activate yolov5


declare -A test_dict
test_dict["yolov5n"]="yolov5n.pt"
test_dict["yolov5s"]="yolov5s.pt"
test_dict["yolov5m"]="yolov5m.pt"
test_dict["yolov5l"]="yolov5l.pt"
test_dict["yolov5x"]="yolov5x.pt"
test_dict["yolov5n6"]="yolov5n6.pt"
test_dict["yolov5s6"]="yolov5s6.pt"
test_dict["yolov5m6"]="yolov5m6.pt"
test_dict["yolov5l6"]="yolov5l6.pt"
test_dict["yolov5x6"]="yolov5x6.pt"



# yolov5n
for FOLD in 1 2 3
do
    python /storage/bic/Aakash/aakash-rao-capstone-project/yolo/yolov5/train.py \
        --img 640 \
        --batch 16 \
        --epochs 100 \
        --data /storage/bic/Aakash/aakash-rao-capstone-project/yolo/configs/fold_$FOLD.yaml \
        --weights /storage/bic/Aakash/aakash-rao-capstone-project/yolo/bases/yolov5n.pt \
        --name yolov5n_fold_$FOLD \
        --project /storage/bic/Aakash/aakash-rao-capstone-project/outputs/yolo/ 
done