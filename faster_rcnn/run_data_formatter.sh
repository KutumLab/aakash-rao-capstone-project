


# BASE_PATH=/storage/bic/Aakash/aakash-rao-capstone-project

# # multiline run data formatter
# python3 /storage/bic/Aakash/aakash-rao-capstone-project/faster_rcnn/utils/data_formatter.py \
#     -i $BASE_PATH/datasets/master/NuCLSEvalSet/rgb \
#     -m $BASE_PATH/datasets/master/NuCLSEvalSet/csv \
#     -s $BASE_PATH/datasets/detectron \
#     -p run \
#     -f 3 \
#     -v None \
#     --seed 42 \

# python3 /storage/bic/Aakash/aakash-rao-capstone-project/faster_rcnn/utils/data_formatter.py \
#     -i $BASE_PATH/datasets/master/NuCLSEvalSet/rgb \
#     -m $BASE_PATH/datasets/master/NuCLSEvalSet/csv \
#     -s $BASE_PATH/datasets/detectron_single \
#     -p run \
#     -f 3 \
#     -v single \
#     --seed 42 \

# python3 /storage/bic/Aakash/aakash-rao-capstone-project/faster_rcnn/utils/data_formatter.py \
#     -i $BASE_PATH/datasets/master/NuCLSEvalSet/rgb \
#     -m $BASE_PATH/datasets/master/NuCLSEvalSet/csv \
#     -s $BASE_PATH/datasets/detectron_three_class \
#     -mp $BASE_PATH/hypothesis_1/output/models/Xception_three_class/Xception_three_class.h5 \
#     -p run \
#     -f 3 \
#     -v three_class \
#     --seed 42 \


BASE_PATH=/Users/mraoaakash/Documents/research/aakash-rao-capstone-project

python3 utils/data_formatter.py \
    -i $BASE_PATH/datasets/master/EvaluationSet/rgb \
    -m $BASE_PATH/datasets/master/EvaluationSet/csv \
    -s $BASE_PATH/datasets/detectron \
    -p run \
    -f 3 \
    -v None \
    --seed 42 \

python3 utils/data_formatter.py \
    -i $BASE_PATH/datasets/master/EvaluationSet/rgb \
    -m $BASE_PATH/datasets/master/EvaluationSet/csv \
    -s $BASE_PATH/datasets/detectron_single \
    -p run \
    -f 3 \
    -v single \
    --seed 42 \

python3 utils/data_formatter.py \
    -i $BASE_PATH/datasets/master/EvaluationSet/rgb \
    -m $BASE_PATH/datasets/master/EvaluationSet/csv \
    -s $BASE_PATH/datasets/detectron_three_class \
    -mp $BASE_PATH/hypothesis_1/output/models/Xception_three_class/Xception_three_class.h5 \
    -p run \
    -f 3 \
    -v three_class \
    --seed 42 \