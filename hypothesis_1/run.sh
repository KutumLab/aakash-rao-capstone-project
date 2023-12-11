BASE=/Users/mraoaakash/Documents/research/aakash-rao-capstone-project
EPOCHS=100
BATCH_SIZE=200


# python3 utils/organize_data_four_class.py \
#     -i $BASE/datasets/master/EvaluationSet/rgb\
#     -m $BASE/datasets/master/EvaluationSet/csv\
#     -s $BASE/hypothesis_1/datasets/clus

# python3 simple_clus/cluster.py \
    # -s $BASE/hypothesis_1/

# python3 Xception_four_class/train.py \
#     -t $BASE/hypothesis_1/datasets/clus/master\
#     -p $BASE/hypothesis_1/output\
#     -m Xception\
#     -e $EPOCHS\
#     -b $BATCH_SIZE\
#     -l 0.000001\
#     -s _four_class

# python3 utils/organize_data_three_class.py \
#     -i $BASE/datasets/master/EvaluationSet/rgb\
#     -m $BASE/datasets/master/EvaluationSet/csv\
#     -s $BASE/hypothesis_1/datasets/clus_three

python3 Xception_three_class/train.py \
    -t $BASE/hypothesis_1/datasets/clus_three/master\
    -p $BASE/hypothesis_1/output\
    -m Xception\
    -e $EPOCHS\
    -b $BATCH_SIZE\
    -l 0.000001\
    -s _three_class