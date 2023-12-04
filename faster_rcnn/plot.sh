# for loop
BASEPATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron


MODEL=faster_rcnn_R_50_C4_1x
for FOLD in 1 2 3 
do
    python3 utils/json_gen.py \
        --path $BASEPATH/"$MODEL"_fold_${FOLD}

    python3 utils/plot_model.py \
        --path $BASEPATH/"$MODEL"_fold_${FOLD} \
        --name "ResNet101 at 1x" 
done



MODEL=faster_rcnn_R_50_C4_3x
for FOLD in 1 2 3 
do
    python3 utils/json_gen.py \
        --path $BASEPATH/"$MODEL"_fold_${FOLD}

    python3 utils/plot_model.py \
        --path $BASEPATH/"$MODEL"_fold_${FOLD} \
        --name "ResNet50 at 1x" 
done

MODEL=faster_rcnn_R_50_DC5_1x
for FOLD in 1 2 3 
do
    python3 utils/json_gen.py \
        --path $BASEPATH/"$MODEL"_fold_${FOLD}

    python3 utils/plot_model.py \
        --path $BASEPATH/"$MODEL"_fold_${FOLD} \
        --name "ResNet50 at 3x" 
done

MODEL=faster_rcnn_R_101_DC5_3x
for FOLD in 1 2 3 
do
    python3 utils/json_gen.py \
        --path $BASEPATH/"$MODEL"_fold_${FOLD}

    python3 utils/plot_model.py \
        --path $BASEPATH/"$MODEL"_fold_${FOLD} \
        --name "ResNet50 at 3x" 
done




# MODEL=retinanet_R_101_FPN_3x
python3 utils/plot_by_model.py \
    --path $BASEPATH \
    --phase run