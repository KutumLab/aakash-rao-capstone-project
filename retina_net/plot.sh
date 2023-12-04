# for loop
BASEPATH=/media/chs.gpu/19b31863-d2db-44c2-b407-9a4ffbebcbd1/DATA/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron


MODEL=retinanet_R_101_FPN_3x
for FOLD in 1 2 3 
do
    python3 utils/json_gen.py \
        --path $BASEPATH/"$MODEL"_fold_${FOLD}

    # python3 utils/plot_model.py \
    #     --path $BASEPATH/"$MODEL"_fold_${FOLD} \
    #     --name "ResNet50+FPN at 1x" 
done


MODEL=retinanet_R_50_FPN_1x
for FOLD in 1 2 3 
do
    python3 utils/json_gen.py \
        --path $BASEPATH/"$MODEL"_fold_${FOLD}

    # python3 utils/plot_model.py \
    #     --path $BASEPATH/"$MODEL"_fold_${FOLD} \
    #     --name "ResNet50+FPN at 3x" 
done

MODEL=retinanet_R_50_FPN_3x
for FOLD in 1 2 3 
do
    python3 utils/json_gen.py \
        --path $BASEPATH/"$MODEL"_fold_${FOLD}

    # python3 utils/plot_model.py \
    #     --path $BASEPATH/"$MODEL"_fold_${FOLD} \
    #     --name "ResNet101+FPN at 3x" 
done