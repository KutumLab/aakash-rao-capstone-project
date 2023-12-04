# for loop
for FOLD in 1 2 3 
do
    python3 utils/json_gen.py \
        --path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron/retinanet_R_50_FPN_1x_fold_${FOLD}

    python3 utils/plot_model.py \
        --path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron/retinanet_R_50_FPN_1x_fold_${FOLD} \
        --name "ResNet50+FPN at 1x" 
done

for FOLD in 1 2 3 
do
    python3 utils/json_gen.py \
        --path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron/retinanet_R_50_FPN_3x_fold_${FOLD}

    python3 utils/plot_model.py \
        --path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron/retinanet_R_50_FPN_3x_fold_${FOLD} \
        --name "ResNet50+FPN at 3x" 
done

for FOLD in 1 2 3 
do
    python3 utils/json_gen.py \
        --path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron/retinanet_R_101_FPN_3x_fold_${FOLD}

    python3 utils/plot_model.py \
        --path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron/retinanet_R_101_FPN_3x_fold_${FOLD} \
        --name "ResNet101+FPN at 3x" 
done