# for loop
for FOLD in 1 2 3 
do
    python3 utils/json_gen.py \
        --path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron/faster_rcnn_R_50_DC5_1x_fold_${FOLD}

    python3 utils/plot_model.py \
        --path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron/faster_rcnn_R_50_DC5_1x_fold_${FOLD} \
        --name "ResNet50+DC5"
done