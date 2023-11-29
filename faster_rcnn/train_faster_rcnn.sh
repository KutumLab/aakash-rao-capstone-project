python3 train_faster_rcnn.py \
    --data_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/detectron \
    --config_info configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yaml \
    --max_iters 20000 \
    --name faster_rcnn_r50_fpn_1x_coco \
    --fold 1 \
    --project capstone-project \
    --save_path /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs \

