DATA_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/detectron
SAVE_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/mmdetection

python3 ./utils/run_data_formatter.py \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH \ 