DATA_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/detectron
SAVE_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/mmdetection

python3 ./utils/data_formatter.py \
    --data_path $DATA_PATH \
    --outp_dir $SAVE_PATH \
    --fold 1 \