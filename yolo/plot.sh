INDIR = /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/yolo
OUTDIR = /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/yolo/csvs
python3 yolo/utils/generate_csvs.py \
    -d $INDIR \
    -o $OUTDIR