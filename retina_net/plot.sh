INDIR=/media/chs.gpu/19b31863-d2db-44c2-b407-9a4ffbebcbd1/DATA/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron
python3 utils/json_gen.py \
    --path $INDIR 


INDIR=/media/chs.gpu/19b31863-d2db-44c2-b407-9a4ffbebcbd1/DATA/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron
OUTDIR=/media/chs.gpu/19b31863-d2db-44c2-b407-9a4ffbebcbd1/DATA/research-cancerPathology/aakash-rao-capstone-project/plots/retina_net/csvs
python3 utils/generate_csv.py \
    -d $INDIR \
    -o $OUTDIR \



INDIR=/media/chs.gpu/19b31863-d2db-44c2-b407-9a4ffbebcbd1/DATA/research-cancerPathology/aakash-rao-capstone-project/plots/retina_net/csvs
OUTDIR=/media/chs.gpu/19b31863-d2db-44c2-b407-9a4ffbebcbd1/DATA/research-cancerPathology/aakash-rao-capstone-project/plots/retina_net/plots
python3 utils/plot_faster_rcnn.py \
    -d $INDIR \
    -o $OUTDIR