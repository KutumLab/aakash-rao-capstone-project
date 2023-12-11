INDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron
python3 utils/json_gen.py \
    --path $INDIR 


INDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron
OUTDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/plots/retina_net/csvs
python3 utils/generate_csv.py \
    -d $INDIR \
    -o $OUTDIR \



INDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/plots/retina_net/csvs
OUTDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/plots/retina_net/plots
python3 utils/plot_retina_net.py \
    -d $INDIR \
    -o $OUTDIR

# OLDDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_run_archive/detectron/retina_net/csvs
# NEWDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/plots/retina_net/csvs
# OUTDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/plots/retina_net/comparative
# python3 utils/comparative.py \
#     -d $OLDDIR \
#     -o $OUTDIR \
#     -n $NEWDIR 