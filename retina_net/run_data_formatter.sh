eval "$(conda shell.bash hook)"
conda activate detectron
# multiline run data formatter
python3 utils/data_formatter.py \
    -i /media/chs.gpu/19b31863-d2db-44c2-b407-9a4ffbebcbd1/DATA/research-cancerPathology/aakash-rao-capstone-project/datasets/master/NuCLSEvalSet/rgb \
    -m /media/chs.gpu/19b31863-d2db-44c2-b407-9a4ffbebcbd1/DATA/research-cancerPathology/aakash-rao-capstone-project/datasets/master/NuCLSEvalSet/csv \
    -s /media/chs.gpu/19b31863-d2db-44c2-b407-9a4ffbebcbd1/DATA/research-cancerPathology/aakash-rao-capstone-project/datasets/detectron \
    -p run \
    -f 3 \
    --seed 42 \