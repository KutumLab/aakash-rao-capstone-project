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


python3 utils/data_formatter.py \
    -i /media/chs.gpu/19b31863-d2db-44c2-b407-9a4ffbebcbd1/DATA/research-cancerPathology/aakash-rao-capstone-project/datasets/master/NuCLSEvalSet/rgb \
    -m /media/chs.gpu/19b31863-d2db-44c2-b407-9a4ffbebcbd1/DATA/research-cancerPathology/aakash-rao-capstone-project/datasets/master/NuCLSEvalSet/csv \
    -s /media/chs.gpu/19b31863-d2db-44c2-b407-9a4ffbebcbd1/DATA/research-cancerPathology/aakash-rao-capstone-project/datasets/detectron_three_class \
    -mp /media/chs.gpu/19b31863-d2db-44c2-b407-9a4ffbebcbd1/DATA/research-cancerPathology/aakash-rao-capstone-project/hypothesis_1/output/models/Xception_three_class/Xception_three_class.h5 \
    -p run \
    -f 3 \
    -v three_class \
    --seed 42 \