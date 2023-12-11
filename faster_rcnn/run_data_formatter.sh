# multiline run data formatter
# python3 utils/data_formatter.py \
#     -i /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/master/NuCLSEvalSet/rgb \
#     -m /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/master/NuCLSEvalSet/csv \
#     -s /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/detectron \
#     -p run \
#     -f 3 \
#     -v None \
#     --seed 42 \

# python3 utils/data_formatter.py \
#     -i /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/master/NuCLSEvalSet/rgb \
#     -m /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/master/NuCLSEvalSet/csv \
#     -s /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/detectron_single \
#     -p run \
#     -f 3 \
#     -v single \
#     --seed 42 \


python3 utils/data_formatter.py \
    -i /Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/master/EvaluationSet/rgb \
    -m /Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/master/EvaluationSet/csv \
    -s /Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/detectron \
    -p run \
    -f 3 \
    -v None \
    --seed 42 \

python3 utils/data_formatter.py \
    -i /Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/master/EvaluationSet/rgb \
    -m /Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/master/EvaluationSet/csv \
    -s /Users/mraoaakash/Documents/research/aakash-rao-capstone-project/datasets/detectron_single \
    -p run \
    -f 3 \
    -v single \
    --seed 42 \