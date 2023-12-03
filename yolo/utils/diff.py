import os

l1 = os.listdir('/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/yolov5/fold_1/train/images')
l2 = os.listdir('/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/yolov5/test/images')

diff = set(l1). difference(l2) 
print(diff)