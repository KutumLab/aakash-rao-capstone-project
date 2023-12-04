import os 
import numpy as np
import argparse
import pandas as pd

model_list = ['faster_rcnn_R_50_C4_1x','faster_rcnn_R_50_C4_3x','faster_rcnn_R_50_DC5_1x','faster_rcnn_R_101_DC5_3x']

def metrics(src_path, phase):
    for model in model_list:
        for fold in range(1,4):
            path = os.path.join(src_path, model+f"_fold_{fold}",)
            results = np.load(os.path.join(path, "results", "results.csv"), allow_pickle=True)
            print(results)
            break


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-s', '--src_path',
        help='path to the source folder',
        default='x'
    )
    argparser.add_argument(
        '-p', '--phase',
        help='phase of the model',
        default='train'
    )
    args = argparser.parse_args()
    metrics(args.src_path, args.phase)