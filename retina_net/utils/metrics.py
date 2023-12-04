import os 
import numpy as np
import argparse
import math
import pandas as pd

model_list = ['retinanet_R_50_FPN_1x','retinanet_R_50_FPN_3x','retinanet_R_101_FPN_3x']
name_key = {'retinanet_R_50_FPN_1x': 'RetinaNet R50 1x', 'retinanet_R_50_FPN_3x': 'RetinaNet R50 3x', 'retinanet_R_101_FPN_3x': 'RetinaNet R101 3x'}

def metrics(src_path, phase):
    meanap50 = []
    meanap75 = []
    meanap = []
    for model in model_list:
        sumap50 = 0
        sumap75 = 0
        sumap = 0
        for fold in range(1,4):
            path = os.path.join(src_path, model+f"_fold_{fold}",)
            results = np.load(os.path.join(path, "results", "results.npy"), allow_pickle=True)
            # convert to dict
            results = results.tolist()
            results = pd.DataFrame(results)
            # print(results)
            results = results['bbox'].values
            sumap50 += results[5]
            sumap75 += results[6]
            sumap += results[0]
        # print(f"Model: {model}")
        print(f"{name_key[model]} & {round(sumap50/3,2)} & {round(sumap75/3,2)} & {round(sumap/3,2)} \\\\")


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