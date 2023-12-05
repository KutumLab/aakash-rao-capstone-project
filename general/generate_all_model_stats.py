import numpy as np
import pandas as pd
import os
import argparse

models = ['faster_rcnn', 'retina_net', 'yolo']
dict_oof = {'yolo':['mAP_'], 'retina_net':['AP'], 'faster_rcnn':['AP']}
def gen_stat(src_path):
    models_list = []
    ap50_list = []
    ap5095_list = []
    for model in os.listdir(src_path):
        if '.gitkeep' in model or 'NuCLS_Stastistics' in model:
            continue
        model_path = os.path.join(src_path, model, "csvs")
        for submodel in os.listdir(model_path):
            if '.gitkeep' in submodel:
                continue
            mean_df = pd.read_csv(os.path.join(model_path, submodel, f'mean_{submodel}.csv'), header=0)
            std_df = pd.read_csv(os.path.join(model_path, submodel, f'std_{submodel}.csv'), header=0)
            models_list.append(model)
            ap50 = mean_df[f'{dict_oof[model][0]}50'].values
            # appending max value
            ap50_list.append(np.max(ap50))
            ap5095 = mean_df[f'{dict_oof[model][0]}5095'].values
            # appending max value
            ap5095_list.append(np.max(ap5095))
    print(models_list)
    print(ap50_list)
    print(ap5095_list)



    pass

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-s','--src_path', type=str, default='data')
    args = argparse.parse_args()
    gen_stat(args.src_path)
    pass