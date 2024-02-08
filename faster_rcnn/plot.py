import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def clean(model_name, inpath, outpath):
    print (model_name)
    fold_1 = os.path.join(inpath, f'{model_name}_four_class_fold_1')
    fold_2 = os.path.join(inpath, f'{model_name}_four_class_fold_2')
    fold_3 = os.path.join(inpath, f'{model_name}_four_class_fold_3')

    outpath = os.path.join(outpath, 'plots', model_name)
    json_path = os.path.join(outpath, 'json')
    csv_path = os.path.join(outpath, 'csv')

    print (fold_1)
    print (fold_2)
    print (fold_3)

    print (outpath)
    print (json_path)
    print (csv_path)

    # reading fold-wise json
    fold_1_json = pd.read_json(os.path.join(fold_1, f'metrics.json'))
    fold_2_json = pd.read_json(os.path.join(fold_2, f'metrics.json'))
    fold_3_json = pd.read_json(os.path.join(fold_3, f'metrics.json'))

    # making a dataframe out of the json
    fold_1_df = pd.DataFrame(fold_1_json)
    fold_2_df = pd.DataFrame(fold_2_json)
    fold_3_df = pd.DataFrame(fold_3_json)

    print (fold_1_df)
    print (fold_2_df)
    print (fold_3_df)


    pass

if __name__ == '__main__':
    argparseer = argparse.ArgumentParser()
    argparseer.add_argument('--inpath', type=str, default='../outputs/detectron')
    argparseer.add_argument('--model_name', type=str, default='../data/plot.csv')
    argparseer.add_argument('--output_path', type=str, default='../data/plot.png')
    args = argparseer.parse_args()
    clean(args.model_name, args.inpath, args.output_path)