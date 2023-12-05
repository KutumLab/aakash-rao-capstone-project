import os
import pandas as pd
import argparse

original_titles = []
translation = []
relevant = []
rename = {}


def generate_csvs(data_dir, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for folder in os.listdir(data_dir):
        metrics = pd.read_csv(os.path.join(data_dir, folder, 'metrics.csv'))
        print(metrics.columns)
        foldr_split = folder.split('_')
        net_name = '_'.join(foldr_split[:-2])
        fold_num = '_'.join(foldr_split[-2:])
        print(net_name, fold_num)
        
        
        break
    pass

def mean_and_std_fold(data_dir, output_dir):
    

    pass

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-d','--data_dir', type=str, default='data')
    argparse.add_argument('-o','--output_dir', type=str, default='data')

    args = argparse.parse_args()
    generate_csvs(args.data_dir, args.output_dir)
    # mean_and_std_fold(args.output_dir, args.output_dir)