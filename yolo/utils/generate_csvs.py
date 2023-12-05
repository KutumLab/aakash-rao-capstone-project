import os
import pandas as pd
import argparse


def generate_csvs(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        fold = folder.split('-')[-1]
        model = folder.split('-')[0]
        model_output_dir = os.path.join(output_dir, model, fold)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        csv = pd.read_csv(os.path.join(folder_path, 'class_stats.csv'), header=0)
        print(csv.head())
        break

        
    pass

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-d','--data_dir', type=str, default='data')
    argparse.add_argument('-o','--output_dir', type=str, default='data')

    args = argparse.parse_args()
    generate_csvs(args.data_dir, args.output_dir)