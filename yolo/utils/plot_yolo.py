import os
import pandas as pd
import argparse


def plot_model(data_dir, plot_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for model in os.listdir(data_dir):
        model_plot_dir = os.path.join(plot_dir, model)
        if not os.path.exists(model_plot_dir):
            os.makedirs(model_plot_dir)
            
        mean_df = pd.read_csv(os.path.join(data_dir, model, f'mean_{model}.csv'), header=0)
        std_df = pd.read_csv(os.path.join(data_dir, model, f'std_{model}.csv'), header=0)
        print(mean_df.columns)
        print(std_df.columns)
        break
    pass


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-d','--data_dir', type=str, default='data')
    argparse.add_argument('-o','--output_dir', type=str, default='data')

    args = argparse.parse_args()
    plot_model(args.data_dir, args.output_dir)