import json
import argparse
import os
import numpy as np
import pandas as pd


def plot_model(path):
    # read json file
    info = pd.read_csv(os.path.join(path, 'metrics.csv'), index_col=False, header=0)
    info_cols = info.columns

    print(info_cols)

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--path', help='path to json file')
    args = argparse.parse_args()
    plot_model(args.path)