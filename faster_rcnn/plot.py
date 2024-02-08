import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from math import sqrt

def clean(model_name, inpath, outpath):
    print (model_name)
    fold_1 = os.path.join(inpath, f'{model_name}_four_class_fold_1')
    fold_2 = os.path.join(inpath, f'{model_name}_four_class_fold_2')
    fold_3 = os.path.join(inpath, f'{model_name}_four_class_fold_3')

    outpath = os.path.join(outpath, 'plots', model_name)
    json_path = os.path.join(outpath, 'json')
    csv_path = os.path.join(outpath, 'csv')
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(json_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)

    print (fold_1)
    print (fold_2)
    print (fold_3)

    print (outpath)
    print (json_path)
    print (csv_path)

    # reading fold-wise json
    fold_1_json = []
    with open(os.path.join(fold_1, 'metrics.json')) as f:
        for line in f:
            fold_1_json.append(json.loads(line))

    fold_2_json = []
    with open(os.path.join(fold_2, 'metrics.json')) as f:
        for line in f:
            fold_2_json.append(json.loads(line))

    fold_3_json = []
    with open(os.path.join(fold_3, 'metrics.json')) as f:
        for line in f:
            fold_3_json.append(json.loads(line))

    cols = []
    for i in range(len(fold_1_json)):
        for key in fold_1_json[i].keys():
            cols.append(key)

    cols = np.array(cols)
    cols = np.unique(cols)

    df_1 = pd.DataFrame(columns=cols)
    df_2 = pd.DataFrame(columns=cols)
    df_3 = pd.DataFrame(columns=cols)

    for i in range(len(fold_1_json)):
        df_1 = pd.concat([df_1, pd.DataFrame(fold_1_json[i], index=[0])], ignore_index=True)

    for i in range(len(fold_2_json)):
        df_2 = pd.concat([df_2, pd.DataFrame(fold_2_json[i], index=[0])], ignore_index=True)
    
    for i in range(len(fold_3_json)):
        df_3 = pd.concat([df_3, pd.DataFrame(fold_3_json[i], index=[0])], ignore_index=True)
    
    print (df_1)
    print (df_2)
    print (df_3)

    df_1.to_csv(os.path.join(csv_path, 'fold_1.csv'), index=False)
    df_2.to_csv(os.path.join(csv_path, 'fold_2.csv'), index=False)
    df_3.to_csv(os.path.join(csv_path, 'fold_3.csv'), index=False)

    mean = (df_1 + df_2 + df_3) / 3
    mean.to_csv(os.path.join(csv_path, 'mean.csv'), index=False)
    print (mean)

    # replace nan with 0
    df_1 = df_1.fillna(0)
    df_2 = df_2.fillna(0)
    df_3 = df_3.fillna(0)
    # compute standard error across folds
    sem = pd.DataFrame(columns=cols)
    for col in cols:
        sem[col] = pd.concat([df_1[col], df_2[col], df_3[col]], axis=1).sem(axis=1).values

    # insert nans in SEM where there are nans in df_1
    for col in cols:
        sem[col] = sem[col].where(df_1[col].notna(), np.nan)
    sem.to_csv(os.path.join(csv_path, 'sem.csv'), index=False)
    print (sem)

    pass

def plot(outpath, model_name):
    outpath = os.path.join(outpath, 'plots', model_name)
    csv_path = os.path.join(outpath, 'csv')
    mean = pd.read_csv(os.path.join(csv_path, 'mean.csv'))
    sem = pd.read_csv(os.path.join(csv_path, 'sem.csv'))
    figures_path = os.path.join(outpath, 'figures')
    os.makedirs(figures_path, exist_ok=True)

    for col in mean.columns:
        mean[col] = mean[col].astype(float)
        sem[col] = sem[col].astype(float)

        col_mean = mean[col].values
        col_sem = sem[col].values

        x = mean['iteration'].values
        # find indices of 0.0 values in col_mean
        zero_indices = np.where(col_mean == 0.0)[0]
        # remove 0.0 values from col_mean and col_sem
        col_mean = np.delete(col_mean, zero_indices)
        col_sem = np.delete(col_sem, zero_indices)
        x = np.delete(x, zero_indices)

        fig, ax = plt.subplots()
        ax.plot(x, col_mean, label='Mean', marker='o', markersize=0.001
        ax.fill_between(x, col_mean - col_sem, col_mean + col_sem, alpha=0.2, label='SEM')

        ax.set_ylabel('Iteration')
        ax.set_xlabel(col)
        ax.set_title(f'{model_name} {col}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        col_name = col.replace('/', '_')

        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, f'{col_name}.png'), bbox_inches='tight', dpi=300)


        
    pass

if __name__ == '__main__':
    argparseer = argparse.ArgumentParser()
    argparseer.add_argument('--inpath', type=str, default='../outputs/detectron')
    argparseer.add_argument('--model_name', type=str, default='../data/plot.csv')
    argparseer.add_argument('--output_path', type=str, default='../data/plot.png')
    args = argparseer.parse_args()
    clean(args.model_name, args.inpath, args.output_path)
    plot(args.output_path, args.model_name)