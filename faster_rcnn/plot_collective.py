import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from math import sqrt
from matplotlib import rc 


model_name_dict = {
    "faster_rcnn_R_50_C4_1x": "R50: C4 at 1x",
    "faster_rcnn_R_50_DC5_1x": "R50: DC5 at 1x",
    "faster_rcnn_R_50_FPN_1x": "R50: FPN at 1x",
    "faster_rcnn_R_50_C4_3x": "R50: C4 at 3x",
    "faster_rcnn_R_50_DC5_3x": "R50: DC5 at 3x",
    "faster_rcnn_R_50_FPN_3x": "R50: FPN at 3x",
    "faster_rcnn_R_101_C4_3x": "R101: C4 at 3x",
    "faster_rcnn_R_101_DC5_3x": "R101: DC5 at 3x",
    "faster_rcnn_R_101_FPN_3x": "R101: FPN at 3x",
    "faster_rcnn_X_101_32x8d_FPN_3x": "X101: FPN at 3x",
}
plot_col_titles = {
    "bbox/AP":  "Average Precision at IoU 0.50:0.95",
    "bbox/AP50": "Average Precision at IoU 0.50",
    "total_loss": "Total Loss",
    "validation_loss": "Validation Loss",
}
subtitles = {
    "bbox/AP":  "Average Precision at IoU thresholds of 0.50:0.95 across all models and folds",
    "bbox/AP50": "Average Precision at IoU threshold of 0.50 across all models and folds",
    "total_loss": "Total Loss over all iterations across all models and folds",
    "validation_loss": "Validation Loss over all iterations across all models and folds",
}
col_list = ["bbox/AP", "bbox/AP50", "total_loss", "validation_loss"]

axes_titles = {
    "bbox/AP": "Average Precision (AP)",
    "bbox/AP50": "Average Precision (AP)",
    "total_loss": "Loss",
    "validation_loss": "Loss",
}

def clean(model_name, inpath, outpath):
    print (model_name)
    fold_1 = os.path.join(inpath, f'{model_name}_four_class_fold_1')
    fold_2 = os.path.join(inpath, f'{model_name}_four_class_fold_2')
    fold_3 = os.path.join(inpath, f'{model_name}_four_class_fold_3')

    outpath = os.path.join(outpath, 'plots', 'four_class', model_name)
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

   


    df_1 = pd.DataFrame(columns=col_list)
    df_2 = pd.DataFrame(columns=col_list)
    df_3 = pd.DataFrame(columns=col_list)

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
    sem = pd.DataFrame(columns=col_list)
    for col in col_list:
        sem[col] = pd.concat([df_1[col], df_2[col], df_3[col]], axis=1).sem(axis=1).values

    # insert nans in SEM where there are nans in df_1
    for col in col_list:
        sem[col] = sem[col].where(df_1[col].notna(), np.nan)
    sem.to_csv(os.path.join(csv_path, 'sem.csv'), index=False)
    print (sem)

    pass




def plot(outpath, model_names):
    savepath = os.path.join(outpath, 'plots', 'four_class', 'collective')
    os.makedirs(savepath, exist_ok=True)
    for col in col_list:
        fig, ax = plt.subplots(figsize=(5,3))
        for model_name in model_names:
            csv_path = os.path.join(outpath, 'plots', 'four_class', model_name, 'csv')
            mean = pd.read_csv(os.path.join(csv_path, 'mean.csv'))
            sem = pd.read_csv(os.path.join(csv_path, 'sem.csv'))
            mean[col] = mean[col].astype(float)
            sem[col] = sem[col].astype(float)

            col_mean = mean[col].values
            col_sem = sem[col].values

            col_mean = mean[col].dropna()
            x = mean['iteration'].values[col_mean.index]

            col_sem = sem[col].values[col_mean.index]
            print (col_mean)
            ax.plot(x, col_mean, label=f"{model_name_dict[model_name]}", marker='o', markersize=0.001, linewidth=0.2)
            ax.fill_between(x, col_mean - col_sem, col_mean + col_sem, alpha=0.2)
            if "validation_loss" in col:
                if "X_101" in model_name:
                    point_of_min_loss = np.argmin(col_mean)
                    ax.axvline(x[point_of_min_loss], color='r', linestyle='--', linewidth=0.25, label=f"Lowest Validation Loss for \n X 101 with FPN at 3x \n at {int(x[point_of_min_loss])}th iteration")


        ax.set_xlabel('Iterations', fontsize=10, fontweight='bold')
        ax.set_xlim(0, 18001)
        ax.set_xticks(np.arange(0, 18001, 6000), list(map(str, np.arange(0, 18001, 6000))), fontsize=8)

        ax.set_ylabel(axes_titles[col], fontsize=10, fontweight='bold')
        plt.suptitle(f'{plot_col_titles[col]}' , fontsize=14, fontweight='bold')
        ax.set_title(f'{subtitles[col]}' , fontsize=8, fontweight='bold', style='italic')
        col_name = col.replace('/', '_')

        if "AP" in col_name:
            ax.set_ylim(0, 100)
            ax.set_yticks(ticks=np.arange(0, 101, 10), labels=list(map(str, np.arange(0, 101, 10))), fontsize=8)
        elif "accuracy" in col_name or "negative" in col_name:
            ax.set_ylim(0, 1)
            ax.set_yticks(ticks=np.arange(0, 1.1, 0.1), labels=list(map(str, np.arange(0, 1.1, 0.1))), fontsize=8)

        # text title for legend
        # insert custom line in the legend
        ax.legend(bbox_to_anchor=(1.05, 1),  title="Configuration", loc='upper left', fontsize=6, title_fontsize=8, frameon=False)
            

        plt.tight_layout()
        plt.savefig(os.path.join(savepath, f'{col_name}.png'), bbox_inches='tight', dpi=300)


        
    pass


def plot_at_lowest_loss(outpath, model_names):
    savepath = os.path.join(outpath, 'plots', 'four_class', 'collective_at_lowest_loss')
    os.makedirs(savepath, exist_ok=True)
    limit = 6959
    for col in col_list:
        col_df = pd.DataFrame()
        fig, ax = plt.subplots(figsize=(5, 3))
        for model_name in model_names:
            csv_path = os.path.join(outpath, 'plots', 'four_class', model_name, 'csv')
            mean = pd.read_csv(os.path.join(csv_path, 'mean.csv'))
            sem = pd.read_csv(os.path.join(csv_path, 'sem.csv'))
            mean[col] = mean[col].astype(float)
            sem[col] = sem[col].astype(float)

            # only take the first 6959 iterations
            mean = mean[:limit]
            sem = sem[:limit]


            col_mean = mean[col].values
            col_sem = sem[col].values

            col_mean = mean[col].dropna()
            x = mean['iteration'].values[col_mean.index]

            col_sem = sem[col].values[col_mean.index]
            print (col_mean)
            ax.plot(x, col_mean, label=f"{model_name_dict[model_name]}", marker='o', markersize=0.001, linewidth=0.2)
            ax.fill_between(x, col_mean - col_sem, col_mean + col_sem, alpha=0.2)
            col_df[f"{model_name}_mean"] = col_mean
            col_df[f"{model_name}_sem"] = col_sem
        col_df.to_csv(os.path.join(savepath, f"{col.replace('/', '_')}.csv"), index=False)


        ax.set_xlabel('Iterations', fontsize=10)
        ax.set_xlim(0, limit+1)
        ax.set_xticks(np.arange(0, limit+1, limit//3), list(map(str, np.arange(0, limit+1, limit//3))), fontsize=8)

        ax.set_ylabel(axes_titles[col], fontsize=10)

        ax.set_title(f'{plot_col_titles[col]}' , fontsize=14, fontweight='bold')
        col_name = col.replace('/', '_')

        if "AP" in col_name:
            ax.set_ylim(0, 100)
            ax.set_yticks(ticks=np.arange(0, 101, 10), labels=list(map(str, np.arange(0, 101, 10))), fontsize=8)
        elif "accuracy" in col_name or "negative" in col_name:
            ax.set_ylim(0, 1)
            ax.set_yticks(ticks=np.arange(0, 1.1, 0.1), labels=list(map(str, np.arange(0, 1.1, 0.1))), fontsize=8)

        # text title for legend
        # ax.legend(bbox_to_anchor=(1.05, 1),  title="Configuration", loc='upper left', fontsize=6, title_fontsize=8, frameon=False)
            

        plt.tight_layout()
        plt.savefig(os.path.join(savepath, f'{col_name}.png'), bbox_inches='tight', dpi=300)
        plt.close('all')
    pass



# custom plot




if __name__ == '__main__':
    argparseer = argparse.ArgumentParser()
    argparseer.add_argument('--inpath', type=str, default='../outputs/detectron')
    argparseer.add_argument('--output_path', type=str, default='../data/plot.png')
    args = argparseer.parse_args()
    model_names = ["faster_rcnn_R_50_C4_1x", "faster_rcnn_R_50_DC5_1x", "faster_rcnn_R_50_FPN_1x", "faster_rcnn_R_50_C4_3x", "faster_rcnn_R_50_DC5_3x", "faster_rcnn_R_50_FPN_3x", "faster_rcnn_R_101_C4_3x", "faster_rcnn_R_101_DC5_3x", "faster_rcnn_R_101_FPN_3x", "faster_rcnn_X_101_32x8d_FPN_3x"]
    # clean(args.model_name, args.inpath, args.output_path)
    plot(args.output_path, model_names)
    plot_at_lowest_loss(args.output_path, model_names)