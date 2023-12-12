import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt

cols = ['AP5095', 'AP50', 'AP75', 'iteration', 'loss_box_reg', 'loss_cls',  'total_loss', 'validation_loss']
titles = ['mAP@50:95', 'mAP@50', 'mAP@75',  'Iteration', 'Box Loss', 'Object Loss', 'Total Loss', 'Validation Loss']
axis = ['mAP', 'mAP', 'mAP',  'Iteration', 'Loss', 'Loss', 'Loss', 'Loss']
title_dict = dict(zip(cols, titles))
y_axis_dict = dict(zip(cols, axis))
x_axis_dict = "No. of Epochs"
model_list = ['faster_rcnn_R_50_DC5_3x']
names = ['ResNet50 with FPN at 3x']
model_dict = dict(zip(model_list, names))

archi = {
    'faster_rcnn': 'faster_rcnn_R_50_DC5_3x'
}



def plot_metric(four_class_path, three_class_path, single_path, output_dir, model):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for column in cols:
        if column == 'iteration' or 'unnamed' in column.lower():
            continue
        plt.figure(figsize=(4,4))
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.grid(alpha=0.5, linestyle='--', linewidth=0.75)


        four_class_df = pd.read_csv(os.path.join(four_class_path, model, 'csvs',archi[model], f'mean_{archi[model]}.csv'), header=0)
        three_class_df = pd.read_csv(os.path.join(three_class_path, model, 'csvs',archi[model], f'mean_{archi[model]}.csv'), header=0)
        single_df = pd.read_csv(os.path.join(single_path, model,  'csvs',archi[model], f'mean_{archi[model]}.csv'), header=0)
        print(four_class_df.head())
        print(three_class_df.head())
        print(single_df.head())



        # plt.title(f'{title_dict[column]} for {model_dict[model]}', fontsize=12, fontweight='bold')
        # plt.title(f'{title_dict[column]} for Faster R-CNN', fontsize=12, fontweight='bold')
        # plt.xlabel(x_axis_dict, fontsize=12, fontweight='bold')
        # plt.ylabel(y_axis_dict[column], fontsize=12, fontweight='bold')
        # plt.xticks(fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.tight_layout()
        # plt.savefig(os.path.join(output_dir, f'{column}.png'), dpi=300)
        # plt.close()

        
    pass

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-fcp','--four_class_path', type=str, default='data')
    argparse.add_argument('-tcp','--three_class_path', type=str, default='data')
    argparse.add_argument('-sp','--single_path', type=str, default='data')
    argparse.add_argument('-o','--output_dir', type=str, default='data')
    argparse.add_argument('-m','--model', type=str, default='faster_rcnn_R_50_DC5_3x')

    args = argparse.parse_args()
    plot_metric(args.four_class_path, args.three_class_path, args.single_path, args.output_dir, args.model)
