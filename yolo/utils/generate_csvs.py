import os
import pandas as pd
import argparse

original_titles = ['               epoch', '      train/box_loss', '      train/obj_loss','      train/cls_loss', '   metrics/precision', '      metrics/recall','     metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', '        val/box_loss','        val/obj_loss', '        val/cls_loss', '               x/lr0','               x/lr1', '               x/lr2']
translation = ['epoch', 'train/box_loss', 'train/obj_loss','train/cls_loss', 'metrics/precision', 'metrics/recall','metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss','val/obj_loss', 'val/cls_loss', 'x/lr0','x/lr1', 'x/lr2']
relevant = ['epoch', 'train/box_loss', 'train/obj_loss','train/cls_loss', 'metrics/precision', 'metrics/recall','metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss','val/obj_loss', 'val/cls_loss']
rename = {'train/box_loss': 'train_box_loss', 'train/obj_loss': 'train_obj_loss', 'train/cls_loss': 'train_cls_loss', 'metrics/precision': 'precision', 'metrics/recall': 'recall', 'metrics/mAP_0.5': 'mAP_50', 'metrics/mAP_0.5:0.95': 'mAP_5095', 'val/box_loss': 'val_box_loss', 'val/obj_loss': 'val_obj_loss', 'val/cls_loss': 'val_cls_loss'}


def generate_csvs(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        fold = folder.split('-')[-1]
        model = folder.split('-')[0]
        model_output_dir = os.path.join(output_dir, model)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        csv = pd.read_csv(os.path.join(folder_path, 'results.csv'), header=0)
        csv = csv[original_titles]
        csv.columns = translation
        csv = csv[relevant]
        csv = csv.rename(columns=rename)
        csv.to_csv(os.path.join(model_output_dir, f'results_{fold}.csv'), index=False)
        print(csv.columns)
        break

        
    pass

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-d','--data_dir', type=str, default='data')
    argparse.add_argument('-o','--output_dir', type=str, default='data')

    args = argparse.parse_args()
    generate_csvs(args.data_dir, args.output_dir)