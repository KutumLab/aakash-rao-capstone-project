import os
import pandas as pd
import argparse

original_titles = ['bbox/AP', 'bbox/AP-nonTIL_stromal', 'bbox/AP-other', 'bbox/AP-sTIL','bbox/AP-tumor_any', 'bbox/AP50', 'bbox/AP75', 'bbox/APl', 'bbox/APm','bbox/APs', 'data_time', 'eta_seconds', 'fast_rcnn/cls_accuracy','fast_rcnn/false_negative', 'fast_rcnn/fg_cls_accuracy', 'iteration','loss_box_reg', 'loss_cls', 'loss_rpn_cls', 'loss_rpn_loc', 'lr','rank_data_time', 'roi_head/num_bg_samples', 'roi_head/num_fg_samples','rpn/num_neg_anchors', 'rpn/num_pos_anchors', 'time', 'timetest','total_loss', 'validation_loss']
relevant = ['bbox/AP', 'bbox/AP50', 'bbox/AP75', 'fast_rcnn/cls_accuracy', 'iteration','loss_box_reg', 'loss_cls',  'total_loss', 'validation_loss']
translation = ['AP5095', 'AP50', 'AP75', 'cls_accuracy', 'iteration', 'loss_box_reg', 'loss_cls',  'total_loss', 'validation_loss']
rename = {'bbox/AP': 'AP5095', 'bbox/AP50': 'AP50', 'bbox/AP75': 'AP75', 'fast_rcnn/cls_accuracy': 'cls_accuracy', 'iteration': 'iteration', 'loss_box_reg': 'loss_box_reg', 'loss_cls': 'loss_cls',  'total_loss': 'total_loss', 'validation_loss': 'validation_loss'}



def generate_csvs(data_dir, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for folder in os.listdir(data_dir):
        metrics = pd.read_csv(os.path.join(data_dir, folder, 'metrics.csv'))
        metrics = metrics[relevant]
        metrics = metrics.rename(columns=rename)
        print(metrics.columns)
        foldr_split = folder.split('_')
        net_name = '_'.join(foldr_split[:-2])
        fold_num = '_'.join(foldr_split[-2:])
        outpath = os.path.join(output_dir, net_name,"folds")
        if not os.path.exists(os.path.join(outpath)):
            os.makedirs(os.path.join(outpath))
        metrics.to_csv(os.path.join(outpath, "results_"+fold_num + '.csv'))
            
        print(net_name, fold_num)
    pass

def mean_and_std_fold(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for model in os.listdir(data_dir):
        out_dir = os.path.join(output_dir, model)
        fold_1 = pd.read_csv(os.path.join(data_dir, model, "folds", f'results_fold_1.csv'), header=0)
        fold_2 = pd.read_csv(os.path.join(data_dir, model, "folds", f'results_fold_2.csv'), header=0)
        fold_3 = pd.read_csv(os.path.join(data_dir, model, "folds", f'results_fold_3.csv'), header=0)
        print(f'{model} & fold 1 & {round(fold_1["AP50"].max(),2)} & {round(fold_1["AP5095"].max(),2)} \\\\')
        print(f'{model} & fold 2 & {round(fold_2["AP50"].max(),2)} & {round(fold_2["AP5095"].max(),2)} \\\\')
        print(f'{model} & fold 3 & {round(fold_3["AP50"].max(),2)} & {round(fold_3["AP5095"].max(),2)} \\\\')

        mean_df = pd.DataFrame(columns=fold_1.columns)
        std_df = pd.DataFrame(columns=fold_1.columns)
        mean_df['iteration'] = fold_1['iteration']
        std_df['iteration'] = fold_1['iteration']
        for column in fold_1.columns:
            if column == 'iteration':
                continue
            mean_df[column] = (fold_1[column] + fold_2[column] + fold_3[column]) / 3
            std_array = [fold_1[column], fold_2[column], fold_3[column]]
            std_df[column] = pd.concat(std_array, axis=1).std(axis=1)

        mean_df.to_csv(os.path.join(out_dir, f'mean_{model}.csv'), index=False)
        std_df.to_csv(os.path.join(out_dir, f'std_{model}.csv'), index=False)

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-d','--data_dir', type=str, default='data')
    argparse.add_argument('-o','--output_dir', type=str, default='data')

    args = argparse.parse_args()
    generate_csvs(args.data_dir, args.output_dir)
    mean_and_std_fold(args.output_dir, args.output_dir)