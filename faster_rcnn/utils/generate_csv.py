import os
import pandas as pd
import argparse

original_titles = ['bbox/AP', 'bbox/AP-nonTIL_stromal', 'bbox/AP-other', 'bbox/AP-sTIL','bbox/AP-tumor_any', 'bbox/AP50', 'bbox/AP75', 'bbox/APl', 'bbox/APm','bbox/APs', 'data_time', 'eta_seconds', 'fast_rcnn/cls_accuracy','fast_rcnn/false_negative', 'fast_rcnn/fg_cls_accuracy', 'iteration','loss_box_reg', 'loss_cls', 'loss_rpn_cls', 'loss_rpn_loc', 'lr','rank_data_time', 'roi_head/num_bg_samples', 'roi_head/num_fg_samples','rpn/num_neg_anchors', 'rpn/num_pos_anchors', 'time', 'timetest','total_loss', 'validation_loss']
relevant = ['bbox/AP', 'bbox/AP50', 'bbox/AP75', 'fast_rcnn/cls_accuracy', 'iteration','loss_box_reg', 'loss_cls',  'total_loss', 'validation_loss']
translation = ['AP5095', 'AP50', 'AP75', 'cls_accuracy', 'iteration', 'loss_box_reg', 'loss_cls',  'total_loss', 'validation_loss']
rename = {original_titles[i]: translation[i] for i in range(len(relevant))}



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