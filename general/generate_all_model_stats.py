import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt




models = ['faster_rcnn', 'retina_net', 'yolo']
dict_oof = {'yolo':['mAP_'], 'retina_net':['AP'], 'faster_rcnn':['AP']}
def gen_stat(src_path):
    models_list = []
    ap50_list = []
    ap5095_list = []
    for model in os.listdir(src_path):
        if '.gitkeep' in model or 'NuCLS_Stastistics' in model:
            continue
        model_path = os.path.join(src_path, model, "csvs")
        for submodel in os.listdir(model_path):
            if '.gitkeep' in submodel:
                continue
            mean_df = pd.read_csv(os.path.join(model_path, submodel, f'mean_{submodel}.csv'), header=0)
            print(mean_df)
            std_df = pd.read_csv(os.path.join(model_path, submodel, f'std_{submodel}.csv'), header=0)
            models_list.append(submodel)
            ap50 = mean_df[f'{dict_oof[model][0]}50'].dropna(how='all').values
            # appending max value
            ap5095 = mean_df[f'{dict_oof[model][0]}5095'].dropna(how='all').values
            # appending max value
            if 'yo' in model:
                ap50 = ap50*100
                ap5095 = ap5095*100
            ap50_list.append(round(np.max(ap50), 2))
            ap5095_list.append(round(np.max(ap5095), 2))
    print(models_list)
    print(ap50_list)
    print(ap5095_list)

    df = pd.DataFrame({'model':models_list, 'ap50':ap50_list, 'ap5095':ap5095_list})
    print(df.to_latex(index=False))


    pass

best_from_all = ['faster_rcnn_R_50_C4_3x', 'retina_net_R_101_FPN_3x', 'yolov5m']

def grp_plot(src_path):
    mean_frcnn = pd.read_csv(os.path.join(src_path, 'faster_rcnn', 'csvs','faster_rcnn_R_50_C4_3x', 'mean_faster_rcnn_R_50_C4_3x.csv'), header=0)
    mean_frcnn_50 = mean_frcnn['AP50'].dropna(how='all').values
    mean_frcnn_5095 = mean_frcnn['AP5095'].dropna(how='all').values
    mean_retinanet = pd.read_csv(os.path.join(src_path, 'retina_net', 'csvs','retinanet_R_101_FPN_3x', 'mean_retinanet_R_101_FPN_3x.csv'), header=0)
    mean_retinanet_50 = mean_retinanet['AP50'].dropna(how='all').values
    mean_retinanet_5095 = mean_retinanet['AP5095'].dropna(how='all').values
    mean_yolo = pd.read_csv(os.path.join(src_path, 'yolo', 'csvs', 'yolov5m', 'mean_yolov5m.csv'), header=0)
    mean_yolo['mAP_50'] = mean_yolo['mAP_50']*100
    mean_yolo['mAP_50_95'] = mean_yolo['mAP_5095']*100
    mean_yolo['epoch'] = mean_yolo['epoch']*100
    mean_yolo_50 = mean_yolo['mAP_50'].dropna(how='all').values
    mean_yolo_5095 = mean_yolo['mAP_50_95'].dropna(how='all').values
    std_frcnn = pd.read_csv(os.path.join(src_path, 'faster_rcnn', 'csvs','faster_rcnn_R_50_C4_3x', 'std_faster_rcnn_R_50_C4_3x.csv'), header=0)
    std_frcnn_50 = std_frcnn['AP50'].dropna(how='all').values
    std_frcnn_5095 = std_frcnn['AP5095'].dropna(how='all').values
    std_retinanet = pd.read_csv(os.path.join(src_path, 'retina_net', 'csvs','retinanet_R_101_FPN_3x', 'std_retinanet_R_101_FPN_3x.csv'), header=0)
    std_retinanet_50 = std_retinanet['AP50'].dropna(how='all').values
    std_retinanet_5095 = std_retinanet['AP5095'].dropna(how='all').values
    std_yolo = pd.read_csv(os.path.join(src_path, 'yolo', 'csvs', 'yolov5m', 'std_yolov5m.csv'), header=0)
    std_yolo['mAP_50'] = std_yolo['mAP_50']*100
    std_yolo['mAP_50_95'] = std_yolo['mAP_5095']*100
    std_yolo['epoch'] = std_yolo['epoch']*100
    std_yolo_50 = std_yolo['mAP_50'].dropna(how='all').values
    std_yolo_5095 = std_yolo['mAP_50_95'].dropna(how='all').values

    mean_frcnn_50 = mean_frcnn_50[:99]
    mean_retinanet_50 = mean_retinanet_50[:99]
    mean_yolo_50 = mean_yolo_50[:99]
    std_frcnn_50 = std_frcnn_50[:99]
    std_retinanet_50 = std_retinanet_50[:99]
    std_yolo_50 = std_yolo_50[:99]
    
    plt.figure(figsize=(4, 4))
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.grid(alpha=0.5, linestyle='--', linewidth=0.75)
    plt.plot(mean_frcnn_50, label='Faster R-CNN', color='r')
    plt.plot(mean_retinanet_50, label='RetinaNet', color='b')
    plt.plot(mean_yolo_50, label='YOLOv5', color='g')
    plt.fill_between(np.arange(len(mean_frcnn_50)), mean_frcnn_50-std_frcnn_50, mean_frcnn_50+std_frcnn_50, alpha=0.3, color='r')
    plt.fill_between(np.arange(len(mean_retinanet_50)), mean_retinanet_50-std_retinanet_50, mean_retinanet_50+std_retinanet_50, alpha=0.3, color='b')
    plt.fill_between(np.arange(len(mean_yolo_50)), mean_yolo_50-std_yolo_50, mean_yolo_50+std_yolo_50, alpha=0.3, color='g')
    plt.legend(fontsize=10)
    plt.title(f'Best Performers for mAP@50', fontsize=12, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12, fontweight='bold')
    plt.ylabel("Mean AP", fontsize=12, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{src_path}/best_ap50.png',dpi=300)
    plt.close()

    mean_frcnn_5095 = mean_frcnn_5095[:99]
    mean_retinanet_5095 = mean_retinanet_5095[:99]
    mean_yolo_5095 = mean_yolo_5095[:99]
    std_frcnn_5095 = std_frcnn_5095[:99]
    std_retinanet_5095 = std_retinanet_5095[:99]
    std_yolo_5095 = std_yolo_5095[:99]

    plt.figure(figsize=(4, 4))
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.grid(alpha=0.5, linestyle='--', linewidth=0.75)
    plt.plot(mean_frcnn_5095, label='Faster R-CNN', color='r')
    plt.plot(mean_retinanet_5095, label='RetinaNet', color='b')
    plt.plot(mean_yolo_5095, label='YOLOv5', color='g')
    plt.fill_between(np.arange(len(mean_frcnn_5095)), mean_frcnn_5095-std_frcnn_5095, mean_frcnn_5095+std_frcnn_5095, alpha=0.3, color='r')
    plt.fill_between(np.arange(len(mean_retinanet_5095)), mean_retinanet_5095-std_retinanet_5095, mean_retinanet_5095+std_retinanet_5095, alpha=0.3, color='b')
    plt.fill_between(np.arange(len(mean_yolo_5095)), mean_yolo_5095-std_yolo_5095, mean_yolo_5095+std_yolo_5095, alpha=0.3, color='g')
    plt.legend(fontsize=10)
    plt.title(f'Best Performers for mAP@50:95', fontsize=12, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12, fontweight='bold')
    plt.ylabel("Mean AP", fontsize=12, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{src_path}/best_ap5095.png',dpi=300)
    plt.close()





    pass

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-s','--src_path', type=str, default='data')
    args = argparse.parse_args()
    # gen_stat(args.src_path)
    grp_plot(args.src_path)
    pass