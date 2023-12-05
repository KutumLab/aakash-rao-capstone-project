import os 
import numpy as np
import pandas as pd
import cv2
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
import time
import json


def plot_stats(image_dir, plot_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    classes = ['Stromal', 'sTIL', 'Tumor', 'Other']
    class_stats = pd.read_csv(os.path.join(image_dir, 'class_stats.csv'), header=0)
    class_stats = class_stats['0'].values.tolist()
    class_stats = np.array(class_stats)
    print(class_stats)
    num_classes_per_image = pd.read_csv(os.path.join(image_dir, 'num_classes_per_image.csv'), header=0)
    num_classes_per_image = num_classes_per_image['len_masks'].values.tolist()
    num_classes_per_image = np.array(num_classes_per_image)

    total_images = class_stats[-1]
    print("Total Images: {}".format(total_images))
    total_annotations = int(class_stats[-2])
    class_stats = class_stats[:-2].astype(int)

    fig_1 = plt.figure(figsize=(4,4))
    plt.bar(np.arange(len(class_stats)), class_stats)
    plt.xticks(np.arange(len(class_stats)), class_stats)
    plt.xlabel('Class', fontsize=14, fontweight='bold')
    plt.ylabel('No of Images', fontsize=14, fontweight='bold')
    plt.xticks(np.arange(len(class_stats)), classes, rotation=45, fontsize=8)
    plt.title('Number of Images per Class', fontsize=14, fontweight='bold')
    fig_1.tight_layout()
    fig_1.savefig(os.path.join(plot_dir, 'num_images_per_class.png'), bbox_inches='tight', dpi=300)
    plt.close()

    fig_2 = plt.figure(figsize=(2,4))
    plt.boxplot(num_classes_per_image, widths=2, showfliers=False)
    plt.ylabel('No of Annotations', fontsize=14, fontweight='bold')
    plt.xlabel('Images', fontsize=14, fontweight='bold')
    plt.title('Annotations Boxplot', fontsize=14, fontweight='bold')
    fig_2.tight_layout()
    fig_2.savefig(os.path.join(plot_dir, 'num_images_per_num_classes.png'), bbox_inches='tight', dpi=300)
    plt.close()


    
    

    


                



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Data Formatter')
    argparser.add_argument('-i', '--image_dir', required=True, help='image directory')
    argparser.add_argument('-p', '--plot_dir', required=True, help='save directory')
    args = argparser.parse_args()

    print("Generating plots for NuCLS Dataset Statistics...")
    plot_stats(args.image_dir, args.plot_dir)